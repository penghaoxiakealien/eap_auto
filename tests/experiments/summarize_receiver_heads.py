#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize multiple head explanations into a compact group description."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from api import OpenRouter


def timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def initialize_openrouter(model: str = "claude-sonnet-4-20250514-thinking") -> OpenRouter:
    """Initialize OpenRouter client (shared defaults with auto_s_att/auto_NMH)."""
    api_key = "sk-99F0IFe53pSHOPQ3phWbEAEx86ZDOqkE58Ov9aYCS9AOQ2C7"
    return OpenRouter(model=model, api_key=api_key)


def _build_prompt(head_descriptions: List[Tuple[str, str]], group_label: str, task: str) -> Tuple[str, str]:
    task = (task or "ioi").lower()
    if task == "garden":
        system_prompt = (
            "**--- Key Concepts in the Garden Path NP/Z v-trans Task (Crucial Background) ---**\n"
            "You must know these linguistic roles in the context of a sentence. The disambiguation token at the end may be hidden.\n"
            "- **Subordinator:** clause introducer (e.g., 'As', 'While').\n"
            "- **Subject (SUBJ):** agent of the main clause.\n"
            "- **Ambiguous Verb (VERB):** verb whose transitivity creates NP/Z ambiguity.\n"
            "- **Object NP Head (OBJ_HEAD):** head noun of the ambiguous object phrase.\n"
            "- **Relative Pronoun / Verb (REL_PRON / REL_VERB):** the relative clause following the object.\n"
            "- **Disambiguation Position (END):** final token that signals the correct parse.\n"
            "Heads can both attend to a token and suppress/promote it downstream; attention pattern and causal effect are distinct.\n"
        )
    else:
        system_prompt = (
            "**--- Key Concepts in the IOI Task (Crucial Background) ---**\n"
            "To understand the head behaviors, you must know these linguistic roles in the context of a sentence. "
            "In an IOI task, we may not provide the whole sentence, which means that IO at the end of the sentence may be hidden for prediction.\n"
            "- **Subject (S):** The one performing the action.\n"
            "- **Indirect Object (IO):** The recipient of the action.\n"
            "- **Direct Object (DO):** The object being transferred.\n"
            "Heads can both attend to a token and suppress/promote it downstream; attention pattern and causal effect are distinct.\n"
        )

    desc_lines = []
    for head, desc in head_descriptions:
        desc_lines.append(f"- **Head {head}**: {desc}")
    desc_block = "\n".join(desc_lines)

    user_prompt = (
        "You are summarizing a group of attention-head explanations for the Garden Path NP/Z v-trans task.\n"
        f"Group label: {group_label}\n\n"
        "Your goal is to produce a compact group summary that:\n"
        "1) captures shared **attention patterns** (what tokens/roles are attended), and\n"
        "2) captures shared **causal effects** (what tokens/roles are promoted or suppressed).\n"
        "If there are meaningful differences within the group, mention them in one short sentence.\n"
        "Keep it to 3-5 sentences.\n\n"
        "**Head explanations:**\n"
        f"{desc_block}\n\n"
        "**Response Format (Strict):**\n"
        "[SUMMARY]: <your 3-5 sentence summary>\n"
    )
    return system_prompt, user_prompt


def _extract_summary(text: str) -> str:
    match = re.search(r"\[SUMMARY\]:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


async def summarize_head_group(
    open_router: OpenRouter,
    head_descriptions: Dict[str, str],
    heads: List[str],
    group_label: str,
    output_dir: str,
    task: str = "ioi",
) -> Optional[str]:
    chosen = [(h, head_descriptions[h]) for h in heads if h in head_descriptions and head_descriptions[h]]
    if len(chosen) <= 1:
        return None

    timestamp = timestamp_suffix()
    run_dir = Path(output_dir) / "receiver_summaries" / f"{group_label}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    system_prompt, user_prompt = _build_prompt(chosen, group_label, task)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = await open_router.generate(messages=messages, output_dir=str(run_dir))
    summary = _extract_summary(response.text)

    payload = {
        "group_label": group_label,
        "heads": [h for h, _ in chosen],
        "summary": summary,
        "source_descriptions": {h: d for h, d in chosen},
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize multiple head explanations into a compact group summary.")
    p.add_argument("--descriptions-json", type=Path, required=True, help="JSON mapping head->description.")
    p.add_argument("--heads", type=str, required=True, help="Comma-separated head list to summarize.")
    p.add_argument("--group-label", type=str, default="receiver_heads", help="Label for this head group.")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory for logs and summary.")
    p.add_argument("--model", type=str, default="claude-sonnet-4-20250514-thinking")
    return p.parse_args()


async def _main() -> None:
    args = parse_args()
    descriptions = json.loads(args.descriptions_json.read_text())
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    open_router = initialize_openrouter(args.model)
    summary = await summarize_head_group(
        open_router=open_router,
        head_descriptions=descriptions,
        heads=heads,
        group_label=args.group_label,
        output_dir=str(args.output_dir),
        task="ioi",
    )
    if summary is None:
        print("No summary generated (not enough head descriptions).")
    else:
        print(summary)


if __name__ == "__main__":
    asyncio.run(_main())
