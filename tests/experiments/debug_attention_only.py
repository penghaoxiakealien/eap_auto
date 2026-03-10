#!/usr/bin/env python3
import argparse
import json
import re
from typing import Tuple

from api import OpenRouter


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def extract_target_line(text: str, sid: str) -> str:
    target = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("`") and line.endswith("`"):
            line = line[1:-1].strip()
        if line.startswith(f"{sid}:"):
            target = line
            break
    if not target:
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith("`") and line.endswith("`"):
                line = line[1:-1].strip()
            if re.search(r"\\bioi_\\d+:", line):
                target = line
                break
    return target or text


def count_markers(line: str) -> Tuple[int, int, int]:
    inc = len(re.findall(r"<<[^<>]+>>", line))
    dec = len(re.findall(r"\\[\\[[^\\[\\]]+\\]\\]", line))
    return inc + dec, inc, dec


def build_prompt(hypothesis: str, sid: str, sentence_text: str, top_k: int) -> str:
    return (
        "You are a meticulous AI researcher tasked with predicting the direct attention pattern of a specific attention head.\n"
        "You are given a hypothesis about the head's function and a sentence. Your task is to predict which tokens the head will attend to most strongly from the final token position of the sentence.\n\n"
        f"**Your Task & Guidelines:**\n"
        "1.  **Analyze the Hypothesis:** Carefully read the provided hypothesis.\n"
        "2.  **Apply to Sentence:** Apply this hypothesis to the given sentence.\n"
        "3.  **Predict Top-K Tokens:** Identify the **Top {top_k}** tokens.\n"
        "4.  **Mark the Tokens:** You MUST highlight the most attended token with `<<token>>` and the second most attended token with `[[token]]`.\n\n"
        "**MUST FOLLOW RULES:**\n"
        "- **Strict Formatting:** Your response MUST include a `[PREDICTION]` line, followed by exactly one line containing the marked sentence.\n"
        "- **Exact Count:** You MUST predict and mark exactly {top_k} tokens.\n"
        "- **Single-Token Markers Only:** Each << >> or [[ ]] must wrap exactly ONE token. Do NOT wrap phrases.\n"
        "- **Single-Line Output:** Output exactly one sentence line after `[PREDICTION]`.\n"
        "- **No Extra Text:** Do not include any reasoning or extra lines.\n\n"
        f"**Hypothesis:** {hypothesis}\n"
        f"**Sentence to Analyze:**\n`{sid}: {sentence_text}`\n\n"
        f"**Top_K to Predict:** {top_k}\n\n"
        "**Your Prediction:**"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug attention-only prediction formatting.")
    parser.add_argument("--hypothesis", required=True, help="Hypothesis string.")
    parser.add_argument("--sid", required=True, help="Sentence id, e.g., ioi_0053")
    parser.add_argument("--sentence", required=True, help="Sentence text.")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--model", default="claude-sonnet-4-20250514-thinking")
    args = parser.parse_args()

    client = OpenRouter(model=args.model, api_key=None)
    prompt = build_prompt(args.hypothesis, args.sid, args.sentence, args.top_k)

    system_prompt = (
        "You must output exactly ONE line containing the marked sentence and nothing else. "
        "Do not include reasoning, analysis, or extra lines."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

    for i in range(1, args.rounds + 1):
        resp = client.generate(messages=messages, output_dir=None)
        raw = resp.text.strip()
        stripped = strip_think_blocks(raw)
        pred_block = re.findall(r"\\[PREDICTION\\]\\s*(.*)", stripped, re.DOTALL | re.IGNORECASE)
        if pred_block:
            stripped = pred_block[-1].strip()
        line = extract_target_line(stripped, args.sid)
        total, inc, dec = count_markers(line)
        print("=" * 80)
        print(f"Round {i}")
        print("RAW:", raw[:500].replace("\\n", "\\\\n"))
        print("LINE:", line)
        print(f"MARKERS total={total} inc={inc} dec={dec}")


if __name__ == "__main__":
    main()
