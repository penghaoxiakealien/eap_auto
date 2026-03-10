#!/usr/bin/env python3
"""
读取多个 head 的 best_hypothesis.json，将其汇总后喂给 LLM，生成群组级别的名称与共同解释。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api import OpenRouter

DEFAULT_MODEL = "claude-sonnet-4-20250514-thinking"
DEFAULT_API_KEY = "sk-ssCRXZzlj8qhPNs6Ps2BxTXZQXq97vJvKATpFXdwYV0E0gUO"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a group of heads with help from an LLM.")
    parser.add_argument(
        "--best-files",
        nargs="+",
        required=True,
        help="一组 best_hypothesis.json 的路径。",
    )
    parser.add_argument(
        "--group-label",
        type=str,
        default="Unnamed Group",
        help="可选：预设的群组标签，将写入提示中。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenRouter 模型名称。",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API Key，默认读取环境变量 OPENROUTER_API_KEY。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="汇总结果输出目录。默认写入 results/ioi/hypothesis/Group_Summary/<timestamp>",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="提示允许的最大输出 token 数。",
    )
    return parser.parse_args()


def initialize_openrouter(model: str, provided_key: str | None) -> OpenRouter:
    """与 auto_s_att.py 等脚本保持一致的 OpenRouter 初始化逻辑。"""
    api_key = provided_key or os.environ.get("OPENROUTER_API_KEY") or DEFAULT_API_KEY
    if not api_key:
        raise RuntimeError("缺少 OpenRouter API Key，请设置环境变量或通过 --api-key 指定。")
    return OpenRouter(model=model, api_key=api_key)


def load_entries(paths: List[str]) -> List[Dict]:
    entries: List[Dict] = []
    for path_str in paths:
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        data = json.loads(path.read_text())
        head = data.get("head") or path.parent.name
        typename = data.get("typename", "")
        hypothesis = data.get("best_hypothesis") or data.get("hypothesis") or ""
        if not hypothesis:
            raise ValueError(f"{path} does not include best_hypothesis/hypothesis field")
        scores = data.get("validation_scores") or data.get("scores") or {}
        entries.append(
            {
                "path": str(path),
                "head": head,
                "typename": typename,
                "scores": {
                    "causal_f1": scores.get("causal_f1"),
                    "attention_f1": scores.get("attention_f1")
                    or scores.get("direct_attention_f1"),
                },
                "hypothesis": hypothesis.strip(),
            }
        )
    return entries


def build_prompt(label: str, samples: List[Dict]) -> str:
    sections = []
    for idx, sample in enumerate(samples, start=1):
        causal = sample["scores"].get("causal_f1")
        attn = sample["scores"].get("attention_f1")
        score_line = f"causal_f1={causal:.2f}" if causal is not None else "causal_f1=NA"
        score_line += ", "
        score_line += f"attention_f1={attn:.2f}" if attn is not None else "attention_f1=NA"
        sections.append(
            f"[Head {idx}] id={sample['head']} ({sample['typename']})\n"
            f"Scores: {score_line}\n"
            f"Hypothesis:\n{sample['hypothesis']}\n"
        )

    instructions = """
You are an interpretability researcher. Read the hypotheses for each head and:
1. Summarize the shared behavior/IOI role/mechanism across these heads.
2. List any notable differences (if none, return an empty array).
3. Propose a concise English group name (e.g., “Name Mover Head”).

Output strict JSON with fields:
{
  "group_name": "...",
  "shared_behavior": "...",
  "key_points": ["...", "..."],
  "notable_differences": ["..."],
  "heads_covered": [
    {"head": "10.7", "role": "..."},
    ...
  ]
}
"""

    body = "\n".join(sections)
    prompt = f"Group label: {label}\n{instructions}\n=== HEAD HYPOTHESES ===\n{body}\n=== END ==="
    return prompt


async def summarize(entries: List[Dict], args: argparse.Namespace, output_dir: Path) -> Dict:
    open_router = initialize_openrouter(args.model, args.api_key)
    prompt = build_prompt(args.group_label, entries)

    messages = [
        {
            "role": "system",
            "content": "You are an expert model interpretability assistant who writes concise JSON outputs.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    response = await open_router.generate(
        messages=messages,
        output_dir=str(output_dir),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    text = response.text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"raw_response": text}
    return parsed


def get_default_output_dir(repo_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return repo_root / "results" / "ioi" / "hypothesis" / "Group_Summary" / timestamp


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or get_default_output_dir(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_entries(args.best_files)
    entries_path = output_dir / "input_heads.json"
    entries_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2))

    summary = asyncio.run(summarize(entries, args, output_dir))
    summary_path = output_dir / "group_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"✅ 汇总完成，结果已保存到 {summary_path}")


if __name__ == "__main__":
    main()
