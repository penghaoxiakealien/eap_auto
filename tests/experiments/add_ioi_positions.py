#!/usr/bin/env python3
import argparse
import json
import os
import sys

from transformers import AutoTokenizer  # type: ignore

# Ensure repo root on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests.experiments.ioi_dataset import IOIDataset  # type: ignore


def load_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add IOI positions (S1/S2/IO/END) to structured_sentences.jsonl."
    )
    parser.add_argument("--input", required=True, help="Input structured_sentences.jsonl")
    parser.add_argument("--output", required=True, help="Output jsonl with positions")
    parser.add_argument(
        "--tokenizer-path",
        default="/data31/private/wangziran/eap-ig/gpt2",
        help="Local GPT-2 tokenizer path",
    )
    parser.add_argument(
        "--prompt-type",
        default="mixed",
        help="Prompt type passed to IOIDataset (used for indexing only)",
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    if not records:
        raise ValueError(f"No records found in {args.input}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    for rec in records:
        io_token = (rec.get("io_token") or "").strip()
        s_token = (rec.get("s_token") or "").strip()
        text = rec.get("sentence_text") or ""
        if not io_token or not s_token or not text:
            raise ValueError("Missing io_token/s_token/sentence_text in input records.")
        prompts.append(
            {
                "text": text,
                "IO": io_token,
                "S": s_token,
                "TEMPLATE_IDX": 0,
            }
        )

    dataset = IOIDataset(
        prompt_type=args.prompt_type,
        N=len(prompts),
        tokenizer=tokenizer,
        prompts=prompts,
        prepend_bos=False,
        device="cpu",
    )

    io_idx = dataset.word_idx["IO"].tolist()
    s1_idx = dataset.word_idx["S1"].tolist()
    s2_idx = dataset.word_idx["S2"].tolist()
    end_idx = dataset.word_idx["end"].tolist()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            rec = dict(rec)
            text = rec.get("sentence_text") or ""
            if not text:
                raise ValueError("Missing sentence_text in input records.")
            # For our truncated IOI sentences (ending at "to"), we want END to be
            # the last token in the sentence, not the IOIDataset 'end' heuristic.
            end_override = len(tokenizer.encode(text, add_special_tokens=False)) - 1
            rec["positions"] = {
                "io": int(io_idx[i]),
                "s1": int(s1_idx[i]),
                "s2": int(s2_idx[i]),
                "end": int(end_override),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records with positions to {args.output}")


if __name__ == "__main__":
    main()
