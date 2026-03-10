#!/usr/bin/env python
import argparse
import asyncio
import json
import os
import random

from auto_NMH import (
    initialize_openrouter,
    load_preprocessed_attention_dataset,
    load_preprocessed_causal_dataset,
    build_sentences_from_ids,
    build_causal_examples_from_ids,
    get_real_attention_pattern_for_sampling,
    format_examples_and_tokens,
    underscore_important_tokens,
    convert_predict_attention_to_json,
    compare_attention_f1,
    evaluate_causal_predictions,
    predict_causal_effects_for_sentences,
    chunk_list,
)


def parse_args():
    p = argparse.ArgumentParser(description="Debug NMH batch prediction pipeline with a small sample.")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--head", type=int, required=True)
    p.add_argument("--data-source-dir", type=str, required=True)
    p.add_argument("--hypothesis-file", type=str, required=True, help="Path to best_hypothesis.json or a plain text file.")
    p.add_argument("--sample-size", type=int, default=5)
    p.add_argument("--mode", choices=["causal", "attention", "both"], default="both")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, required=True, help="Debug output directory for prompt/raw logs.")
    return p.parse_args()


def load_hypothesis(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            hyp = data.get("best_hypothesis") or data.get("hypothesis")
            if hyp:
                return str(hyp)
    except json.JSONDecodeError:
        pass
    return text


async def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    hypothesis = load_hypothesis(args.hypothesis_file)
    sender_head = (args.layer, args.head)

    preprocessed_attention_path = os.path.join(args.data_source_dir, "preprocessed_attention_scores.json")
    preprocessed_causal_path = os.path.join(args.data_source_dir, f"causal_effects_{args.layer}_{args.head}.json")

    attn_data = load_preprocessed_attention_dataset(preprocessed_attention_path)
    causal_data = load_preprocessed_causal_dataset(preprocessed_causal_path)

    common_ids = [sid for sid in causal_data.keys() if sid in attn_data]
    if not common_ids:
        raise RuntimeError("No overlapping sentence ids between attention and causal datasets.")

    random.seed(args.seed)
    sample_ids = random.sample(common_ids, min(args.sample_size, len(common_ids)))
    sentences = build_sentences_from_ids(sample_ids, attn_data)
    print(f"Sampled ids: {sample_ids}")

    open_router = initialize_openrouter(model="claude-sonnet-4-20250514-thinking")

    if args.mode in ("causal", "both"):
        print("\n=== Causal (batch) ===")
        causal_preds = await predict_causal_effects_for_sentences(
            open_router, hypothesis, sentences, sender_head, args.output_dir
        )
        causal_gt = build_causal_examples_from_ids(sample_ids, causal_data, sentences)
        causal_f1, causal_feedback = evaluate_causal_predictions(causal_preds, causal_gt, sender_head)
        print(f"Causal F1: {causal_f1:.4f}")
        print(causal_feedback)

    if args.mode in ("attention", "both"):
        print("\n=== Attention (chunk=5) ===")
        real_attention = get_real_attention_pattern_for_sampling(sentences, attn_data)
        attn_preds = []
        for chunk in chunk_list(real_attention, 5):
            formatted = format_examples_and_tokens(chunk)
            highlighted = await underscore_important_tokens(
                open_router, args.layer, args.head, hypothesis, formatted, args.output_dir
            )
            attn_preds.extend(convert_predict_attention_to_json(highlighted, chunk))
        attn_f1, _ = compare_attention_f1(attn_preds, real_attention)
        print(f"Attention F1: {attn_f1:.4f}")
        print("Predictions:", json.dumps(attn_preds, ensure_ascii=False, indent=2))

    print(f"\nDebug logs written to: {args.output_dir}")
    print("Check prompt/raw logs in prompt.txt and raw_api_responses.jsonl")


if __name__ == "__main__":
    asyncio.run(main())

