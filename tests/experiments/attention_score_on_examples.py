from attention_score_by_head import run
import asyncio
import sys
import json
import os
import re
import math
import argparse
    

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run attention score analysis on examples.")
    parser.add_argument("--layer", type=int, required=True, help="Layer number")
    parser.add_argument("--head", type=int, required=True, help="Head number")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory path") 
    args = parser.parse_args()
    layer = args.layer
    head = args.head
    output_dir = args.output_dir
    # 打印传入的参数
    print(f"Layer: {layer}, Head: {head}")
    print(f"Output directory: {output_dir}")
    # 从当前路径/data63/private/chensiyuan/EAP-IG/tests/experiments/attention_score_on_examples.py索引到/data63/private/chensiyuan/EAP-IG/results/ioi/hypothesis/{layer}.{head}，不要含有绝对路径
    # output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ioi", "hypothesis", f"{layer}.{head}")
    sentence_path = os.path.join(output_dir, "..", "sentences", "generated_sentences.json")
    with open(sentence_path, "r") as f:
        data = json.load(f)

    run(
        layer=layer,
        head=head,
        output_dir=output_dir,
        sequence = data,
        picture_mode = False,
        outputfile = "raw_model_prompt_attention_scores.jsonl"
    )

if __name__ == "__main__":
    main()