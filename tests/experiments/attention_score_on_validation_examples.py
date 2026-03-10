from attention_score_by_head import run
import json
import os
import argparse


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run attention score analysis on examples.")
    parser.add_argument("--heads", type=str, required=True, help="Head number")
    parser.add_argument("--typename", type=str, required=True, help="Type name")
    args = parser.parse_args()

    specific_heads = args.heads.split()
    typename = args.typename
    # 打印传入的参数
    print(f"Heads: {specific_heads}")
    print(f"Type of Heads: {typename}")
    # 创建文件夹

    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ioi", "hypothesis", typename)
    os.makedirs(output_dir, exist_ok=True)
    sentence_path = os.path.join(output_dir, "..", "sentences", "validate_sentences.json")
    with open(sentence_path, "r") as f:
        data = json.load(f)
    for specific_head in specific_heads:
        layer, head= map(int, specific_head.split("."))
        head_dir = os.path.join(output_dir, specific_head)
        run(
            layer=layer,
            head=head,
            output_dir=head_dir,
            sequence =data,
            picture_mode=False,
            outputfile="raw_model_prompt_attention_scores.jsonl"
        )
    # common_sentences=filter_sentences(specific_heads, output_dir,token_num=token_num)
    # print(f"Common sentences: {common_sentences}")
    # with open(output_path, "w") as f:
    #     json.dump(list(common_sentences), f, indent=4)
    # print(f"Common sentences saved to {output_path}")

if __name__ == "__main__":
    main()