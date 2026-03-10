import torch as t
import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse

from transformer_lens import HookedTransformer

LOCAL_MODEL_DIR = "/data31/private/wangziran/eap-ig/gpt2"


def load_model(device: str = "cuda"):
    if os.path.isdir(LOCAL_MODEL_DIR):
        print(f"🔥 正在从本地缓存加载模型: {LOCAL_MODEL_DIR}")
        return HookedTransformer.from_pretrained("gpt2", device=device, cache_dir=LOCAL_MODEL_DIR)
    print("⚠️ 未找到本地模型目录，回退到默认的 gpt2-small。")
    return HookedTransformer.from_pretrained("gpt2-small", device=device)


def load_input_data(input_file: str):
    path = Path(input_file)
    text = path.read_text().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "samples" in data:
            return [
                {
                    "sentence_id": sample.get("sample_id", idx),
                    "sentence_text": sample["clean"]["sentence"],
                    "io_token": sample["clean"].get("io_token", ""),
                    "s_token": sample["clean"].get("s_token", ""),
                    "positions": sample.get("positions", {}),
                }
                for idx, sample in enumerate(data["samples"])
            ]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    sentences = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sentences.append(json.loads(line))
    return sentences


def compute_attention_for_sample(model, sample_data, target_head, attention_position: str = "end"):
    clean_sentence = sample_data["sentence_text"]
    io_token = sample_data.get("io_token", "")
    s_token = sample_data.get("s_token", "")

    tokens = model.to_tokens(clean_sentence, prepend_bos=False)[0].unsqueeze(0).to("cuda")
    str_tokens = model.to_str_tokens(tokens[0])

    positions = sample_data.get("positions", {})
    attention_position = (attention_position or "end").lower()
    if attention_position in positions and isinstance(positions.get(attention_position), int):
        target_pos = positions.get(attention_position)
    else:
        target_pos = positions.get("end", len(str_tokens) - 1)
    actual_end_pos = min(int(target_pos), tokens.shape[1] - 1)

    layer, head_idx = target_head
    attn_hook_name = f"blocks.{layer}.attn.hook_pattern"

    with t.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: name == attn_hook_name)

    attention_scores = cache[attn_hook_name][0, head_idx, actual_end_pos, :].cpu()

    token_analysis = []
    for pos, score in enumerate(attention_scores):
        if pos <= actual_end_pos:
            token_analysis.append({
                "token": str_tokens[pos],
                "position": pos,
                "score": score.item()
            })
    token_analysis.sort(key=lambda x: x["score"], reverse=True)

    return {
        "sample_id": sample_data.get("sentence_id"),
        "sentence_text": clean_sentence,
        "io_token": io_token,
        "s_token": s_token,
        "target_head": f"{layer}.{head_idx}",
        "end_position": actual_end_pos,
        "top_attended_tokens": token_analysis
    }


def main():
    parser = argparse.ArgumentParser(description="从structured_sentences或standard_ioi_data计算原始注意力分数。")
    parser.add_argument("--input_file", type=str, default="structured_sentences.jsonl", help="输入的IOI数据文件。可为 JSON 或 JSONL。")
    parser.add_argument("--output_file", type=str, required=True, help="输出的原始注意力分数JSON文件。")
    parser.add_argument("--head", type=str, default="9.6", help="要分析的目标头 (格式: layer.head)。")
    parser.add_argument("--max_samples", type=int, default=None, help="处理的最大样本数（默认全部）。")
    parser.add_argument("--attention-position", type=str, default="end", help="注意力位置（如 end/s1/s2/io/io1/io2）。")

    args = parser.parse_args()

    layer, head_idx = map(int, args.head.split('.'))
    target_head = (layer, head_idx)

    print(f"--- 第1步: 为头 {args.head} 计算原始注意力分数 ---")

    model = load_model()

    print(f"📊 加载数据: {args.input_file}")
    samples = load_input_data(args.input_file)
    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"将处理 {len(samples)} 个样本...")

    all_results = []
    for sample in tqdm(samples, desc="计算注意力"):
        if not sample.get("sentence_text"):
            continue
        result = compute_attention_for_sample(model, sample, target_head, args.attention_position)
        all_results.append(result)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 原始注意力分数计算完成！结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()
