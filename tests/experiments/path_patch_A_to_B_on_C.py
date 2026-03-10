import argparse
import json
import os
import sys
import torch
from tqdm import tqdm
from functools import partial

# 确保可以从父目录导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from transformer_lens import HookedTransformer, utils, ActivationCache
from transformer_lens.hook_points import HookPoint
from eap_auto.tests.experiments.ioi_dataset import IOIDataset

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def patch_or_freeze_head_vectors(
    orig_head_vector: torch.Tensor,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: tuple[int, int],
):
    """
    与 precompute_causal_new.py 中完全相同的钩子函数。
    """
    orig_head_vector[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return orig_head_vector

def main():
    parser = argparse.ArgumentParser(description="Path patch from head A to B and measure the effect on head C's attention pattern.")
    parser.add_argument("--sender_head", type=str, required=True, help="Sender head (A) in format 'layer.head', e.g., '9.9'")
    parser.add_argument("--intermediate_head", type=str, required=True, help="Intermediate head (B) in format 'layer.head', e.g., '10.0'")
    parser.add_argument("--target_head", type=str, required=True, help="Target head (C) whose attention is measured, e.g., '10.7'")
    parser.add_argument("--receiver_input", type=str, default="v", choices=["q", "k", "v"], help="Input to patch on the intermediate head (B). Default is 'v'.")
    parser.add_argument("--input_file", type=str, default="../../results/ioi/path_patching/structured_sentences.jsonl", help="Path to the structured sentences file.")
    parser.add_argument("--output_dir", type=str, default="../../results/ioi/A_to_B_on_C_analysis", help="Directory to save the output analysis file.")
    # 【新功能】: 添加新的命令行标志
    parser.add_argument(
        "--save_end_token_only", 
        action="store_true", 
        help="If set, only save the attention vector from the final token position."
    )
    args = parser.parse_args()

    sender_layer, sender_h_idx = map(int, args.sender_head.split('.'))
    intermediate_layer, intermediate_h_idx = map(int, args.intermediate_head.split('.'))
    target_layer, target_h_idx = map(int, args.target_head.split('.'))

    path_str = f"A{args.sender_head}_to_B{args.intermediate_head}_on_C{args.target_head}"
    print(f"--- Analyzing Path: {path_str} ---")

    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
    model.cfg.use_attn_result = True

    try:
        with open(args.input_file, 'r') as f:
            sentences_data = [json.loads(line) for line in f if 'sentence_text' in line and line.strip()]
        print(f"成功加载 {len(sentences_data)} 个有效句子。")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 加载或解析输入文件失败: {e}")
        return

    full_results = []
    for sentence_info in tqdm(sentences_data, desc="Processing Sentences"):
        clean_text = sentence_info["sentence_text"]
        # 生成对应的损坏文本
        io_token = sentence_info['io_token']
        s_token = sentence_info['s_token']
        corrupted_text = clean_text.replace(io_token, s_token) # 简单的替换规则，可根据需要调整

        # 直接处理文本，更鲁棒
        clean_tokens = model.to_tokens(clean_text)
        corrupted_tokens = model.to_tokens(corrupted_text)
        str_tokens = model.to_str_tokens(clean_tokens)
        
        # --- 核心路径修补逻辑 ---
        
        z_name_filter = lambda name: name.endswith("z")
        _, clean_cache = model.run_with_cache(clean_tokens, names_filter=z_name_filter)
        _, corrupted_cache = model.run_with_cache(corrupted_tokens, names_filter=z_name_filter)

        hook_fn_sender = partial(
            patch_or_freeze_head_vectors,
            new_cache=corrupted_cache,
            orig_cache=clean_cache,
            head_to_patch=(sender_layer, sender_h_idx),
        )
        model.add_hook(z_name_filter, hook_fn_sender)

        receiver_input_hook_name_B = utils.get_act_name(args.receiver_input, intermediate_layer)
        _, patched_B_input_cache = model.run_with_cache(
            clean_tokens, names_filter=lambda name: name == receiver_input_hook_name_B
        )
        model.reset_hooks()

        patched_C_pattern_cache = {}
        def cache_target_pattern_hook(activation, hook):
            patched_C_pattern_cache[hook.name] = activation.detach()

        def patch_intermediate_input_hook(activation, hook):
            activation[:, :, intermediate_h_idx, :] = patched_B_input_cache[hook.name][:, :, intermediate_h_idx, :]
            return activation

        pattern_hook_name_C = utils.get_act_name("pattern", target_layer)
        model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[
                (receiver_input_hook_name_B, patch_intermediate_input_hook),
                (pattern_hook_name_C, cache_target_pattern_hook)
            ],
            return_type=None
        )
        patched_pattern_C = patched_C_pattern_cache[pattern_hook_name_C][0, target_h_idx]

        _, clean_C_cache = model.run_with_cache(clean_tokens, names_filter=lambda name: name == pattern_hook_name_C)
        clean_pattern_C = clean_C_cache[pattern_hook_name_C][0, target_h_idx]

        diff_pattern_C = patched_pattern_C - clean_pattern_C
        
        # --- 【新功能】: 根据标志选择性保存 ---
        result_entry = {
            "sentence_id": sentence_info["sentence_id"],
            "path": path_str,
            "sentence_info": sentence_info,
            "tokens": str_tokens,
        }

        if args.save_end_token_only:
            end_idx = -1 # 最后一个 token
            
            clean_end_attn_vector = clean_pattern_C[end_idx, :]
            patched_end_attn_vector = patched_pattern_C[end_idx, :]
            diff_end_attn_vector = diff_pattern_C[end_idx, :]

            result_entry.update({
                "end_token_position": len(str_tokens) - 1,
                "end_token_clean_attention": clean_end_attn_vector.cpu().tolist(),
                "end_token_patched_attention": patched_end_attn_vector.cpu().tolist(),
                "end_token_diff_attention": diff_end_attn_vector.cpu().tolist()
            })
        else:
            result_entry.update({
                "clean_attention_pattern": clean_pattern_C.cpu().tolist(),
                "patched_attention_pattern": patched_pattern_C.cpu().tolist(),
                "diff_attention_pattern": diff_pattern_C.cpu().tolist()
            })
        
        full_results.append(result_entry)

    # ... 保存文件的代码 ...
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f"analysis_{path_str}.json"
    output_path = os.path.join(args.output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2)

    print("\n--- 分析完成 ---")
    print(f"完整结果已保存至: {output_path}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()