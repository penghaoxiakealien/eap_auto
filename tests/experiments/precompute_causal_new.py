import torch
import json
import os
from tqdm import tqdm
from functools import partial
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from transformer_lens import HookedTransformer, utils
from eap_auto.tests.experiments.ioi_dataset import IOIDataset
import einops
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint

# 使用您原始脚本中经过验证的、正确的路径修补逻辑
def patch_or_freeze_head_vectors(
    orig_head_vector: torch.Tensor,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: tuple[int, int],
):
    orig_head_vector[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return orig_head_vector


def calculate_attention_difference(model, ioi_dataset, abc_dataset, sender_head, receiver_head, receiver_input="q"):
    sender_layer, sender_h_idx = sender_head
    receiver_layer, receiver_h_idx = receiver_head

    z_name_filter = lambda name: name.endswith("z")
    pattern_name_filter = lambda name: name == utils.get_act_name("pattern", receiver_layer)
    
    _, clean_cache = model.run_with_cache(ioi_dataset.toks, names_filter=lambda name: z_name_filter(name) or pattern_name_filter(name))
    _, corrupted_cache = model.run_with_cache(abc_dataset.toks, names_filter=z_name_filter)

    clean_receiver_pattern = clean_cache[utils.get_act_name("pattern", receiver_layer)][:, receiver_h_idx, :, :]

    hook_fn_sender = partial(
        patch_or_freeze_head_vectors,
        new_cache=corrupted_cache,
        orig_cache=clean_cache,
        head_to_patch=(sender_layer, sender_h_idx),
    )
    model.add_hook(z_name_filter, hook_fn_sender)

    receiver_input_hook_name = utils.get_act_name(receiver_input, receiver_layer)
    receiver_input_filter = lambda name: name == receiver_input_hook_name
    _, patched_receiver_input_cache = model.run_with_cache(
        ioi_dataset.toks, names_filter=receiver_input_filter
    )
    model.reset_hooks()

    patched_pattern_cache = {}
    def cache_patched_pattern_hook(activation, hook):
        patched_pattern_cache[hook.name] = activation
        return activation

    def patch_receiver_input_hook(activation, hook):
        activation[:, :, receiver_h_idx] = patched_receiver_input_cache[hook.name][:, :, receiver_h_idx]
        return activation

    model.run_with_hooks(
        ioi_dataset.toks,
        fwd_hooks=[
            (receiver_input_filter, patch_receiver_input_hook),
            (pattern_name_filter, cache_patched_pattern_hook)
        ],
        return_type=None
    )

    patched_receiver_pattern = patched_pattern_cache[utils.get_act_name("pattern", receiver_layer)][:, receiver_h_idx, :, :]

    avg_clean_pattern = einops.reduce(clean_receiver_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
    avg_patched_pattern = einops.reduce(patched_receiver_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
    
    return avg_patched_pattern - avg_clean_pattern, avg_clean_pattern, avg_patched_pattern


def get_attention_summary(attention_vector_tensor: torch.Tensor, tokens: list[str]) -> dict:
    """
    从给定的注意力向量中提取最大/次大/最小/次小值及其token。
    """
    if len(tokens) < 2: # 增加一个保护，防止token太少无法排序
        return {"error": "Not enough tokens to generate summary."}
    sorted_values, sorted_indices = torch.sort(attention_vector_tensor)
    
    return {
        "max_token": tokens[sorted_indices[-1]],
        "max_value": sorted_values[-1].item(),
        "second_max_token": tokens[sorted_indices[-2]],
        "second_max_value": sorted_values[-2].item(),
        "min_token": tokens[sorted_indices[0]],
        "min_value": sorted_values[0].item(),
        "second_min_token": tokens[sorted_indices[1]],
        "second_min_value": sorted_values[1].item(),
    }
    

def main():
    """
    主函数，对指定文件中的每个句子计算因果路径的注意力变化，并仅保存END位置的差异向量。
    """
    # --- 配置 ---
    SENDER_HEADS = [(0, 1), (0, 10), (3, 0)]  # Duplicate Token Heads
    RECEIVER_HEADS = [(7, 3), (7, 9), (8, 6), (8, 10)] # S-inhibition Heads
    
    # --- 【已修正】文件路径配置 ---
    # 所有输出都将保存在 DTH 子目录中
    BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ioi", "path_patching", "DTH")
    SENTENCE_FILE = os.path.join(os.path.dirname(BASE_DIR), "structured_sentences.jsonl")
    
    OUTPUT_FILE = os.path.join(BASE_DIR, "causal_effects_dup_to_sinhibit.json")
    SUMMARY_OUTPUT_FILE = os.path.join(BASE_DIR, "causal_effects_summary_dup_to_sinhibit.json")
    ATTENTION_SUMMARY_OUTPUT_FILE = os.path.join(BASE_DIR, "attention_vectors_summary.json")
    
    # --- 初始化模型 ---
    print("Loading model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
    model.cfg.use_attn_result = True

    # --- 加载句子 ---
    print(f"Loading sentences from {SENTENCE_FILE}...")
    sentences_data = []
    with open(SENTENCE_FILE, "r") as f:
        for line in f:
            sentences_data.append(json.loads(line))

    all_causal_effects = {}
    attention_summary_data = {}
            
    # --- 遍历句子 ---
    for sentence_info in tqdm(sentences_data, desc="Processing Sentences"):
        sentence_id = sentence_info["sentence_id"]
        sentence_text = sentence_info["sentence_text"]
        
        ioi_dataset = IOIDataset(prompt_type="mixed", N=1, tokenizer=model.tokenizer, prepend_bos=False, device="cuda")
        ioi_dataset.ioi_prompts = [{'text': sentence_text, 'IO': sentence_info['io_token'], 'S': sentence_info['s_token'], 'TEMPLATE_IDX': 0}]
        
        abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")

        all_texts = [sentence_text, abc_dataset.ioi_prompts[0]['text']]
        all_tokens = model.tokenizer(all_texts, padding='longest', return_tensors='pt', add_special_tokens=False).input_ids.to("cuda")
        ioi_dataset.toks = all_tokens[0:1]
        abc_dataset.toks = all_tokens[1:2]
        
        str_tokens = model.to_str_tokens(ioi_dataset.toks[0])
        
        all_causal_effects[sentence_id] = {"sentence": sentence_text, "tokens": str_tokens, "diff_vectors": {}}
        attention_summary_data[sentence_id] = {"sentence": sentence_text, "tokens": str_tokens, "paths": {}}
        
        # --- 遍历因果路径 ---
        for sender_head in SENDER_HEADS:
            for receiver_head in RECEIVER_HEADS:
                if sender_head[0] >= receiver_head[0]:
                    continue

                avg_diff_pattern, avg_clean_pattern, avg_patched_pattern = calculate_attention_difference(
                    model, ioi_dataset, abc_dataset, sender_head, receiver_head, receiver_input="q"
                )
                
                path_key = f"{sender_head[0]}.{sender_head[1]}->{receiver_head[0]}.{receiver_head[1]}"
                
                diff_vector = avg_diff_pattern[-1].cpu().tolist()
                all_causal_effects[sentence_id]["diff_vectors"][path_key] = diff_vector

                clean_attention_vector = avg_clean_pattern[-1].cpu()
                patched_attention_vector = avg_patched_pattern[-1].cpu()

                attention_summary_data[sentence_id]["paths"][path_key] = {
                    "clean_attention": get_attention_summary(clean_attention_vector, str_tokens),
                    "patched_attention": get_attention_summary(patched_attention_vector, str_tokens)
                }

    # 写入文件
    os.makedirs(BASE_DIR, exist_ok=True)
    
    print(f"\nSaving full causal effects to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_causal_effects, f, indent=2)
    
    print(f"Generating and saving causal effects summary to {SUMMARY_OUTPUT_FILE}...")
    summary_data = {}
    for sentence_id, data in tqdm(all_causal_effects.items(), desc="Generating Diff Summary"):
        tokens = data["tokens"]
        summary_data[sentence_id] = {"sentence": data["sentence"], "paths": {}}
        for path_key, diff_vector in data["diff_vectors"].items():
            if not diff_vector: continue
            max_change_idx = diff_vector.index(max(diff_vector))
            min_change_idx = diff_vector.index(min(diff_vector))
            
            summary_data[sentence_id]["paths"][path_key] = {
                "max_positive_change_token": tokens[max_change_idx],
                "max_positive_change_value": max(diff_vector),
                "max_negative_change_token": tokens[min_change_idx],
                "max_negative_change_value": min(diff_vector)
            }
            
    with open(SUMMARY_OUTPUT_FILE, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Saving attention vectors summary to {ATTENTION_SUMMARY_OUTPUT_FILE}...")
    with open(ATTENTION_SUMMARY_OUTPUT_FILE, "w") as f:
        json.dump(attention_summary_data, f, indent=2)
        
    print("\nAll tasks complete.")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()