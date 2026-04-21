import torch
import json
import os
from tqdm import tqdm
from functools import partial
import sys

# 依赖项与路径设置
sys.path.append("/home/wangziran/eap_auto/") 
from tests.experiments.ioi_dataset import IOIDataset
from transformer_lens import HookedTransformer, utils
import einops
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint

# --- Path Patching 核心函数 (保持不变) ---
def patch_or_freeze_head_vectors(
    orig_head_vector: torch.Tensor, hook: HookPoint, new_cache: ActivationCache,
    orig_cache: ActivationCache, head_to_patch: tuple[int, int]
):
    orig_head_vector[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return orig_head_vector

def calculate_attention_difference(model, ioi_dataset, abc_dataset, sender_head, receiver_head):
    sender_layer, sender_h_idx = sender_head
    receiver_layer, receiver_h_idx = receiver_head
    z_name_filter = lambda name: name.endswith("z")
    pattern_name_filter = lambda name: name == utils.get_act_name("pattern", receiver_layer)
    
    _, clean_cache = model.run_with_cache(ioi_dataset.toks, names_filter=lambda name: z_name_filter(name) or pattern_name_filter(name))
    _, corrupted_cache = model.run_with_cache(abc_dataset.toks, names_filter=z_name_filter)
    clean_receiver_pattern = clean_cache[utils.get_act_name("pattern", receiver_layer)][:, receiver_h_idx, :, :]

    hook_fn_sender = partial(patch_or_freeze_head_vectors, new_cache=corrupted_cache, orig_cache=clean_cache, head_to_patch=(sender_layer, sender_h_idx))
    model.add_hook(z_name_filter, hook_fn_sender)
    
    receiver_input_hook_name = utils.get_act_name("q", receiver_layer)
    _, patched_receiver_input_cache = model.run_with_cache(ioi_dataset.toks, names_filter=lambda name: name == receiver_input_hook_name)
    model.reset_hooks()

    patched_pattern_cache = {}
    def cache_patched_pattern_hook(activation, hook): patched_pattern_cache[hook.name] = activation
    def patch_receiver_input_hook(activation, hook):
        activation[:, :, receiver_h_idx] = patched_receiver_input_cache[hook.name][:, :, receiver_h_idx]
        return activation

    model.run_with_hooks(
        ioi_dataset.toks,
        fwd_hooks=[(lambda name: name == receiver_input_hook_name, patch_receiver_input_hook), (pattern_name_filter, cache_patched_pattern_hook)],
    )
    patched_receiver_pattern = patched_pattern_cache[utils.get_act_name("pattern", receiver_layer)][:, receiver_h_idx, :, :]
    
    avg_clean_pattern = einops.reduce(clean_receiver_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
    avg_patched_pattern = einops.reduce(patched_receiver_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
    return avg_patched_pattern - avg_clean_pattern

def main():
    """主函数：读取句子文件，执行Path Patching，保存最终结果。"""
    # --- 配置 ---
    S_INHIBITION_HEADS = [(7, 3), (7, 9), (8, 6), (8, 10)]
    NAME_MOVER_HEADS = [(9, 6), (9, 9)]
    BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ioi", "path_patching")
    INPUT_SENTENCE_FILE = os.path.join(BASE_OUTPUT_DIR, "structured_sentences.jsonl")
    FINAL_OUTPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "causal_dataset.json")

    # --- 1. 初始化模型 ---
    print("Loading model for Path Patching...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
    model.cfg.use_attn_result = True

    # --- 2. 加载句子 ---
    print(f"Loading sentences from {INPUT_SENTENCE_FILE}...")
    sentences_to_process = []
    with open(INPUT_SENTENCE_FILE, "r") as f:
        for line in f:
            sentences_to_process.append(json.loads(line))

    # --- 3. 对每个句子执行Path Patching并整合数据 ---
    final_dataset = {}
    for sentence_info in tqdm(sentences_to_process, desc="Path Patching Sentences"):
        sentence_id = sentence_info["sentence_id"]
        sentence_text = sentence_info["sentence_text"]
        
        # 【已修复】不再使用 IOIDataset(N=0)。直接创建临时的、包含数据的实例。
        # 创建原始数据集
        temp_ioi_dataset = IOIDataset(prompt_type="mixed", N=1, tokenizer=model.tokenizer, prepend_bos=False, device="cuda")
        
        # 【最终修复】为 ioi_prompts 提供完整的元数据，包括 IO, S, 和 TEMPLATE_IDX
        temp_ioi_dataset.ioi_prompts = [{'text': sentence_text, 'IO': sentence_info['io_token'], 'S': sentence_info['s_token'], 'TEMPLATE_IDX': 0}]
        
        # 创建损坏的数据集
        abc_dataset = temp_ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")
        
        # --- 关键修复点：强制两个数据集的序列长度一致 ---
        # 将两个句子的文本放在一起进行分词，并启用padding
        all_texts = [sentence_text, abc_dataset.ioi_prompts[0]['text']]
        all_tokens = model.tokenizer(all_texts, padding='longest', return_tensors='pt', add_special_tokens=False).input_ids.to("cuda")
        
        # 将补齐后的tokens分别赋给两个数据集
        temp_ioi_dataset.toks = all_tokens[0:1]
        abc_dataset.toks = all_tokens[1:2]
        
        str_tokens = model.to_str_tokens(temp_ioi_dataset.toks[0])
        
        final_dataset[sentence_id] = {
            **sentence_info, # 复制所有原始元数据
            "tokens": str_tokens,
            "diff_vectors": {} 
        }
        
        for sender_head in S_INHIBITION_HEADS:
            for receiver_head in NAME_MOVER_HEADS:
                if sender_head[0] >= receiver_head[0]: continue
                avg_diff_pattern = calculate_attention_difference(model, temp_ioi_dataset, abc_dataset, sender_head, receiver_head)
                diff_vector = avg_diff_pattern[-1].cpu().tolist()
                path_key = f"{sender_head[0]}.{sender_head[1]}->{receiver_head[0]}.{receiver_head[1]}"
                final_dataset[sentence_id]["diff_vectors"][path_key] = diff_vector

    # --- 4. 写入最终的整合文件 ---
    with open(FINAL_OUTPUT_FILE, "w") as f:
        json.dump(final_dataset, f, indent=2)
    
    print(f"\n🎉 步骤二完成！最终数据集已保存至: {FINAL_OUTPUT_FILE}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()