import torch
import json
import os
from tqdm import tqdm
from functools import partial

from transformer_lens import HookedTransformer, utils, loading_from_pretrained as loading
from transformers import AutoModelForCausalLM, AutoTokenizer
from ioi_dataset import IOIDataset
import einops
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint

LOCAL_MODEL_DIR = "/home/wangziran/gpt2"


def load_local_hooked_transformer(local_model_dir: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(local_model_dir, local_files_only=True)
    cfg = loading.get_pretrained_model_config(
        local_model_dir,
        device=device,
        local_files_only=True,
    )
    model = HookedTransformer(
        cfg,
        tokenizer=tokenizer,
        move_to_device=False,
    )
    state_dict = loading.get_pretrained_state_dict(
        local_model_dir,
        cfg,
        hf_model=hf_model,
        local_files_only=True,
    )
    model.load_and_process_state_dict(state_dict)
    model.move_model_modules_to_device()
    return model


def load_model(device: str = "cuda"):
    """优先以官方名称加载本地缓存的gpt2模型，若目录不存在则回退到gpt2-small。"""
    if os.path.isdir(LOCAL_MODEL_DIR):
        print(f"🔥 正在从本地缓存加载模型: {LOCAL_MODEL_DIR}")
        return load_local_hooked_transformer(LOCAL_MODEL_DIR, device=device)
    print("⚠️ 未找到本地模型目录，回退到默认的 gpt2-small。")
    return HookedTransformer.from_pretrained("gpt2-small", device=device)

# --- 关键：将工具函数直接复制到此脚本中，使其独立 ---
def patch_or_freeze_head_vectors(
    orig_head_vector: torch.Tensor,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: tuple[int, int],
):
    """
    这个钩子函数实现了路径修补的核心逻辑。
    它将所有头的输出(z)冻结为它们在原始缓存(orig_cache)中的值，
    除了一个特定的头(head_to_patch)，该头的值将从新缓存(new_cache)中修补过来。
    """
    # 默认情况下，将所有头的输出设置为原始缓存中的值
    orig_head_vector[...] = orig_cache[hook.name][...]
    # 如果当前层是我们要修补的发送者头所在的层
    if head_to_patch[0] == hook.layer():
        # 则只将该特定头的输出替换为新缓存中的值
        orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return orig_head_vector


def calculate_attention_difference(model, ioi_dataset, abc_dataset, sender_head, receiver_head, receiver_input="q"):
    """
    执行路径修补并返回 receiver_head 的平均注意力模式差异。
    此版本完全复用您项目中已有的、经过验证的路径修补逻辑。
    """
    sender_layer, sender_h_idx = sender_head
    receiver_layer, receiver_h_idx = receiver_head

    # --- Step 0: 准备缓存 ---
    # 缓存所有头的输出(z)和receiver的注意力模式(pattern)
    z_name_filter = lambda name: name.endswith("z")
    pattern_name_filter = lambda name: name == utils.get_act_name("pattern", receiver_layer)
    
    _, clean_cache = model.run_with_cache(ioi_dataset.toks, names_filter=lambda name: z_name_filter(name) or pattern_name_filter(name))
    _, corrupted_cache = model.run_with_cache(abc_dataset.toks, names_filter=z_name_filter)

    clean_receiver_pattern = clean_cache[utils.get_act_name("pattern", receiver_layer)][:, receiver_h_idx, :, :]

    # --- Step 1: 使用 add_hook 添加发送者钩子 ---
    hook_fn_sender = partial(
        patch_or_freeze_head_vectors,
        new_cache=corrupted_cache,
        orig_cache=clean_cache,
        head_to_patch=(sender_layer, sender_h_idx),
    )
    model.add_hook(z_name_filter, hook_fn_sender)

    # --- Step 2: 在被修补的模型上运行，并缓存接收者的输入 ---
    receiver_input_hook_name = utils.get_act_name(receiver_input, receiver_layer)
    receiver_input_filter = lambda name: name == receiver_input_hook_name
    _, patched_receiver_input_cache = model.run_with_cache(
        ioi_dataset.toks, names_filter=receiver_input_filter
    )
    model.reset_hooks() # **非常重要**：立即清理钩子

    # --- Step 3: 使用 run_with_hooks 修补接收者，并捕获最终的注意力模式 ---
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

    # --- Step 4: 计算差异 ---
    avg_clean_pattern = einops.reduce(clean_receiver_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
    avg_patched_pattern = einops.reduce(patched_receiver_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
    return avg_patched_pattern - avg_clean_pattern


def main():
    """
    主函数，对指定文件中的每个句子计算因果路径的注意力变化，并仅保存END位置的差异向量。
    """
    # --- 配置 ---
    S_INHIBITION_HEADS = [(7, 3), (7, 9), (8, 6), (8, 10)]
    NAME_MOVER_HEADS = [(9, 6), (9, 9), (10, 0), (10, 7)]
    
    # 输入和输出文件路径
    SENTENCE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ioi", "hypothesis", "sentences", "generated_sentences.json")
    OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ioi", "path_patching", "causal_effects_by_sentence.json")
    
    # --- 初始化模型 ---
    print("Loading model...")
    model = load_model()
    model.cfg.use_attn_result = True

    # --- 加载句子 ---
    print(f"Loading sentences from {SENTENCE_FILE}...")
    with open(SENTENCE_FILE, "r") as f:
        sentences_data = json.load(f)

    all_causal_effects = {}
        
    # --- 遍历 `generated_sentences.json` 中的每一条句子 ---
    for sentence_id, sentence_info in tqdm(sentences_data.items(), desc="Processing Sentences"):
        sentence_text = sentence_info["sentence"]
        
        # 为当前句子创建IOIDataset对象
        # 1. 创建一个空的 IOIDataset 对象 (使用合法的 prompt_type 和 N=0)
        ioi_dataset = IOIDataset(prompt_type="mixed", N=0, tokenizer=model.tokenizer, prepend_bos=False, device="cuda")
        
        # 2. 手动将当前句子的数据填充进去
        ioi_dataset.prompts = [{'text': sentence_text, 'IO': sentence_info.get('io', ''), 'S': sentence_info.get('s', '')}]
        ioi_dataset.toks = model.to_tokens(sentence_text, prepend_bos=False)
        ioi_dataset.N = 1 # 更新数据集大小为1
        
        # 为当前句子创建对应的损坏数据集
        abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")
        
        str_tokens = model.to_str_tokens(ioi_dataset.toks[0])
        
        all_causal_effects[sentence_id] = {
            "sentence": sentence_text,
            "tokens": str_tokens,
            "diff_vectors": {} # 存储所有路径的END位置差异向量
        }
        
        # --- 遍历所有我们关心的因果路径 ---
        for sender_head in S_INHIBITION_HEADS:
            for receiver_head in NAME_MOVER_HEADS:
                if sender_head[0] >= receiver_head[0]:
                    continue

                # --- 计算差值矩阵 ---
                avg_diff_pattern = calculate_attention_difference(
                    model, ioi_dataset, abc_dataset, sender_head, receiver_head, receiver_input="q"
                )
                
                # --- 只保存END位置的差异向量 ---
                diff_vector = avg_diff_pattern[-1].cpu().tolist()
                
                path_key = f"{sender_head[0]}.{sender_head[1]}->{receiver_head[0]}.{receiver_head[1]}"
                all_causal_effects[sentence_id]["diff_vectors"][path_key] = diff_vector

    # --- 写入文件 ---
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_causal_effects, f, indent=2)
    
    print(f"\nPre-computation complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
