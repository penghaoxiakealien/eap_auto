import torch
import json
import os
import re
import asyncio
from tqdm import tqdm
from functools import partial
import sys

# --- 依赖项与路径设置 ---
# 确保可以导入项目中的其他模块
# 这个路径需要指向你项目的根目录
sys.path.append("/home/wangziran/eap_auto/") 
from api import OpenRouter
from tests.experiments.ioi_dataset import IOIDataset
from transformer_lens import HookedTransformer, utils
import einops
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint

# ======================================================================================
# == 阶段一：生成带有元数据的句子
# ======================================================================================

def initialize_openrouter(model: str = "gpt-4o"):
    """初始化OpenRouter API"""
    # 建议从环境变量或安全配置中读取API Key，而不是硬编码
    api_key = "sk-t9ooUIrk73Zg4s72dFCf2QAYWrNsobTW1gT8P7AG7m1r4Wbd" # 请确保这是你有效的Key
    return OpenRouter(model=model, api_key=api_key)

async def generate_ioi_sentences_with_metadata(open_router, num_sentences=100, output_dir=None):
    """
    调用LLM生成IOI句子，并直接返回包含IO和S token的结构化JSON。
    """
    print(f"向LLM请求生成 {num_sentences} 条带有元数据的IOI句子...")
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert AI data linguist. Your task is to generate high-quality sentence prompts for the Indirect Object Identification (IOI) task. "
                "For each sentence, you must identify the Subject (S) and the Indirect Object (IO).\n\n"
                "**CRITICAL INSTRUCTIONS:**\n"
                "1.  The `sentence_text` must be a prompt ending just before the indirect object would be named.\n"
                "2.  The Subject (`s_token`) must appear at least twice.\n"
                "3.  The Indirect Object (`io_token`) must appear exactly once.\n"
                "4.  **Format:** You MUST respond with ONLY a single JSON array of objects. Each object must contain three keys: `sentence_text`, `io_token`, and `s_token`.\n"
                "5.  The `io_token` and `s_token` values MUST match the tokens in the sentence exactly, including any preceding spaces (e.g., ' Mary', not 'Mary').\n\n"
                "**EXAMPLE:**\n"
                "[\n"
                "  {\n"
                "    \"sentence_text\": \"Then, Mary and John went to the store. John gave a book to\",\n"
                "    \"io_token\": \" Mary\",\n"
                "    \"s_token\": \" John\"\n"
                "  }\n"
                "]"
            )
        },
        {
            "role": "user",
            "content": f"Please generate {num_sentences} unique IOI sentences in the specified JSON format.",
        },
    ]
    
    # 【已修复】将 output_dir 参数传递给 generate 函数，用于保存API请求日志
    response = await open_router.generate(messages=messages, output_dir=output_dir)
    
    try:
        structured_data = json.loads(response.text)
        print(f"成功从LLM响应中解析出 {len(structured_data)} 条句子。")
        return structured_data
    except json.JSONDecodeError:
        print("错误：LLM未能返回有效的JSON格式。请检查Prompt或模型。")
        match = re.search(r'\[\s*\{.*\}\s*\]', response.text, re.DOTALL)
        if match:
            print("后备方案：通过正则表达式找到JSON。")
            return json.loads(match.group(0))
        return []

# ======================================================================================
# == 阶段二：执行路径修补 (这部分代码保持不变)
# ======================================================================================

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
    receiver_input_filter = lambda name: name == receiver_input_hook_name
    _, patched_receiver_input_cache = model.run_with_cache(ioi_dataset.toks, names_filter=receiver_input_filter)
    model.reset_hooks()

    patched_pattern_cache = {}
    def cache_patched_pattern_hook(activation, hook):
        patched_pattern_cache[hook.name] = activation
    def patch_receiver_input_hook(activation, hook):
        activation[:, :, receiver_h_idx] = patched_receiver_input_cache[hook.name][:, :, receiver_h_idx]
        return activation

    model.run_with_hooks(
        ioi_dataset.toks,
        fwd_hooks=[(receiver_input_filter, patch_receiver_input_hook), (pattern_name_filter, cache_patched_pattern_hook)],
        return_type=None
    )
    patched_receiver_pattern = patched_pattern_cache[utils.get_act_name("pattern", receiver_layer)][:, receiver_h_idx, :, :]
    
    avg_clean_pattern = einops.reduce(clean_receiver_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
    avg_patched_pattern = einops.reduce(patched_receiver_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
    return avg_patched_pattern - avg_patched_pattern

# ======================================================================================
# == 阶段三：主流程 - 整合所有步骤
# ======================================================================================

async def main():
    """
    完整流程：生成句子 -> 对每个句子做Path Patching -> 保存整合后的结果。
    """
    # --- 全局配置 ---
    NUM_SENTENCES_TO_GENERATE = 100
    S_INHIBITION_HEADS = [(7, 3), (7, 9), (8, 6), (8, 10)]
    NAME_MOVER_HEADS = [(9, 6), (9, 9), (10, 0), (10, 7)]
    OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ioi", "path_patching", "causal_dataset.json")
    
    # 【已修复】定义一个用于存放API请求日志的目录
    generation_log_dir = os.path.join(os.path.dirname(OUTPUT_FILE), "generation_logs")
    os.makedirs(generation_log_dir, exist_ok=True)

    # --- 1. 生成句子和元数据 ---
    open_router = initialize_openrouter()
    # 【已修复】将日志目录传递给生成函数
    sentences_with_metadata = await generate_ioi_sentences_with_metadata(
        open_router, 
        NUM_SENTENCES_TO_GENERATE,
        output_dir=generation_log_dir
    )
    
    if not sentences_with_metadata:
        print("未能生成句子，程序终止。")
        return

    # --- 2. 初始化模型 ---
    print("Loading model for Path Patching...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
    model.cfg.use_attn_result = True

    # --- 3. 对每个句子执行Path Patching并整合数据 ---
    final_dataset = {}
    for i, sentence_info in enumerate(tqdm(sentences_with_metadata, desc="Path Patching Sentences")):
        sentence_id = f"ioi_{i+1:04d}"
        sentence_text = sentence_info["sentence_text"]
        
        ioi_dataset = IOIDataset(prompt_type="mixed", N=0, tokenizer=model.tokenizer, prepend_bos=False, device="cuda")
        ioi_dataset.prompts = [{'text': sentence_text}]
        ioi_dataset.toks = model.to_tokens(sentence_text, prepend_bos=False)
        ioi_dataset.N = 1
        abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")
        
        str_tokens = model.to_str_tokens(ioi_dataset.toks[0])
        
        final_dataset[sentence_id] = {
            "sentence_text": sentence_text,
            "tokens": str_tokens,
            "io_token": sentence_info["io_token"],
            "s_token": sentence_info["s_token"],
            "diff_vectors": {} 
        }
        
        for sender_head in S_INHIBITION_HEADS:
            for receiver_head in NAME_MOVER_HEADS:
                if sender_head[0] >= receiver_head[0]:
                    continue

                avg_diff_pattern = calculate_attention_difference(
                    model, ioi_dataset, abc_dataset, sender_head, receiver_head
                )
                
                diff_vector = avg_diff_pattern[-1].cpu().tolist()
                path_key = f"{sender_head[0]}.{sender_head[1]}->{receiver_head[0]}.{receiver_head[1]}"
                final_dataset[sentence_id]["diff_vectors"][path_key] = diff_vector

    # --- 4. 写入最终的整合文件 ---
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_dataset, f, indent=2)
    
    print(f"\n🎉 数据集构建完成！所有结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    asyncio.run(main())