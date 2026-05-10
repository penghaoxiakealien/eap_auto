import asyncio
import sys
import json
import os
import re
import math
import random
import numpy as np
from datetime import datetime
from scipy.stats import kendalltau
from itertools import groupby
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import argparse
import copy
from sklearn.metrics import ndcg_score

sys.path.append("/home/wangziran/eap_auto/")
from api import OpenRouter


VERBOSE_DEBUG = False

async def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Run dual-dimension hypothesis optimization with causal effects and attention patterns.")
    parser.add_argument("--layer", type=int, required=True, help="The layer number to analyze.")
    parser.add_argument("--head", type=int, required=True, help="The head number to analyze.")
    parser.add_argument("--rounds", type=int, default=1, help="(Deprecated) Kept for compatibility; no longer controls iterations.")
    parser.add_argument("--typename", type=str, default="", help="(Deprecated) Optional typename of head.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory path.")
    parser.add_argument("--data_source_dir", type=str, required=True, help="Directory path for input data.")
    parser.add_argument(
        "--optimize-only",
        choices=["dual", "causal", "attention"],
        default="dual",
        help="Refinement focus during iterations (dual=causal+attention, causal=only causal, attention=only attention).",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=2,
        help="Run validation every N epochs.",
    )
    parser.add_argument(
        "--validation-sample-size",
        type=int,
        default=0,
        help="Validation sentences per checkpoint (0 means full validation split).",
    )
    parser.add_argument(
        "--test-all-validations",
        action="store_true",
        help="Evaluate every validation checkpoint hypothesis on the test split.",
    )
    parser.add_argument(
        "--test-sample-size",
        type=int,
        default=0,
        help="Test sentences for final evaluation (0 means full test split).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2-2025-12-11",
        help="OpenRouter model name.",
    )
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument(
        "--with-reasoning",
        action="store_true",
        help="Require [REASONING] blocks in LLM outputs.",
    )
    reasoning_group.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Do not require [REASONING] blocks; use concise outputs only (default).",
    )
    return parser.parse_args()

def initialize_openrouter(model: str = "gpt-5.2-2025-12-11"):
    """初始化OpenRouter API."""
    # api_key = "sk-Z3pwy4dD8WY2XZlbzch66NP5hQIoFKeU7KvI2XD8bQSyFVGO"
    # api_key = "sk-ssCRXZzlj8qhPNs6Ps2BxTXZQXq97vJvKATpFXdwYV0E0gUO"
    api_key = "sk-cdENMjpwVIpdd1Iv0auFiHizYdgnWM0ZFKhHN3UBYqKIoqpA"
    return OpenRouter(model=model, api_key=api_key)

def split_dataset(full_dataset, train_split=0.4, validation_split=0.2, seed=42):
    """将完整数据集确定性地划分为训练/验证/测试集"""
    test_split = 1.0 - train_split - validation_split
    if test_split < 0:
        raise ValueError("Split ratios must sum to <= 1.0")
    print(
        f"将数据集划分为 {train_split:.0%} 训练集、{validation_split:.0%} 验证集、{test_split:.0%} 测试集..."
    )
    sentence_ids = [str(sid) for sid in full_dataset.keys()]
    train_end = int(len(sentence_ids) * train_split)
    val_end = int(len(sentence_ids) * (train_split + validation_split))
    train_ids = sentence_ids[:train_end]
    validation_ids = sentence_ids[train_end:val_end]
    test_ids = sentence_ids[val_end:]

    train_dataset = {sid: full_dataset[sid] for sid in train_ids}
    validation_dataset = {sid: full_dataset[sid] for sid in validation_ids}
    test_dataset = {sid: full_dataset[sid] for sid in test_ids}

    print(
        f"训练集大小: {len(train_dataset)}, 验证集大小: {len(validation_dataset)}, 测试集大小: {len(test_dataset)}"
    )
    return train_dataset, validation_dataset, test_dataset

def normalize_token(token):
    """标准化token用于比较"""
    return token.strip().lower()


def log_raw_api_response(output_dir, stage, payload):
    """Write raw LLM responses for debugging. Overwrite on run_start."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "raw_api_responses.jsonl")
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stage": stage,
            **payload,
        }
        mode = "a"
        if stage == "run_start":
            mode = "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"警告: 无法写入 raw_api_responses.jsonl: {e}")

def extract_hypothesis_text(hypothesis_text):
    """从响应中提取假设文本，并返回匹配部分之前的字符串"""
    tag_re = re.compile(r"(?:\*\*\s*)?\[HYPOTHESIS\]\s*:?(?:\s*\*\*)?", re.IGNORECASE)
    tags_re = re.compile(r"(?:\*\*\s*)?\[[A-Z_]+\]\s*:?(?:\s*\*\*)?", re.IGNORECASE)
    match = tag_re.search(hypothesis_text)
    if not match:
        alt_re = re.compile(r"(?:\*\*\s*)?HYPOTHESIS\s*:?(?:\s*\*\*)?", re.IGNORECASE)
        match = alt_re.search(hypothesis_text)
        if not match:
            print("No hypothesis found in the response.")
            return None, None
    before_hypothesis = hypothesis_text[: match.start()].strip()
    tail = hypothesis_text[match.end() :].strip()
    next_tag = tags_re.search(tail)
    if next_tag:
        tail = tail[: next_tag.start()].strip()
    tail = re.sub(r"^\s*\[REASONING\].*?$", "", tail, flags=re.IGNORECASE | re.MULTILINE).strip()
    return before_hypothesis, tail or None

# -----------------------------------------------------------------------------
# 数据加载函数 (适配预处理脚本)
# -----------------------------------------------------------------------------

def load_examples_from_preprocessed(attention_scores_ground_truth_path, num_examples=5):
    """
    从precompute_attention_scores.py的输出文件加载示例
    适配格式：attention_scores_ground_truth.jsonl
    """
    example_sentence, example_activations, example_indirect_object = [], [], []
    
    try:
        with open(attention_scores_ground_truth_path, "r") as f:
            items = [json.loads(line) for line in f if line.strip()]
        
        # 取前num_examples个作为示例
        for item in items[:num_examples]:
            sentence_text = item.get("original_sentence", "")
            io = item.get("indirect_object", "")
            # 注意：这里的top_5_attended_tokens是旧格式，现在改为从attention_scores里取
            top_tokens = sorted(item.get("attention_scores", []), key=lambda x: x['score'], reverse=True)[:2]

            marked_sentence = sentence_text
            for token_info in top_tokens:
                token = token_info["token"].split('_')[0] # 去掉后缀
                marked_sentence = marked_sentence.replace(token, f"<<{token}>>", 1)
            
            example_sentence.append(f"{marked_sentence} {{{{DISAMB}}}}")
            
            activations_str = ", ".join([f'("{t["token"].split("_")[0]}", {t["score"]:.2f})' for t in top_tokens])
            example_activations.append(f"Activations: {activations_str}")
            
            example_indirect_object.append("{{DISAMB}}")
            
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法加载示例文件 {attention_scores_ground_truth_path}: {e}")
        return [], [], []
    
    return example_sentence, example_activations, example_indirect_object

def load_preprocessed_attention_dataset(preprocessed_attention_path):
    """
    加载preprocess_attention_scores.py的输出文件
    格式：[{"sentence_id": "...", "sentence_text": "...", "top_k_tokens": ["token1_1", "token2_1"]}]
    """
    try:
        with open(preprocessed_attention_path, "r") as f:
            data = json.load(f)
        # --- 修改点：将列表转换为以 sentence_id 为键的字典，方便快速查找 ---
        converted_data = {}
        for item in data:
            sentence_id = item['sentence_id']
            sentence_text = item.get("sentence_text", "")
            top_tokens = item.get("top_k_tokens", [])
            converted_data[sentence_id] = {
                "sentence_id": sentence_id,
                "sentence_text": sentence_text,
                "top_k_tokens": _suffix_tokens_with_sentence(top_tokens, sentence_text),
            }
        print(f"成功从 {preprocessed_attention_path} 加载并转换 {len(converted_data)} 条注意力数据。")
        return converted_data
        
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法加载预处理注意力数据 {preprocessed_attention_path}: {e}")
        return {}

def load_preprocessed_causal_dataset(preprocessed_causal_path):
    """
    加载preprocess_causal_effects.py的输出文件
    格式：[{"sentence_id": "...", "ground_truth": {"increase": [...], "decrease": [...]}}]
    转换为auto_NMH.py需要的字典格式
    """
    try:
        with open(preprocessed_causal_path, "r") as f:
            data = json.load(f)
        
        converted_data = {}
        for item in data:
            sid = item["sentence_id"]
            sentence_text = item.get("sentence_text", "")
            ground_truth = item.get("ground_truth", {})
            converted_data[sid] = {
                "sentence_text": sentence_text,
                "ground_truth_tokens": {
                    "increase": _suffix_tokens_with_sentence(ground_truth.get("increase", []), sentence_text),
                    "decrease": _suffix_tokens_with_sentence(ground_truth.get("decrease", []), sentence_text),
                }
            }
        print(f"成功从 {preprocessed_causal_path} 加载 {len(converted_data)} 条因果数据。")
        return converted_data
        
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法加载预处理因果数据 {preprocessed_causal_path}: {e}")
        return {}

def load_head_logit_effects(logit_effect_path):
    """加载 head 的 logit 贡献信息。

    返回:
      - effects: {"layer.head": value} (整体平均)
      - per_sentence: {"layer.head": {sentence_id: delta}} (逐句)
    """
    try:
        with open(logit_effect_path, "r") as f:
            data = json.load(f)
        effects = {}
        per_sentence = {}
        for k, v in data.items():
            if k in {"per_sentence", "meta"}:
                continue
            if isinstance(v, (int, float)):
                effects[str(k)] = float(v)
        per_records = data.get("per_sentence")
        if isinstance(per_records, list):
            by_head = {}
            for rec in per_records:
                head = rec.get("head") or (data.get("meta", {}) or {}).get("head")
                sid = rec.get("sentence_id")
                delta = rec.get("delta_logit_diff")
                if head is None or sid is None or delta is None:
                    continue
                by_head.setdefault(str(head), {})[str(sid)] = float(delta)
            per_sentence = by_head
        return effects, per_sentence
    except FileNotFoundError:
        print(f"警告: 未找到 logit 贡献文件 {logit_effect_path}，将跳过该证据。")
    except json.JSONDecodeError as e:
        print(f"警告: 解析 logit 贡献文件 {logit_effect_path} 失败: {e}")
    return {}, {}

def get_head_logit_direction(logit_effects, sender_head):
    """根据 head 的直接 logit 贡献判断方向（>=0 为 increase，<0 为 decrease）"""
    key = f"{sender_head[0]}.{sender_head[1]}"
    if key not in logit_effects:
        raise ValueError(f"未找到 head {key} 的 logit 贡献方向")
    value = float(logit_effects[key])
    direction = "increase" if value >= 0 else "decrease"
    return direction, value

def format_head_logit_effect(layer, head, logit_effects):
    """生成格式化的 logit 贡献描述字符串"""
    key = f"{layer}.{head}"
    if key not in logit_effects:
        return "(未找到该注意力头的 logit 贡献记录)"

    value = logit_effects[key]
    direction = "提升" if value > 0 else "压低" if value < 0 else "几乎没有影响"
    return (
        "Logit Contribution Evidence:\n"
        f"- Head {layer}.{head} 在 path patching 后对正确 logit 差值的直接影响约为 {value:.2f}。"
        f" 该数值表示该注意力头被保留时会{direction}正确答案与干扰项的 logits 差距。\n"
        "- 请结合该数值判断它在 garden path NP/Z v-trans 任务中的助攻或抑制效应。\n"
    )

def extract_io_from_sentence(sentence_text):
    """从句子中提取间接宾语（简化版本）"""
    names = re.findall(r'\b[A-Z][a-z]+\b', sentence_text)
    if len(names) >= 2:
        return names[-1]
    return "Unknown"

# -----------------------------------------------------------------------------
# 数据采样函数
# -----------------------------------------------------------------------------

def calculate_sample_distribution(token_num_freq, batch_size):
    """根据 token_num 的频率计算采样分布，确保总和等于 batch_size"""
    total_freq = sum(token_num_freq.values())
    if total_freq == 0: return {}
    normalized_freq = {k: v / total_freq for k, v in token_num_freq.items()}
    initial_distribution = {k: v * batch_size for k, v in normalized_freq.items()}
    rounded_distribution = {k: round(v) for k, v in initial_distribution.items()}
    total_samples = sum(rounded_distribution.values())

    if total_samples != batch_size:
        sorted_token_nums = sorted(normalized_freq.keys(), 
                                 key=lambda k: initial_distribution[k] - rounded_distribution[k], 
                                 reverse=True)
        diff = batch_size - total_samples
        
        for token_num in sorted_token_nums:
            if diff == 0:
                break
            adjustment = 1 if diff > 0 else -1
            if rounded_distribution[token_num] + adjustment >= 0:
                rounded_distribution[token_num] += adjustment
                diff -= adjustment

    return rounded_distribution

def random_sample_sentences_from_preprocessed(preprocessed_attention_data, output_sentence_path, batch_size, iteration):
    """从预处理的注意力数据中采样句子"""
    # preprocessed_attention_data 现在是字典
    all_items = list(preprocessed_attention_data.values())

    if VERBOSE_DEBUG:
        print("=== 采样调试信息 ===")
        print(f"preprocessed_attention_data 类型: {type(preprocessed_attention_data)}")
        print(f"all_items 长度: {len(all_items)}")
        if all_items:
            print(f"第一个item的结构: {all_items[0].keys()}")
            print(f"第一个item示例: {all_items[0]}")
        print("==================")

    token_num_counter = defaultdict(int)
    for item in all_items:
        token_num = len(item.get("top_k_tokens", []))
        if token_num > 0:
            token_num_counter[token_num] += 1
    
    if VERBOSE_DEBUG:
        print(f"token_num_counter: {dict(token_num_counter)}")
    
    total_count = sum(token_num_counter.values())
    if total_count == 0:
        print("❌ 错误: 没有找到任何有效的token数据!")
        return {}
        
    all_token_num_freq = {k: v / total_count for k, v in token_num_counter.items()}
    token_num_freq = {k: v for k, v in all_token_num_freq.items() if v > 0.05}
    sample_distribution = calculate_sample_distribution(token_num_freq, batch_size)

    if VERBOSE_DEBUG:
        print(f"sample_distribution: {sample_distribution}")

    sampled_data = []
    for token_num, sample_count in sample_distribution.items():
        filtered_data = [item for item in all_items if len(item.get("top_k_tokens", [])) == token_num]
        if len(filtered_data) >= sample_count:
            sampled_data.extend(random.sample(filtered_data, int(sample_count)))
        else:
            sampled_data.extend(filtered_data)

    print(f"实际采样到的数据数量: {len(sampled_data)}")

    result_dict = {}
    for i, item in enumerate(sampled_data, start=1):
        key = f"{i}_test"
        original_sentence = item.get("sentence_text", "")
        result_dict[key] = {
            "sentence_id": item.get("sentence_id"), # --- 新增：传递 sentence_id ---
            "sentence": original_sentence,
            "io": item.get("indirect_object") or extract_io_from_sentence(original_sentence),
            "number_of_important_tokens": len(item.get("top_k_tokens", []))
        }
    
    write_dict = {f"iteration_{iteration}": result_dict}
    with open(output_sentence_path, "a") as f:
        f.write(json.dumps(write_dict, ensure_ascii=False) + "\n")
    
    print(f"采样句子已保存到 {output_sentence_path}")
    if VERBOSE_DEBUG:
        print(f"最终返回的result_dict长度: {len(result_dict)}")
    return result_dict

# -----------------------------------------------------------------------------
# 假设生成与精炼
# -----------------------------------------------------------------------------

async def generate_hypothesis(
    open_router,
    layer,
    head,
    explanation,
    example_sentence,
    example_activations,
    example_indirect_object,
    causal_examples,
    optimize_only,
    output_dir,
    require_reasoning: bool,
):
    """生成初始假设"""
    include_causal = optimize_only != "attention"
    include_attention = optimize_only != "causal"
    user_content = ""
    if include_attention:
        user_content = "\n".join(
            f"{sentence}{activations}{io}" for sentence, activations, io in zip(example_sentence, example_activations, example_indirect_object)
        )
    causal_content = ""
    if include_causal:
        lines = []
        for i, ex in enumerate(causal_examples or [], 1):
            lines.append(
                f"**Causal Example {i}:**\n"
                f"- Sentence: \"{ex.get('sentence_text', '')}\"\n"
                f"- Ground-truth effect of corrupting head ({layer},{head}) on correct-vs-incorrect logit diff at DISAMB: {ex.get('direction', 'UNKNOWN')}\n"
            )
        causal_content = "\n".join(lines)
    print(f"为 layer {layer}, head {head} 生成初始假设...")
    task_scope_line = (
        "You should jointly use direct attention evidence and causal evidence.\n\n"
        if include_attention and include_causal
        else "You should focus only on direct attention evidence.\n\n"
        if include_attention
        else "You should focus only on causal evidence.\n\n"
    )
    attention_guidelines = (
        "Additionally, you will receive a list of examples in which the disambiguation position has been inserted at the end of the sentence, marked using double curly braces (e.g., {{DISAMB}}). The same disambiguation marker will also be displayed separately after the sentence as a reference.\n\n"
        "Guidelines:\n"
        "- You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. These words have high attention score from the token immediately before the disambiguation position quoted with {{ }} of the sentence.\n"
        "- If a sequence of consecutive tokens is important, it is fully enclosed within the delimiters <<like this>>.\n"
        "- Each example is followed by a list of important tokens and their scores (between 0 and 1) after 'Activations:', where higher values indicate stronger influence. The total sum of scores will not exceed 1.\n"
        "- Your job is not to infer the hidden disambiguation token, but to hypothesize what this attention head is doing based on the pattern of important tokens and structural positions.\n"
        "- Your hypothesis must focus on the relation between the important tokens, their scores, and Garden NP/Z disambiguation at the sentence end.\n\n"
        if include_attention
        else ""
    )
    causal_guidelines = (
        "You are also given the head's influence on the correct-vs-incorrect logit difference at the disambiguation position.\n\n"
        "When analyzing attention heads, please consider their contribution to the model's final prediction. While attention patterns show what a head focuses on, "
        "logit-direction evidence reveals whether keeping the head helps or hurts the correct Garden disambiguation.\n\n"
        "When forming your hypothesis, consider:\n"
        "- How the head's attention pattern affects the model's ability to choose the correct disambiguation token\n"
        "- In which types of sentence structures this head contributes most\n"
        "- What relationship exists between attention patterns and causal direction\n"
        "- Whether the head is more important in specific structural contexts\n\n"
        if include_causal
        else ""
    )
    hypothesis_constraint = (
        "- In the [HYPOTHESIS] paragraph, explicitly describe both the attention pattern (what the head attends to) and the causal effect on the task/logit difference.\n"
        if include_attention and include_causal
        else "- In the [HYPOTHESIS] paragraph, explicitly describe the attention pattern (what the head attends to) and how this supports Garden NP/Z behavior.\n"
        if include_attention
        else "- In the [HYPOTHESIS] paragraph, explicitly describe the head's causal effect on the task/logit difference.\n"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a meticulous AI researcher conducting an important investigation into patterns found in language. "
                "The text is based on the Garden Path NP/Z v-trans task, where the model is asked to choose the correct disambiguation token "
                "(\"was\" or \"for\") at the end of a sentence.\n\n"
                f"{task_scope_line}"
                f"{attention_guidelines}"
                f"{causal_guidelines}"
                "Important:\n"
                "- The head may help OR hurt the task; do not assume it is beneficial.\n"
                "- Do not mention << >> or {{ }} in your final hypothesis.\n"
                "- The final paragraph must be your hypothesis, beginning with [HYPOTHESIS]:\n"
                "- The [HYPOTHESIS] should be one single paragraph, clearly and thoroughly articulating the head's behavior.\n\n"
                "- The hypothesis should describe the dominant functional behavior using precise linguistic, semantic, or structural terminology. Avoid overly abstract or generic phrasing.\n"
                "- Analyze the examples by grouping them into input categories (e.g., sentence structures, structural positions, or ambiguity types) and explain how the head behaves differently across these types. Base your final hypothesis on this classification-aware reasoning.\n"
                "- Do not include tokens, examples, scores, or error context in the [HYPOTHESIS] paragraph.\n"
                "- The hypothesis should sound like a standalone description of the head's role in the model with no concessions or negations.\n"
                f"{hypothesis_constraint}"
                "- Ensure the [HYPOTHESIS]: tag includes the colon; do not omit it.\n"
            )
        },
        {
            "role": "user",
            "content": (
                f"\n{explanation}"
                f"\n{user_content if include_attention else ''}"
                f"\n{causal_content if include_causal else ''}"
                "\nPlease output only one final [HYPOTHESIS] paragraph."
            ),
        },
    ]
    
    hypothesis = await open_router.generate(messages=messages, output_dir=output_dir)
    print("hypothesis:", hypothesis.text, "\n")
    return hypothesis.text

# -----------------------------------------------------------------------------
# 阶段二：因果效应预测
# -----------------------------------------------------------------------------

def get_causal_effects_from_preprocessed(sampled_sentences, preprocessed_causal_dataset, top_k=1):
    """从预处理的因果数据集中获取真实的因果效应"""
    causal_examples = []
    print(f"正在为 {len(sampled_sentences)} 个抽样句子获取预处理的因果效应...")
    
    # --- 添加调试信息 ---
    if VERBOSE_DEBUG:
        print("=== 调试信息 ===")
        print(f"preprocessed_causal_dataset 类型: {type(preprocessed_causal_dataset)}")
        print(f"preprocessed_causal_dataset 长度: {len(preprocessed_causal_dataset) if hasattr(preprocessed_causal_dataset, '__len__') else 'N/A'}")
        print(f"preprocessed_causal_dataset 前5个键: {list(preprocessed_causal_dataset.keys())[:5] if hasattr(preprocessed_causal_dataset, 'keys') else 'N/A'}")
        print("sampled_sentences 详情:")
        for key, data in list(sampled_sentences.items())[:3]:
            print(f"  {key}: sentence_id = '{data.get('sentence_id')}' (类型: {type(data.get('sentence_id'))})")
        print("==================")

    for key, data in sampled_sentences.items():
        # --- 修改点：使用 sentence_id 进行查找 ---
        sentence_id = data.get('sentence_id')
        if VERBOSE_DEBUG:
            print(f"查找 sentence_id: '{sentence_id}' (类型: {type(sentence_id)})")
        
        if sentence_id in preprocessed_causal_dataset:
            if VERBOSE_DEBUG:
                print(f"✓ 找到匹配的因果数据")
            matched_data = preprocessed_causal_dataset[sentence_id]
        else:
            if VERBOSE_DEBUG:
                print(f"✗ 未找到匹配的因果数据")
                print(f"可用的sentence_id列表: {list(preprocessed_causal_dataset.keys())[:10]}")
            # 尝试查找相似的ID
            similar_ids = [sid for sid in preprocessed_causal_dataset.keys() if str(sentence_id) in str(sid) or str(sid) in str(sentence_id)]
            if similar_ids and VERBOSE_DEBUG:
                print(f"相似的ID: {similar_ids}")
            print(f"警告: 在预处理的因果数据集中未找到 sentence_id '{sentence_id}'")
            continue

        ground_truth = matched_data.get('ground_truth_tokens', {})
        increase_tokens = ground_truth.get('increase', [])
        decrease_tokens = ground_truth.get('decrease', [])
        
        causal_examples.append({
            "key": key,
            "sentence_id": sentence_id, # 传递ID
            "sentence_text": data['sentence'],
            "increase_tokens": increase_tokens[:top_k],
            "decrease_tokens": decrease_tokens[:top_k],
            "io": data['io']
        })
        
    print(f"最终找到 {len(causal_examples)} 个匹配的因果效应示例")
    return causal_examples   


def build_direction_causal_examples_from_ids(sentence_ids, preprocessed_attention_data, default_direction, per_sentence_effects=None):
    """Build Garden terminal causal examples using sentence-level logit-direction truth."""
    examples = []
    for sid in sentence_ids:
        item = preprocessed_attention_data.get(str(sid))
        if not item:
            continue
        direction = default_direction
        if per_sentence_effects and str(sid) in per_sentence_effects:
            delta = per_sentence_effects[str(sid)]
            direction = "increase" if float(delta) >= 0 else "decrease"
        examples.append({
            "sentence_id": str(sid),
            "sentence_text": item.get("sentence_text", ""),
            "direction": direction,
        })
    return examples

async def predict_for_single_sentence_causal(open_router, hypothesis, key, sentence_data, sender_head, output_dir, require_reasoning: bool):
    """为单个句子生成因果方向预测（increase/decrease）"""
    sentence_text = sentence_data['sentence']
    io = sentence_data['io']

    system_prompt = (
        "You are a meticulous AI researcher applying a hypothesis to predict experimental outcomes in a garden path NP/Z v-trans task. Your task is to predict how corrupting a 'Sender Head' will change the logit difference at the disambiguation position.\n\n"
        "**--- Key Concepts in the Garden Path NP/Z v-trans Task (Crucial Background) ---**\n"
        "To understand the hypothesis, you must know these linguistic roles in the context of a sentence. However, in this task, we may not provide the whole sentence, which means that the disambiguation token at the end of the sentence may be hidden for prediction. For example, \"As the criminal shot the woman who told bad jokes [was/for].\"\n"
        "In this sentence:\n"
        "- **Subordinator:** the clause introducer (e.g., 'As', 'While', 'After', 'When').\n"
        "- **Subject (SUBJ):** the agent of the main clause (e.g., 'criminal').\n"
        "- **Ambiguous Verb (VERB):** the main verb whose transitivity drives the NP/Z ambiguity (e.g., 'shot' vs an intransitive verb).\n"
        "- **Object NP Head (OBJ_HEAD):** the head noun of the ambiguous object phrase (e.g., 'woman').\n"
        "- **Relative Pronoun / Verb (REL_PRON / REL_VERB):** the relative clause that follows the object (e.g., 'who told').\n"
        "- **Disambiguation Position:** the final token the model must choose (e.g., 'was' vs 'for'), which signals the correct parse.\n"
        "The task is about correctly predicting the **disambiguation token** at the sentence end. The disambiguation token is always one of: **\"was\"** or **\"for\"**. The correct answer is NOT given in the prompt.\n\n"
        "--- **Core Task & Causal Rules** ---\n"
        f"Your task is to predict how **corrupting Sender Head {sender_head}** will change the logit difference (correct token minus incorrect token) at the disambiguation position, according to the hypothesis.\n\n"
        "This means you must choose exactly one direction:\n"
        "- **INCREASE**: logit difference becomes larger (the head is contributory).\n"
        "- **DECREASE**: logit difference becomes smaller (the head is inhibitory).\n\n"
        "**MUST FOLLOW RULES:**\n"
        + (
            "- **Strict Formatting:** You must first provide a step-by-step analysis in a `[REASONING]` block. Then, on a new line, provide the final answer in a `[PREDICTION]` block.\n"
            if require_reasoning
            else "- **Strict Formatting:** Provide only a `[PREDICTION]` block (no other text).\n"
        )
        + "- **[PREDICTION] Format:** output exactly one word: `INCREASE` or `DECREASE`.\n\n"
    )
    
    user_prompt = (
        f"**Hypothesis:** {hypothesis}\n"
        f"Now, using the **actual hypothesis for Head {sender_head}**, apply the reasoning process to the following sentence.\n\n"
        f"**Sentence to Analyze:**\n`{key}: {sentence_text} {{{{DISAMB}}}}`"
    )
    
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_prompt}
    ]
    response = await open_router.generate(messages=messages, output_dir=output_dir)
    
    response_text = response.text.strip()
    
    if require_reasoning:
        reasoning_match = re.search(r"\[REASONING\]\s*(.*?)\s*\[PREDICTION\]", response_text, re.DOTALL | re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning found."
    else:
        reasoning = ""
    prediction_match = re.search(r"\[PREDICTION\]\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
    pred_text = prediction_match.group(1).strip() if prediction_match else ""
    pred_lower = pred_text.lower()
    if "increase" in pred_lower and "decrease" in pred_lower:
        direction = "unknown"
    elif "increase" in pred_lower:
        direction = "increase"
    elif "decrease" in pred_lower:
        direction = "decrease"
    else:
        direction = "unknown"

    return key, {
        "direction": direction,
        "reasoning": reasoning,
        "sentence_id": sentence_data.get("sentence_id"),
        "raw_response": response_text
    }


async def predict_for_sentence_batch_causal(open_router, hypothesis, sentences_data, sender_head, output_dir, require_reasoning: bool, max_retries=1):
    """按批预测多个句子的因果方向（increase/decrease），失败时回退到单句。"""
    if not sentences_data:
        return {}

    sentence_lines = []
    for key, data in sentences_data.items():
        sentence_text = data.get("sentence", "")
        if sentence_text:
            sentence_lines.append(f"`{key}: {sentence_text} {{{{DISAMB}}}}`")
    if not sentence_lines:
        return {}

    system_prompt = (
        "You are a meticulous AI researcher applying a hypothesis to predict experimental outcomes in a garden path NP/Z v-trans task. "
        "Your task is to predict how corrupting a Sender Head will change the logit difference at the disambiguation position.\n\n"
        "**--- Key Concepts in the Garden Path NP/Z v-trans Task (Crucial Background) ---**\n"
        "The disambiguation token at the sentence end is always one of: **\"was\"** or **\"for\"**. The correct answer is NOT given in the prompt.\n"
        "Key positions:\n"
        "- **Subordinator:** clause introducer (As/While/After/When)\n"
        "- **SUBJ:** main clause subject\n"
        "- **VERB:** ambiguous verb\n"
        "- **OBJ_HEAD:** object NP head\n"
        "- **REL_PRON / REL_VERB:** relative clause cue and verb\n"
        "- **Disambiguation Position:** final token choice (was/for)\n\n"
        "--- **Core Task & Causal Rules** ---\n"
        f"Predict how corrupting Sender Head {sender_head} changes the logit difference (correct token minus incorrect token) at the disambiguation position.\n\n"
        "For each sentence choose exactly one direction:\n"
        "- **INCREASE**: logit difference becomes larger\n"
        "- **DECREASE**: logit difference becomes smaller\n\n"
        "**MUST FOLLOW RULES:**\n"
        + (
            "- You may think silently, but the output must contain only a single `[PREDICTION]` block with one line per sentence.\n"
            if require_reasoning
            else "- Output only a single `[PREDICTION]` block with one line per sentence.\n"
        )
        + "- Each line must be in the format `sid: INCREASE` or `sid: DECREASE`.\n"
    )
    user_prompt = (
        f"**Hypothesis:** {hypothesis}\n\n"
        "**Sentences to Analyze (one output line per sentence):**\n"
        f"{chr(10).join(sentence_lines)}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def _extract_line_by_sid(text, sid):
        candidate = None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("`") and line.endswith("`"):
                line = line[1:-1].strip()
            if line.startswith(f"{sid}:"):
                candidate = line
        return candidate

    for _ in range(max_retries):
        response = await open_router.generate(messages=messages, output_dir=output_dir)
        response_text = response.text.strip()
        prediction_matches = re.findall(r"\[PREDICTION\]\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
        pred_text = prediction_matches[-1].strip() if prediction_matches else response_text
        parsed = {}
        valid = True
        for key, sentence_data in sentences_data.items():
            line = _extract_line_by_sid(pred_text, key)
            if not line:
                valid = False
                break
            line_lower = line.lower()
            if "increase" in line_lower and "decrease" in line_lower:
                valid = False
                break
            if "increase" in line_lower:
                direction = "increase"
            elif "decrease" in line_lower:
                direction = "decrease"
            else:
                valid = False
                break
            parsed[key] = {
                "direction": direction,
                "reasoning": "",
                "sentence_id": sentence_data.get("sentence_id"),
                "raw_response": response_text,
            }
        if valid and len(parsed) == len(sentences_data):
            return parsed

    fallback = {}
    tasks = []
    for key, data in sentences_data.items():
        tasks.append(
            predict_for_single_sentence_causal(
                open_router, hypothesis, key, data, sender_head, output_dir, require_reasoning
            )
        )
    results = await asyncio.gather(*tasks)
    for key, item in results:
        fallback[key] = item
    return fallback

def _get_suffixed_word_map(original_text):
    """辅助函数：为句子的每个词生成带后缀的映射"""
    words = re.findall(r"\w+|[^\w\s]", original_text)
    global_counts = defaultdict(int)
    for w in words:
        global_counts[normalize_token(w)] += 1
    
    running_counts = defaultdict(int)
    suffixed_map = {}
    for i, word in enumerate(words):
        norm_word = normalize_token(word)
        running_counts[norm_word] += 1
        if global_counts[norm_word] > 1:
            suffixed_map[i] = f"{word.strip()}_{running_counts[norm_word]}"
        else:
            suffixed_map[i] = word.strip()
    return words, suffixed_map

def _suffix_tokens_with_sentence(tokens, sentence_text):
    """根据原句为token补齐后缀，确保与内部评测规则一致"""
    if not tokens or not sentence_text:
        return tokens
    # 如果 token 已经带编号后缀，则直接返回
    def already_suffixed(token):
        return bool(re.search(r"_[0-9]+$", token))
    try:
        words, suffixed_map = _get_suffixed_word_map(sentence_text)
    except LookupError:
        # 分词器缺失时退回原始 tokens
        return tokens
    occurrences = defaultdict(list)
    for idx, word in enumerate(words):
        suffixed = suffixed_map.get(idx)
        if suffixed:
            occurrences[normalize_token(word)].append(suffixed)
    usage = defaultdict(int)
    suffixed_tokens = []
    for token in tokens:
        if already_suffixed(token):
            suffixed_tokens.append(token)
            continue
        norm = normalize_token(token)
        available = occurrences.get(norm)
        if not available:
            suffixed_tokens.append(token)
            continue
        pos = usage[norm]
        if pos >= len(available):
            pos = len(available) - 1
        suffixed_tokens.append(available[pos])
        usage[norm] += 1
    return suffixed_tokens

def build_sentence_dict_from_preprocessed(preprocessed_data, prefix="v"):
    """Convert preprocessed attention dict into sampling-style sentence dict."""
    result = {}
    for i, item in enumerate(preprocessed_data.values(), start=1):
        key = f"{prefix}{i}_test"
        sentence_text = item.get("sentence_text", "")
        result[key] = {
            "sentence_id": item.get("sentence_id"),
            "sentence": sentence_text,
            "io": item.get("indirect_object") or extract_io_from_sentence(sentence_text),
            "number_of_important_tokens": len(item.get("top_k_tokens", [])),
        }
    return result


def build_sentences_from_ids(sentence_ids, preprocessed_data, prefix="sample"):
    """Build sampling-style sentence dict from selected sentence ids."""
    result = {}
    for i, sid in enumerate(sentence_ids, start=1):
        item = preprocessed_data.get(str(sid))
        if not item:
            continue
        sentence_text = item.get("sentence_text", "")
        key = f"{prefix}{i}_test"
        result[key] = {
            "sentence_id": str(sid),
            "sentence": sentence_text,
            "io": item.get("indirect_object") or extract_io_from_sentence(sentence_text),
            "number_of_important_tokens": len(item.get("top_k_tokens", [])),
        }
    return result


def chunk_dict(data_dict, chunk_size):
    """按固定大小切分字典，保持顺序。"""
    items = list(data_dict.items())
    return [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]


def sample_sentence_ids(preprocessed_data, batch_size, start_idx=0):
    """Deterministically sample sentence ids in order, wrapping around."""
    all_ids = list(preprocessed_data.keys())
    if not all_ids:
        return [], start_idx
    selected = [all_ids[(start_idx + i) % len(all_ids)] for i in range(batch_size)]
    return selected, start_idx + batch_size

def _parse_and_suffix_tokens(original_sentence_text: str, marked_sentence: str, increase_markers: tuple = ("<<", ">>"), decrease_markers: tuple = ("[[", "]]")) -> tuple[list[str], list[str]]:
    """解析并为token添加后缀"""
    original_tokens = word_tokenize(original_sentence_text)
    global_counts = defaultdict(int)
    for token in original_tokens:
        global_counts[normalize_token(token)] += 1

    marked_tokens = word_tokenize(marked_sentence)
    increase_suffixed, decrease_suffixed = [], []
    running_counts = defaultdict(int)
    
    in_increase = False
    in_decrease = False
    last_token = ""

    for token in marked_tokens:
        if token == increase_markers[0][0] and last_token == increase_markers[0][0]:
            in_increase = True
        elif token == decrease_markers[0][0] and last_token == decrease_markers[0][0]:
            in_decrease = True
        elif token == increase_markers[1][0] and last_token == increase_markers[1][0]:
            in_increase = False
        elif token == decrease_markers[1][0] and last_token == decrease_markers[1][0]:
            in_decrease = False

        is_marker_char = token in ['<', '>', '[', ']']
        if not is_marker_char and (in_increase or in_decrease):
            norm_token = normalize_token(token)
            running_counts[norm_token] += 1
            
            if global_counts.get(norm_token, 0) > 1:
                suffixed_token = f"{token.strip()}_{running_counts[norm_token]}"
            else:
                suffixed_token = token.strip()
            
            if in_increase:
                increase_suffixed.append(suffixed_token)
            elif in_decrease:
                decrease_suffixed.append(suffixed_token)
        
        last_token = token

    return increase_suffixed, decrease_suffixed

async def predict_causal_effects_for_sentences(
    open_router,
    hypothesis,
    sentences_data,
    sender_head,
    output_dir,
    require_reasoning: bool,
):
    """让LLM根据假设，为一批句子按每批5句预测因果方向"""
    print("正在根据假设按每批5句预测因果效应...")
    
    # --- 添加调试输出 ---
    if VERBOSE_DEBUG:
        print("=== 因果预测输入调试信息 ===")
        print(f"假设: {hypothesis[:100]}...")
        print(f"句子数据: {len(sentences_data)} 个句子")
        for key, data in list(sentences_data.items())[:3]:
            print(f"  {key}: {data['sentence']}")
        print("==============================")
    
    predictions = {}
    for batch in chunk_dict(sentences_data, 5):
        batch_predictions = await predict_for_sentence_batch_causal(
            open_router,
            hypothesis,
            batch,
            sender_head,
            output_dir,
            require_reasoning=require_reasoning,
        )
        predictions.update(batch_predictions)
    
    # --- 添加调试输出 ---
    if VERBOSE_DEBUG:
        print("=== 因果预测结果调试信息 ===")
        for key, result in predictions.items():
            print(f"{key}: direction={result.get('direction')}")
            if result.get("reasoning"):
                print(f"  推理: {result['reasoning'][:100]}...")
        print("==============================")
    
    return predictions

async def evaluate_single_hypothesis_terminal(
    hypothesis,
    open_router,
    open_router_highlight,
    validation_sentences,
    sender_head,
    preprocessed_attention_data,
    per_sentence_effects,
    default_direction,
    output_dir,
    with_reasoning,
):
    """Evaluate one hypothesis on the validation set (causal + attention)."""
    causal_predictions = await predict_causal_effects_for_sentences(
        open_router,
        hypothesis,
        validation_sentences,
        sender_head,
        output_dir,
        require_reasoning=with_reasoning,
    )
    sentence_id_map = {k: v.get("sentence_id") for k, v in validation_sentences.items()}
    causal_f1, causal_feedback = evaluate_causal_predictions(
        causal_predictions,
        default_direction,
        per_sentence=per_sentence_effects if per_sentence_effects else None,
        sentence_id_map=sentence_id_map,
    )

    real_attention_json = get_real_attention_pattern_for_sampling(
        validation_sentences, preprocessed_attention_data
    )
    formatted_output = format_examples_and_tokens(real_attention_json)
    highlighted_sentences_text = await underscore_important_tokens(
        open_router_highlight,
        sender_head[0],
        sender_head[1],
        hypothesis,
        formatted_output,
        output_dir,
    )
    predict_attention_json = convert_predict_attention_to_json(
        highlighted_sentences_text, real_attention_json
    )
    attention_f1, attention_f1_text = compare_attention_f1(
        predict_attention_json, real_attention_json
    )

    return {
        "hypothesis": hypothesis,
        "validation_scores": {
            "causal_f1": causal_f1,
            "attention_f1": attention_f1,
            "composite_f1": math.sqrt(causal_f1 * attention_f1) if causal_f1 > 0 and attention_f1 > 0 else 0.0,
        },
        "validation_details": {
            "causal_feedback": causal_feedback,
            "attention_feedback": attention_f1_text,
            "predicted_causal": causal_predictions,
            "predicted_attention": predict_attention_json,
            "real_attention": real_attention_json,
        },
    }
def evaluate_causal_predictions(predictions, ground_truth_direction=None, per_sentence=None, sentence_id_map=None):
    """评估因果方向预测(increase/decrease),返回 micro-F1(单标签任务等同于准确率)。

    ground_truth_direction: 全局方向（旧逻辑）
    per_sentence: 可选逐句真实方向 {sentence_id: delta_logit_diff}
    """
    feedback_details = []
    correct = 0
    count = 0

    if VERBOSE_DEBUG:
        print("=== 因果预测调试信息 ===")
        print(f"预测结果数量: {len(predictions)}")
        if per_sentence:
            print(f"真实方向: per_sentence({len(per_sentence)})")
        else:
            print(f"真实方向: {ground_truth_direction}")
        print("========================")

    for key, pred in predictions.items():
        pred_dir = pred.get("direction", "unknown")
        sid = None
        if sentence_id_map:
            sid = sentence_id_map.get(key)
        if sid is None:
            sid = pred.get("sentence_id", key)
        if per_sentence and str(sid) in per_sentence:
            delta = per_sentence[str(sid)]
            gt_dir = "increase" if delta >= 0 else "decrease"
        else:
            gt_dir = ground_truth_direction or "unknown"
        is_correct = pred_dir == gt_dir
        correct += int(is_correct)
        count += 1
        feedback_details.append(
            f"--- Sentence: {key} ---\n"
            f"  [Your Causal Prediction]: {pred_dir.upper()}\n"
            f"  [Real Causal Answer]:     {gt_dir.upper()}"
        )

    micro_f1 = (correct / count) if count > 0 else 0
    print(f"最终因果F1(micro): {micro_f1:.3f} (来自 {count} 个有效比较)")
    final_feedback = f"Overall Causal F1 (micro) for this batch: {micro_f1:.2f}\n\n" + "\n".join(feedback_details)
    return micro_f1, final_feedback
# -----------------------------------------------------------------------------
# 阶段三：注意力模式预测
# -----------------------------------------------------------------------------

def get_real_attention_pattern_for_sampling(sampled_sentences, preprocessed_attention_data):
    """为采样出的句子，从预处理数据中找到对应的真实注意力模式"""
    real_attention_patterns = []
    print(f"正在为 {len(sampled_sentences)} 个抽样句子获取预处理的注意力模式...")
    
    # --- 添加调试信息 ---
    if VERBOSE_DEBUG:
        print("=== 注意力数据调试信息 ===")
        print(f"preprocessed_attention_data 类型: {type(preprocessed_attention_data)}")
        print(f"preprocessed_attention_data 长度: {len(preprocessed_attention_data) if hasattr(preprocessed_attention_data, '__len__') else 'N/A'}")
        print(f"preprocessed_attention_data 前5个键: {list(preprocessed_attention_data.keys())[:5] if hasattr(preprocessed_attention_data, 'keys') else 'N/A'}")
        print("============================")
    
    for key, data in sampled_sentences.items():
        # --- 使用 sentence_id 进行查找 ---
        sentence_id = data.get('sentence_id')
        
        if sentence_id in preprocessed_attention_data:
            matched_data = preprocessed_attention_data[sentence_id]
        else:
            print(f"警告: 在预处理的注意力数据集中未找到 sentence_id '{sentence_id}'")
            continue
            
        real_attention_patterns.append({
            "key": key,
            "original_sentence": data['sentence'],
            "indirect_object": data['io'],
            "number_of_important_tokens": len(matched_data.get("top_k_tokens", [])),
            "important_tokens": matched_data.get("top_k_tokens", []) # 已经带后缀
        })
        
    print(f"最终找到 {len(real_attention_patterns)} 个匹配的注意力模式")
    return real_attention_patterns

def format_examples_and_tokens(example_attention_data):
    """格式化示例和token"""
    formatted_output = ""
    for example in example_attention_data:
        formatted_output += "Example: " + example["key"] + ": " + example["original_sentence"] + " {{DISAMB}}\n"
        formatted_output += f"Number of important tokens: {example['number_of_important_tokens']}\n"
        formatted_output += "disambiguation token: {{DISAMB}}\n"
    return formatted_output

async def underscore_important_tokens(open_router, layer, head, hypothesis_text, examples, output_dir):
    """标记重要token"""
    print(f"Predicting attention scores for layer {layer}, head {head}...")

    def _parse_expected_counts(examples_text: str) -> dict:
        pattern = r"Example:\s*(\S+)\s*:\s*.+?\nNumber of important tokens:\s*(\d+)"
        matches = re.findall(pattern, examples_text)
        return {key: int(num) for key, num in matches}

    expected_counts = _parse_expected_counts(examples)

    def _normalize_text(text: str) -> str:
        # Some models return literal "\\n" sequences; normalize to real newlines.
        if "\\n" in text:
            text = text.replace("\\n", "\n")
        return text

    def _extract_prediction_block(text):
        text = _normalize_text(text)
        match = re.search(r"\[PREDICTION\](.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        return text

    def validate_highlighted_output(text):
        text = _normalize_text(text)
        block = _extract_prediction_block(text)
        # Extract all predicted lines like "1_test: ...."
        pred_lines = re.findall(r"^\s*\S+_test\s*:\s*.+$", block, re.MULTILINE)
        if not pred_lines:
            print("Mismatch found: no predicted lines detected.")
            return False
        for line in pred_lines:
            key = line.split(":", 1)[0].strip()
            if key not in expected_counts:
                continue
            expected = expected_counts[key]
            actual = len(re.findall(r"<<[^<>]+>>", line))
            if expected != actual:
                print(f"Mismatch found: expected {expected}, got {actual} in: {line}")
                return False
        return True
    
    def extract_token_sentences(text):
        text = _normalize_text(text)
        block = _extract_prediction_block(text)
        return "\n".join(re.findall(r"^\s*\S+_test\s*:\s*.+$", block, re.MULTILINE))
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a meticulous AI researcher working on interpreting the function of a specific attention head in GPT2-small. "
                "You are given:\n"
                "- A hypothesis describing what this attention head might be doing in the garden path NP/Z v-trans task.\n"
                "- A set of example sentences with the disambiguation position annotated using {{DISAMB}}.\n"
                "- For each sentence, a line explicitly states the required number of important tokens to highlight, written as 'Number of important tokens: N'.\n\n"

                "**Crucial Interpretability Context:**\n"
                "You will be given an explanation from a 'Path Patching' experiment. You MUST interpret it correctly:\n"
                "- If the explanation says patching causes performance to 'decrease' or 'get worse', it means the head's original function is **contributory (aids the task)**.\n"
                "- If the explanation says patching causes performance to 'increase' or 'get better', it means the head's original function is **inhibitory (hinders the task)**.\n"
                "Your entire hypothesis must be consistent with this causal interpretation.\n\n"
                
                "Your job is to:\n"
                "1. Predict exactly N important tokens in each sentence, based on the hypothesis and your reasoning. This importance is considered from the perspective of the token right in front of the disambiguation position marked with {{DISAMB}}.\n"
                "2. Highlight the predicted tokens by enclosing them in double angle brackets like <<this>>.\n"
                "3. Ensure the number of highlighted tokens (<< >>) matches **exactly** the number given for that sentence.\n"
                "4. Treat each occurrence of a word as a unique token depending on its position in the sentence (e.g., 'David' at the beginning and 'David' at the end are different).\n"
                "5. The token you highlight with <<>> must be **in front of the disambiguation position marked with {{DISAMB}}**.\n\n"
                "MUST FOLLOW:\n"
                "- Output ONLY one [PREDICTION] block (no other text, no extra analysis).\n"
                "- In [PREDICTION], output one line per example in the exact format:\n"
                "  sentence_key: the sentence with exactly N <<highlighted>> tokens\n"
                "- Do not insert any extra lines between examples in [PREDICTION].\n"
                "- Double check that every sentence has **exactly N** << >> highlighted tokens.\n"
                "- If N=2, you must highlight exactly two tokens; one token is invalid.\n"
                "- If N=1, you must highlight exactly one token; two tokens is invalid.\n"
                "- Do not highlight the disambiguation token marked with {{DISAMB}} in <<>>.\n"
                "- Do not omit or add to the sentence.\n\n"
                "Example (N=2):\n"
                "[PREDICTION]\n"
                "1_test: <<While>> the <<pilot>> blamed the gardener who spoke too softly {{DISAMB}}\n"
            )
        },
        {
            "role": "user",
            "content": (
                f"You are given a hypothesis about attention head {head} in layer {layer} and several examples from the garden path task. "
                "For each example, please identify exactly the number of important tokens stated under 'Number of important tokens', based on the hypothesis. "
                "Highlight those tokens with << >> and strictly follow the output format.\n\n"
                "STRICT OUTPUT FORMAT:\n"
                "[PREDICTION]\n"
                "sentence_key: the sentence with exactly N <<highlighted>> tokens\n"
                "sentence_key: the sentence with exactly N <<highlighted>> tokens\n"
                "(one line per example, no extra lines)\n\n"
                "If you output ANY text outside [PREDICTION], the output is INVALID.\n"
                f"\n{hypothesis_text}"
                f"\n{examples}"
            )
        }
    ]
    
    last_output = None
    for attempt in range(5):
        highlighted = await open_router.generate(messages=messages, output_dir=output_dir)
        output = highlighted.text.strip()
        last_output = output
        log_raw_api_response(
            output_dir,
            "highlight_attempt",
            {
                "layer": layer,
                "head": head,
                "attempt": attempt + 1,
                "response": output,
            },
        )
        print(f"Attempt {attempt + 1}: Checking highlighted token counts...")
        if validate_highlighted_output(output):
            print("All token counts are correct.")
            highlighted_sentences = extract_token_sentences(output)
            if VERBOSE_DEBUG:
                print("Highlighted sentences:", highlighted_sentences)
            return highlighted_sentences
        else:
            print("Mismatch detected. Retrying...\n")

    if last_output:
        print("⚠️ 连续重试失败，使用最后一次输出继续流程。")
        return extract_token_sentences(last_output)
    raise ValueError("Failed to generate valid highlighted tokens after multiple retries.")

def convert_predict_attention_to_json(predict_attention_text, real_attention_json):
    """转换预测的注意力为JSON格式"""
    results = []
    lines = [line.strip() for line in predict_attention_text.split("\n") if line.strip()]
    lines = [line.split("{", 1)[0] for line in lines]
    real_map = {item["key"]: item for item in real_attention_json if "key" in item}
    
    for index, line in enumerate(lines):
        if ":" not in line: continue
        key, sentence = line.split(":", 1)
        highlighted_sentence = sentence
        important_tokens = []
        key_stripped = key.strip()
        real_entry = real_map.get(key_stripped)
        if real_entry is None:
            continue
        io = real_entry["indirect_object"]
        
        # 解析出带后缀的token
        parsed_inc, _ = _parse_and_suffix_tokens(real_entry["original_sentence"], sentence)
        
        results.append({
            "key": key_stripped,
            "highlighted_sentence": f"{highlighted_sentence.strip()} {{{{{io}}}}}",
            "important_tokens": parsed_inc,
            "indirect_object": io,
        })

    return results

def compare_attention_f1(predicted_attention, real_attention):
    """比较预测的注意力分数和真实的注意力分数 (F1)"""
    correct_count = 0
    total_predicted = 0
    total_real = 0
    
    real_map = {item['key']: item for item in real_attention}

    for pred in predicted_attention:
        pred_key = pred['key']
        if pred_key not in real_map:
            continue
        
        real = real_map[pred_key]
        pred_tokens = set(pred["important_tokens"])
        real_tokens = set(real["important_tokens"])
        
        correct_count += len(pred_tokens.intersection(real_tokens))
        total_predicted += len(pred_tokens)
        total_real += len(real_tokens)
        
    precision = correct_count / total_predicted if total_predicted > 0 else 0
    recall = correct_count / total_real if total_real > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    score_text = f"Score under F1 mode: {f1_score}"
    print(score_text, "\n")
    return f1_score, score_text

# -----------------------------------------------------------------------------
# 阶段四：统一的双维度精炼
# -----------------------------------------------------------------------------

def compute_composite_score(causal_f1, attention_f1):
    return math.sqrt(causal_f1 * attention_f1) if causal_f1 > 0 and attention_f1 > 0 else 0.0


def _extract_tagged_section(text: str, tag: str):
    if not text:
        return None
    tag_re = re.compile(
        rf"(?im)^\s*(?:\d+\s*[.)]\s*)?(?:[-*]\s*)?(?:\*\*\s*)?\[{re.escape(tag)}\]\s*[:：]?(?:\s*\*\*)?\s*"
    )
    fallback_tag_re = re.compile(
        rf"(?i)(?:\*\*\s*)?\[{re.escape(tag)}\]\s*[:：]?(?:\s*\*\*)?\s*"
    )
    tags_re = re.compile(
        r"(?im)^\s*(?:\d+\s*[.)]\s*)?(?:[-*]\s*)?(?:\*\*\s*)?\[[A-Z_]+\]\s*[:：]?(?:\s*\*\*)?"
    )
    match = tag_re.search(text) or fallback_tag_re.search(text)
    if not match:
        return None
    tail = text[match.end() :].strip()
    next_tag = tags_re.search(tail)
    if next_tag:
        tail = tail[: next_tag.start()].strip()
    return tail or None


async def _generate_refine_substep(open_router, system_prompt, user_prompt, output_dir):
    response = await open_router.generate(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        output_dir=output_dir,
    )
    prompt_log = f"---SYSTEM PROMPT---\n{system_prompt}\n\n---USER PROMPT---\n{user_prompt}"
    return response.text, prompt_log

async def refine_hypothesis_dual_dimension(
    open_router,
    old_hypothesis,
    sender_head,
    explanation,
    output_dir,
    causal_f1,
    attention_f1,
    causal_feedback=None,
    attention_feedback=None,
    optimize_only="dual",
    require_reasoning: bool = True,
):
    """按 attention -> causal -> synthesis 三步精炼假设"""
    print("按 attention -> causal -> synthesis 三步精炼假设...")

    if optimize_only == "causal":
        attention_feedback = None
        attention_f1 = 0.0
        composite_score = causal_f1
    elif optimize_only == "attention":
        causal_feedback = None
        causal_f1 = 0.0
        composite_score = attention_f1
    else:
        composite_score = compute_composite_score(causal_f1, attention_f1)

    print(f"因果F1: {causal_f1:.2f}, 注意力F1: {attention_f1:.2f}, 综合分数: {composite_score:.2f}")

    common_system_prompt = (
        "**--- Key Concepts in the Garden Path NP/Z v-trans Task (Crucial Background) ---**\n"
        "To understand the hypothesis, you must know these linguistic roles in the context of a sentence. However, in this task, we may not provide the whole sentence, which means that the disambiguation token at the end of the sentence may be hidden for prediction. For example, \"As the criminal shot the woman who told bad jokes [was/for].\"\n"
        "In this sentence:\n"
        "- **Subordinator:** the clause introducer (e.g., 'As', 'While', 'After', 'When').\n"
        "- **Subject (SUBJ):** the agent of the main clause (e.g., 'criminal').\n"
        "- **Ambiguous Verb (VERB):** the main verb whose transitivity drives the NP/Z ambiguity (e.g., 'shot' vs an intransitive verb).\n"
        "- **Object NP Head (OBJ_HEAD):** the head noun of the ambiguous object phrase (e.g., 'woman').\n"
        "- **Relative Pronoun / Verb (REL_PRON / REL_VERB):** the relative clause that follows the object (e.g., 'who told').\n"
        "- **Disambiguation Position:** the final token the model must choose (e.g., 'was' vs 'for'), which signals the correct parse.\n"
        "You must analyze the discrepancies between predicted and real model behavior to understand how the prior hypothesis mistakenly interprets the head's real function. "
        "Moreover, you must realize that the fact that an attention head pays close attention to a certain token does not contradict the fact that one of its functions is to suppress the expression of this token in the final output. The attention to this token may have the effect of reminding other heads to block this token. At the same time, the attention head's attention to a certain token is not necessarily related to its influence on the downstream heads."
        "\n\nReminder: The head may help OR hurt the task; avoid assuming it is beneficial."
    )

    attention_reasoning = ""
    attention_prompt_log = ""
    attention_raw_response = ""
    if attention_feedback and optimize_only != "causal":
        attention_user_prompt = (
            f"You are refining a hypothesis for Sender Head {sender_head}.\n"
            f"**Previous Hypothesis (Flawed):**\n{old_hypothesis}\n\n"
            "--- PERFORMANCE & FEEDBACK ANALYSIS ---\n"
            "**Core Metrics:**\n"
            f"- **Attention F1 Score (What it LOOKS AT): {attention_f1:.2f}**\n\n"
            "**Feedback Source: Direct Attention Prediction (What the head LOOKS AT)**\n"
            "This feedback shows how well the hypothesis predicted the head's own attention targets. A low score means the hypothesis is wrong about what the head is paying attention to.\n"
            f"{attention_feedback}\n\n"
            "**Your Task:**\n"
            "- Analyze the direct-attention errors and trade-offs shown above.\n"
            "- Identify what the previous hypothesis got right about the head's direct attention targets.\n"
            "- Identify the main attention-side mistakes or omissions.\n"
            "- Propose a revised attention-side mechanism that better matches the evidence.\n\n"
            "**Response Format (Strict):**\n"
            "1. **[REASONING]:** Start with this tag. Analyze only the direct-attention evidence, explain what the old hypothesis got right, what it got wrong, and what attention-side mechanism is better supported by the feedback. **The colon is mandatory** (must be `[REASONING]:`).\n"
        )
        attention_raw_response, attention_prompt_log = await _generate_refine_substep(
            open_router, common_system_prompt, attention_user_prompt, output_dir
        )
        attention_reasoning = _extract_tagged_section(attention_raw_response, "REASONING") or attention_raw_response.strip()

    causal_reasoning = ""
    causal_prompt_log = ""
    causal_raw_response = ""
    if causal_feedback and optimize_only != "attention":
        causal_user_prompt = (
            f"You are refining a hypothesis for Sender Head {sender_head}.\n"
            f"**Previous Hypothesis (Flawed):**\n{old_hypothesis}\n\n"
            "--- PERFORMANCE & FEEDBACK ANALYSIS ---\n"
            "**Core Metrics:**\n"
            f"- **Causal F1 Score (What it DOES): {causal_f1:.2f}**\n\n"
            "**Feedback Source: Causal Effect Prediction (What the head DOES)**\n"
            "This feedback reveals the head's true effect on the correct-vs-incorrect disambiguation logit difference. A low score means the hypothesis is wrong about the head's actual causal role.\n"
            f"{causal_feedback}\n\n"
            "**Your Task:**\n"
            "- Analyze the causal-effect errors and trade-offs shown above.\n"
            "- Analyze what the previous hypothesis got right about the head's causal effect on the disambiguation logit difference.\n"
            "- Identify the main causal-side mistakes or omissions.\n"
            "- Propose a revised causal-side mechanism that better matches the evidence.\n\n"
            "Important:\n"
            "- If the causal feedback consistently supports the current sign of effect, preserve that sign and focus on refining the mechanism rather than reversing the direction.\n"
            "- Only revise the sign if the feedback repeatedly and clearly contradicts it.\n\n"
            "**Response Format (Strict):**\n"
            "1. **[REASONING]:** Start with this tag. Analyze only the causal evidence, explain what the old hypothesis got right, what it got wrong, and what causal-side mechanism is better supported by the feedback. **The colon is mandatory** (must be `[REASONING]:`).\n"
        )
        causal_raw_response, causal_prompt_log = await _generate_refine_substep(
            open_router, common_system_prompt, causal_user_prompt, output_dir
        )
        causal_reasoning = _extract_tagged_section(causal_raw_response, "REASONING") or causal_raw_response.strip()

    synthesis_system_prompt = common_system_prompt
    synthesis_user_prompt = (
        f"You are refining a hypothesis for Sender Head {sender_head}.\n"
        f"**Previous Hypothesis (Flawed):**\n{old_hypothesis}\n\n"
        "--- PERFORMANCE & FEEDBACK ANALYSIS ---\n"
        "**Core Metrics:**\n"
        f"- **Causal F1 Score (What it DOES): {causal_f1:.2f}**\n"
        f"- **Attention F1 Score (What it LOOKS AT): {attention_f1:.2f}**\n"
        f"- **Composite F1 Score (Geometric Mean): {composite_score:.2f}**\n\n"
        "**Attention-side Reasoning:**\n"
        f"{attention_reasoning or 'N/A'}\n\n"
        "**Causal-side Reasoning:**\n"
        f"{causal_reasoning or 'N/A'}\n\n"
        "**Your Task:**\n"
        "- Use the two reasonings above as evidence, not as text to copy.\n"
        "- Reconcile the attention-side and causal-side evidence into one unified mechanism.\n"
        "- Produce one improved hypothesis describing this unified mechanism.\n\n"
        "Important:\n"
        "- Your final hypothesis must preserve the causal direction supported by the causal-side reasoning.\n"
        "- You may reframe the mechanism, but do not make the sign of the head's effect ambiguous.\n\n"
        "**Response Format (Strict):**\n"
        "1. **[REASONING]:** Start with this tag. Explicitly analyze how you reconciled the attention-side and causal-side evidence into one coherent mechanism.\n"
        "2. **[HYPOTHESIS]:** Start with this tag. Provide one clean standalone hypothesis describing this unified mechanism.\n"
        "    - This paragraph must clearly and abstractly describe what this head is doing (functionally).\n"
        "    - Do not include tokens, examples, scores, or error context in the `[HYPOTHESIS]` paragraph.\n"
        "    - The hypothesis should sound like a standalone description of the head's role in the model with no concessions or negations.\n"
        "    - The hypothesis should describe the dominant functional behavior using precise linguistic, semantic, or structural terminology. Avoid overly abstract or generic phrasing like 'tracks entities' or 'maintains context'.\n"
    )
    synthesis_raw_response, _synthesis_prompt_log = await _generate_refine_substep(
        open_router, synthesis_system_prompt, synthesis_user_prompt, output_dir
    )

    synthesis_reasoning = _extract_tagged_section(synthesis_raw_response, "REASONING") or "No reasoning found in response."
    new_hypothesis = _extract_tagged_section(synthesis_raw_response, "HYPOTHESIS") or old_hypothesis
    reasoning = (
        "=== ATTENTION REASONING ===\n"
        f"{attention_reasoning or 'N/A'}\n\n"
        "=== CAUSAL REASONING ===\n"
        f"{causal_reasoning or 'N/A'}\n\n"
        "=== SYNTHESIS REASONING ===\n"
        f"{synthesis_reasoning}"
    )
    print("精炼后的假设:", new_hypothesis)
    return new_hypothesis, reasoning

async def main():
    args = await parse_arguments()
    layer = args.layer
    head = args.head
    typename = args.typename
    output_dir = args.output_dir
    sender_head = (layer, head)
    data_source_dir = args.data_source_dir
    optimize_only = args.optimize_only
    validate_every = max(1, int(args.validate_every))
    validation_sample_size = max(0, int(args.validation_sample_size))
    test_sample_size = max(0, int(args.test_sample_size))
    test_all_validations = bool(args.test_all_validations)
    model = args.model
    epochs = 8
    train_batch_size = 10
    init_batch_size = 5
    with_reasoning = bool(args.with_reasoning and not args.no_reasoning)

    print(f"Using layer: {layer}, head: {head}, typename: {typename}")
    print(f"Reading data from: {data_source_dir}")
    print(f"Writing results to: {output_dir}")

    log_raw_api_response(output_dir, "run_start", {"layer": layer, "head": head, "model": model})

    attention_ground_truth_path = os.path.join(data_source_dir, "preprocessed_for_sampling.jsonl")
    preprocessed_attention_path = os.path.join(data_source_dir, "preprocessed_attention_scores.json")
    logit_effect_candidates = [
        os.path.join(data_source_dir, "heads_direct_effect_on_logit_difference.json"),
        os.path.join(os.path.dirname(os.path.abspath(data_source_dir)), "heads_direct_effect_on_logit_difference.json"),
    ]
    logit_effect_path = next((path for path in logit_effect_candidates if os.path.exists(path)), None)

    os.makedirs(output_dir, exist_ok=True)
    output_sentence_path = os.path.join(output_dir, f"{layer}.{head}_training_sentences.jsonl")
    final_results_path = os.path.join(output_dir, f"{layer}.{head}.json")
    final_rounds_summary_path = os.path.join(output_dir, "final_result_all_rounds.json")
    final_hypothesis_path = os.path.join(output_dir, "final_hypothesis.json")
    best_summary_path = os.path.join(output_dir, "best_hypothesis.json")
    iteration_predictions_path = os.path.join(output_dir, "iteration_predictions.jsonl")
    all_causal_gt_path = os.path.join(output_dir, "causal_ground_truth_all.json")
    all_attention_gt_path = os.path.join(output_dir, "attention_ground_truth_all.json")
    iteration_results_dir = os.path.join(output_dir, "iteration_results")
    validation_results_dir = os.path.join(output_dir, "validation_results")
    os.makedirs(iteration_results_dir, exist_ok=True)
    os.makedirs(validation_results_dir, exist_ok=True)

    print("--- 正在加载所有预处理数据 ---")
    preprocessed_attention_data = load_preprocessed_attention_dataset(preprocessed_attention_path)
    logit_effects, per_sentence_effects = load_head_logit_effects(logit_effect_path) if logit_effect_path else ({}, {})
    if not preprocessed_attention_data or not logit_effects:
        print("错误: 一个或多个预处理文件加载失败，程序终止。")
        return

    open_router = initialize_openrouter(model=model)
    open_router_highlight = initialize_openrouter(model=model)
    direction, _ = get_head_logit_direction(logit_effects, sender_head)
    head_key = f"{sender_head[0]}.{sender_head[1]}"
    per_sentence = per_sentence_effects.get(head_key, {}) if isinstance(per_sentence_effects, dict) else {}

    explanation = (
        "Task Context:\n"
        "- The task is garden-path NP/Z v-trans disambiguation.\n"
        "- Causal evidence is measured as the change in logit difference at the disambiguation position.\n"
        "- If patching/ablation makes logit_diff decrease, the head normally helps the task.\n"
        "- If patching/ablation makes logit_diff increase, the head normally hurts the task.\n"
        "- Use direct attention evidence and the logit-direction evidence together.\n"
        "- Do not assume the head is beneficial by default.\n\n"
        + format_head_logit_effect(layer, head, logit_effects)
    )

    train_dataset, validation_dataset, test_dataset = split_dataset(preprocessed_attention_data)

    async def evaluate_hypothesis_on_ids(hypothesis, sentence_ids, label):
        if not sentence_ids:
            return None
        sentences = build_sentences_from_ids(sentence_ids, preprocessed_attention_data, prefix=label.replace(".", "_"))
        if not sentences:
            return None
        result = await evaluate_single_hypothesis_terminal(
            hypothesis,
            open_router,
            open_router_highlight,
            sentences,
            sender_head,
            preprocessed_attention_data,
            per_sentence,
            direction,
            output_dir,
            with_reasoning,
        )
        if result:
            result["label"] = label
            result["sentence_ids"] = sentence_ids
        return result

    validation_history = []
    results = []

    def persist_validation_artifacts():
        validation_path = os.path.join(output_dir, "validation_results.json")
        try:
            with open(validation_path, "w", encoding="utf-8") as f:
                json.dump(validation_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"警告: 无法写入 validation_results.json: {e}")

        for item in validation_history:
            label = str(item.get("label", "validation")).strip() or "validation"
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
            item_path = os.path.join(validation_results_dir, f"{safe_label}.json")
            try:
                with open(item_path, "w", encoding="utf-8") as f:
                    json.dump(item, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"警告: 无法写入 {item_path}: {e}")

        if validation_history:
            best = max(validation_history, key=lambda r: r.get("validation_scores", {}).get("composite_f1", 0.0))
            try:
                with open(best_summary_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "head": f"{layer}.{head}",
                            "iteration": best.get("label"),
                            "best_hypothesis": best.get("hypothesis"),
                            "validation_scores": best.get("validation_scores"),
                            "source": "validation_history",
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception as e:
                print(f"警告: 无法写入 best_hypothesis.json: {e}")

    initial_hypothesis_candidates = []
    train_cursor = 0
    for candidate_idx in range(1, 4):
        init_ids, train_cursor = sample_sentence_ids(train_dataset, init_batch_size, start_idx=train_cursor)
        init_examples = build_sentences_from_ids(init_ids, preprocessed_attention_data, prefix=f"init{candidate_idx}_")
        example_sentence, example_activations, example_indirect_object = [], [], []
        for data in init_examples.values():
            top_tokens = preprocessed_attention_data.get(data["sentence_id"], {}).get("top_k_tokens", [])[:2]
            marked_sentence = data["sentence"]
            for token in top_tokens:
                surface = token.split("_")[0]
                marked_sentence = marked_sentence.replace(surface, f"<<{surface}>>", 1)
            example_sentence.append(f"{marked_sentence} {{{{DISAMB}}}}")
            example_activations.append("Activations: " + ", ".join([f'("{tok.split("_")[0]}", 1.00)' for tok in top_tokens]))
            example_indirect_object.append("{{DISAMB}}")

        init_causal_examples = build_direction_causal_examples_from_ids(
            init_ids,
            preprocessed_attention_data,
            direction,
            per_sentence_effects=per_sentence if per_sentence else None,
        )

        if optimize_only != "attention" and not init_causal_examples:
            print(f"错误: 初始候选 {candidate_idx} 需要因果示例，但未能构建示例，程序终止。")
            return
        if optimize_only != "causal" and not example_sentence:
            print(f"错误: 初始候选 {candidate_idx} 需要注意力示例，但未能构建示例，程序终止。")
            return

        hypothesis_text = await generate_hypothesis(
            open_router,
            layer,
            head,
            explanation,
            example_sentence,
            example_activations,
            example_indirect_object,
            init_causal_examples,
            optimize_only,
            output_dir,
            require_reasoning=with_reasoning,
        )
        hypothesis_analysis, extracted_hypothesis = extract_hypothesis_text(hypothesis_text)
        if not extracted_hypothesis:
            print(f"错误: 无法提取初始候选 {candidate_idx} 的假设，程序终止。")
            return
        initial_hypothesis_candidates.append(
            {
                "candidate_index": candidate_idx,
                "hypothesis": extracted_hypothesis,
                "hypothesis_analysis": hypothesis_analysis,
                "sampled_sentence_ids": init_ids,
            }
        )

    with open(os.path.join(output_dir, "initial_hypothesis_candidates.json"), "w", encoding="utf-8") as f:
        json.dump(initial_hypothesis_candidates, f, ensure_ascii=False, indent=2)

    val_ids_all = list(validation_dataset.keys())
    val_ids = val_ids_all[:validation_sample_size] if validation_sample_size > 0 and len(val_ids_all) > validation_sample_size else val_ids_all
    init_candidate_results = []
    for candidate in initial_hypothesis_candidates:
        candidate_result = await evaluate_hypothesis_on_ids(candidate["hypothesis"], val_ids, f"validation_epoch_0_candidate_{candidate['candidate_index']}")
        if candidate_result:
            candidate_result["candidate_index"] = candidate["candidate_index"]
            candidate_result["sampled_sentence_ids"] = candidate.get("sampled_sentence_ids", [])
            candidate_result["hypothesis_analysis"] = candidate.get("hypothesis_analysis", "")
            candidate_result["decision"] = "candidate_evaluated"
            init_candidate_results.append(candidate_result)
            validation_history.append(candidate_result)

    if not init_candidate_results:
        print("错误: 无法评估任何初始候选假设，程序终止。")
        return

    best_init_candidate = max(init_candidate_results, key=lambda r: r.get("validation_scores", {}).get("composite_f1", 0.0))
    init_val_result = copy.deepcopy(best_init_candidate)
    init_val_result["label"] = "validation_epoch_0_initial"
    init_val_result["decision"] = "selected_as_initial_start"
    validation_history.append(init_val_result)
    persist_validation_artifacts()

    current_hypothesis = init_val_result.get("hypothesis", "")
    current_analysis = init_val_result.get("hypothesis_analysis", "")
    initial_hypothesis = current_hypothesis
    previous_val_result = init_val_result

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_ids, train_cursor = sample_sentence_ids(train_dataset, train_batch_size, start_idx=train_cursor)
        example_sentences = build_sentences_from_ids(train_ids, preprocessed_attention_data, prefix=f"epoch{epoch}_")
        try:
            with open(output_sentence_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({f"epoch_{epoch}": example_sentences}, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"警告: 无法写入训练样本日志: {e}")

        causal_predictions = {}
        causal_f1 = 0.0
        causal_feedback = ""
        causal_ground_truth = {}
        if optimize_only != "attention":
            causal_predictions = await predict_causal_effects_for_sentences(
                open_router, current_hypothesis, example_sentences, sender_head, output_dir, require_reasoning=with_reasoning
            )
            sentence_id_map = {k: v.get("sentence_id") for k, v in example_sentences.items()}
            causal_f1, causal_feedback = evaluate_causal_predictions(
                causal_predictions, direction, per_sentence=per_sentence if per_sentence else None, sentence_id_map=sentence_id_map
            )
            for key, data in example_sentences.items():
                sid = data.get("sentence_id")
                if per_sentence and sid is not None and str(sid) in per_sentence:
                    delta = per_sentence[str(sid)]
                    causal_ground_truth[str(key)] = "increase" if delta >= 0 else "decrease"
                else:
                    causal_ground_truth[str(key)] = direction

        predict_attention_json = []
        real_attention_json = []
        attention_f1 = 0.0
        if optimize_only != "causal":
            real_attention_json = get_real_attention_pattern_for_sampling(example_sentences, preprocessed_attention_data)
            formatted_output = format_examples_and_tokens(real_attention_json)
            highlighted_sentences_text = await underscore_important_tokens(
                open_router_highlight, layer, head, current_hypothesis, formatted_output, output_dir
            )
            predict_attention_json = convert_predict_attention_to_json(highlighted_sentences_text, real_attention_json)
            attention_f1, _ = compare_attention_f1(predict_attention_json, real_attention_json)

        record = {
            "epoch": epoch,
            "hypothesis": current_hypothesis,
            "hypothesis_analysis": current_analysis,
            "scores": {"causal_f1": causal_f1, "attention_f1": attention_f1},
            "predicted_causal": causal_predictions,
            "causal_ground_truth": causal_ground_truth,
            "predicted_attention": predict_attention_json,
            "real_attention": real_attention_json,
        }
        results.append(record)
        with open(os.path.join(iteration_results_dir, f"iteration_{epoch}.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        attention_feedback = ""
        if optimize_only != "causal":
            attention_feedback_details = []
            real_map = {item["key"]: item for item in real_attention_json}
            for pred in predict_attention_json:
                pred_key = pred["key"]
                if pred_key not in real_map:
                    continue
                real = real_map[pred_key]
                pred_tokens = set(pred["important_tokens"])
                real_tokens = set(real["important_tokens"])
                sentence = real["original_sentence"]
                words, suffixed_map = _get_suffixed_word_map(sentence)
                feedback_words = []
                for i, word in enumerate(words):
                    suffixed_word = suffixed_map.get(i)
                    if not suffixed_word:
                        feedback_words.append(word)
                        continue
                    if suffixed_word in pred_tokens:
                        symbol = "✓" if suffixed_word in real_tokens else "✗"
                        feedback_words.append(f"<<{word}>>({symbol})")
                    else:
                        feedback_words.append(word)
                attention_feedback_details.append(f"  [Your Attention Prediction]: {' '.join(feedback_words)}")
                attention_feedback_details.append(f"  [Real Attention Answer]:     {sentence.replace('<<', '').replace('>>', '')}")
            attention_feedback = "\n".join(attention_feedback_details)

        current_hypothesis, current_analysis = await refine_hypothesis_dual_dimension(
            open_router,
            current_hypothesis,
            sender_head,
            explanation,
            output_dir,
            causal_f1,
            attention_f1,
            causal_feedback if optimize_only != "attention" else "",
            attention_feedback if optimize_only != "causal" else "",
            optimize_only=optimize_only,
            require_reasoning=with_reasoning,
        )

        if epoch % validate_every == 0:
            val_result = await evaluate_hypothesis_on_ids(current_hypothesis, val_ids, f"validation_epoch_{epoch}")
            if val_result:
                cur_score = float(val_result.get("validation_scores", {}).get("composite_f1", 0.0))
                prev_score = float(previous_val_result.get("validation_scores", {}).get("composite_f1", 0.0)) if previous_val_result else None
                if previous_val_result is not None and cur_score < prev_score:
                    rollback_hypothesis = previous_val_result.get("hypothesis", "")
                    if rollback_hypothesis:
                        current_hypothesis = rollback_hypothesis
                    val_result["decision"] = "rollback_to_previous_validation"
                    val_result["previous_validation_composite_f1"] = prev_score
                else:
                    val_result["decision"] = "accept_current"
                    previous_val_result = val_result
                validation_history.append(val_result)
                persist_validation_artifacts()

    final_hypothesis = current_hypothesis
    with open(final_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    persist_validation_artifacts()

    try:
        with open(iteration_predictions_path, "w", encoding="utf-8") as f:
            for entry in results:
                payload = {
                    "epoch": entry.get("epoch"),
                    "scores": entry.get("scores", {}),
                    "predicted_causal": entry.get("predicted_causal", {}),
                    "causal_ground_truth": entry.get("causal_ground_truth", {}),
                    "predicted_attention": entry.get("predicted_attention", []),
                    "real_attention": entry.get("real_attention", []),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"警告: 无法写入 iteration_predictions.jsonl: {e}")

    test_ids_all = list(test_dataset.keys())
    test_ids = test_ids_all[:test_sample_size] if test_sample_size > 0 and len(test_ids_all) > test_sample_size else test_ids_all
    test_result = await evaluate_hypothesis_on_ids(final_hypothesis, test_ids, "test_final")
    if test_result:
        with open(os.path.join(output_dir, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump(test_result, f, ensure_ascii=False, indent=2)

    same_as_final = (initial_hypothesis or "").strip() == (final_hypothesis or "").strip()
    if same_as_final:
        initial_test_result = dict(test_result) if isinstance(test_result, dict) else None
        if initial_test_result:
            initial_test_result["label"] = "test_initial"
            initial_test_result["shared_with_final"] = True
    else:
        initial_test_result = await evaluate_hypothesis_on_ids(initial_hypothesis, test_ids, "test_initial")
    if initial_test_result:
        with open(os.path.join(output_dir, "initial_test_results.json"), "w", encoding="utf-8") as f:
            json.dump(initial_test_result, f, ensure_ascii=False, indent=2)

    if test_all_validations and validation_history:
        all_val_test_results = []
        for item in validation_history:
            hyp = item.get("hypothesis", "")
            label = str(item.get("label", "validation"))
            if not hyp:
                continue
            res = await evaluate_hypothesis_on_ids(hyp, test_ids, f"test_from_{label}")
            if res:
                all_val_test_results.append(res)
        with open(os.path.join(output_dir, "test_results_all_validations.json"), "w", encoding="utf-8") as f:
            json.dump(all_val_test_results, f, ensure_ascii=False, indent=2)

    best_validation_entry = max(
        validation_history,
        key=lambda r: r.get("validation_scores", {}).get("composite_f1", 0.0),
    ) if validation_history else None

    try:
        final_rounds_summary = {
            "head": f"{layer}.{head}",
            "optimize_only": optimize_only,
            "model": model,
            "initial_hypothesis_candidates": initial_hypothesis_candidates,
            "initial_hypothesis": initial_hypothesis,
            "final_hypothesis": final_hypothesis,
            "best_validation_iteration": best_validation_entry.get("label") if best_validation_entry else "",
            "best_validation_entry": best_validation_entry,
            "selected_validation_last": previous_val_result,
            "initial_test_results": initial_test_result,
            "final_test_results": test_result,
        }
        with open(final_rounds_summary_path, "w", encoding="utf-8") as f:
            json.dump(final_rounds_summary, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"警告: 无法写入 final_result_all_rounds.json: {e}")

    try:
        final_hypothesis_payload = {
            "head": f"{layer}.{head}",
            "typename": typename,
            "source_test_file": os.path.join(output_dir, "test_results.json"),
            "final_hypothesis": (test_result or {}).get("hypothesis", final_hypothesis),
            "test_scores": (test_result or {}).get("validation_scores", {}),
        }
        with open(final_hypothesis_path, "w", encoding="utf-8") as f:
            json.dump(final_hypothesis_payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"警告: 无法写入 final_hypothesis.json: {e}")

    try:
        all_causal = {}
        if per_sentence:
            for sid, delta in per_sentence.items():
                all_causal[str(sid)] = {"direction": "increase" if float(delta) >= 0 else "decrease", "delta_logit_diff": float(delta)}
        else:
            for sid in preprocessed_attention_data.keys():
                all_causal[str(sid)] = {"direction": direction, "delta_logit_diff": None}
        with open(all_causal_gt_path, "w", encoding="utf-8") as f:
            json.dump({"head": f"{layer}.{head}", "source": "per_sentence" if per_sentence else "global_direction", "causal_ground_truth": all_causal}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"警告: 无法写入 causal_ground_truth_all.json: {e}")

    try:
        all_attention = [{"sentence_id": str(sid), "sentence_text": entry.get("sentence_text", ""), "important_tokens": entry.get("top_k_tokens", [])} for sid, entry in preprocessed_attention_data.items()]
        with open(all_attention_gt_path, "w", encoding="utf-8") as f:
            json.dump({"head": f"{layer}.{head}", "top_k": max((len(x.get("important_tokens", [])) for x in all_attention), default=0), "attention_ground_truth": all_attention}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"警告: 无法写入 attention_ground_truth_all.json: {e}")

if __name__ == "__main__":
    # 确保NLTK的punkt分词器已下载
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
        
    asyncio.run(main())
