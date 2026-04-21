import asyncio
import sys
import json
import os
import numpy as np
import re
import random
import sklearn
from collections import defaultdict
from sklearn.metrics import ndcg_score
import argparse
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:  
    print("未找到NLTK的'punkt'分词模型，正在尝试下载...")
    nltk.download('punkt')
    print("下载完成。")
    
sys.path.append("/home/wangziran/eap_auto/")
from api import OpenRouter

async def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Run automated hypothesis generation and refinement for a given attention head.")
    parser.add_argument("--layer", type=int, required=True, help="The sender head's layer.")
    parser.add_argument("--head", type=int, required=True, help="The sender head's number.")
    parser.add_argument("--rounds", type=int, required=True, help="The round number for saving results.")
    parser.add_argument("--typename", type=str, required=True, help="The typename of the head (e.g., s_inhibition_head).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--ground_truth_file", type=str, required=True, help="Path to the preprocessed ground truth JSON file.")
    parser.add_argument("--attention_ground_truth_file", type=str, help="Path to the preprocessed attention ground truth JSON file.")
    parser.add_argument("--causal_effects_file", type=str, required=True, help="Path to the causal effects JSON file (e.g., analysis_A0.1_to_B7.3_on_C9.6.json).")
    return parser.parse_args()


def split_dataset(full_dataset, validation_split=0.2, seed=42):
    """将完整数据集确定性地划分为训练集和验证集"""
    print(f"将数据集划分为 {1-validation_split:.0%} 训练集和 {validation_split:.0%} 验证集...")
    random.seed(seed)
    sentence_ids = list(full_dataset.keys())
    random.shuffle(sentence_ids)
    
    split_index = int(len(sentence_ids) * (1 - validation_split))
    train_ids = sentence_ids[:split_index]
    validation_ids = sentence_ids[split_index:]
    
    train_dataset = {sid: full_dataset[sid] for sid in train_ids}
    validation_dataset = {sid: full_dataset[sid] for sid in validation_ids}
    
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(validation_dataset)}")
    return train_dataset, validation_dataset

def initialize_openrouter(model: str = "claude-sonnet-4-20250514-thinking"):
    """初始化OpenRouter API"""
    api_key = "sk-Z3pwy4dD8WY2XZlbzch66NP5hQIoFKeU7KvI2XD8bQSyFVGO"
    return OpenRouter(model=model, api_key=api_key)

def extract_hypothesis_text(response_text):
    """从LLM的响应中提取[HYPOTHESIS]部分"""
    match = re.search(r"\[HYPOTHESIS\]:\s*(.*)", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    print("警告: 在LLM响应中未找到 '[HYPOTHESIS]:' 标签。")
    return response_text.strip() # 作为后备，返回全部文本

def normalize_token(token):
    """标准化token用于比较"""
    return token.strip().lower()

def _format_feedback_line(original_text, pred_tokens, real_tokens, markers):
    """
    【新功能】根据用户建议，生成更直观的、带对错标记(✓/✗)的反馈行。
    """
    pred_set = set(pred_tokens)
    real_set = set(real_tokens)
    
    # 使用正则表达式分词，保留标点
    words = re.findall(r"\w+|[^\w\s]", original_text)
    
    formatted_words = []
    
    # 为原始句子中的重复词添加后缀，以便精确匹配
    global_counts = defaultdict(int)
    for w in words:
        global_counts[normalize_token(w)] += 1
    
    running_counts = defaultdict(int)
    suffixed_words_map = {}
    for i, word in enumerate(words):
        norm_word = normalize_token(word)
        running_counts[norm_word] += 1
        if global_counts[norm_word] > 1:
            suffixed_word = f"{word.strip()}_{running_counts[norm_word]}"
        else:
            suffixed_word = word.strip()
        suffixed_words_map[i] = suffixed_word

    # 生成带标记的预测字符串
    for i, word in enumerate(words):
        suffixed_word = suffixed_words_map[i]
        
        if suffixed_word in pred_set:
            is_correct = suffixed_word in real_set
            symbol = "✓" if is_correct else "✗"
            formatted_words.append(f"{markers[0]}{word}{markers[1]}({symbol})")
        else:
            formatted_words.append(word)
            
    return " ".join(formatted_words)

# --- 阶段一：从数据生成初始假设 ---

def get_causal_effects_for_sampling(sampled_sentences, sender_head, top_k=1):
    """
    【已修改】为一批抽样的句子，逐句计算其真实的因果效应。
    现在直接使用 end_token_diff_attention。
    """
    causal_examples = []
    print(f"正在为 {len(sampled_sentences)} 个抽样句子，逐句计算真实的因果效应...")
    for sid, data in sampled_sentences.items():
        diff_vector = data.get('end_token_diff_attention')
        tokens = data.get('tokens', [])
        if not diff_vector or not tokens:
            continue

        # 确保长度一致
        min_len = min(len(diff_vector), len(tokens))
        avg_diff_vector = np.array(diff_vector[:min_len])
        tokens_padded = tokens[:min_len]
        
        token_changes = sorted(zip(tokens_padded, avg_diff_vector), key=lambda x: x[1], reverse=True)
        
        increase_tokens = [normalize_token(t) for t, s in token_changes[:top_k]]
        decrease_tokens = [normalize_token(t) for t, s in token_changes[-top_k:]]
        
        causal_examples.append({
            "sentence_text": data['sentence_info']['sentence_text'],
            "increase_tokens": increase_tokens,
            "decrease_tokens": decrease_tokens
        })
        
    return causal_examples


async def generate_initial_hypothesis(open_router, sender_head, causal_examples, explanation, output_dir):
    """
    【已重构】根据一批具体的因果效应示例，生成初始假设。
    引导LLM进行归纳推理，而不是解读统计数据。
    """
    print(f"为头 {sender_head} 生成初始假设，基于 {len(causal_examples)} 个具体示例...")
    
    # 动态构建示例字符串
    examples_str = ""
    for i, ex in enumerate(causal_examples, 1):
        examples_str += (
            f"**Example {i}:**\n"
            f"- **Sentence:** \"{ex['sentence_text']}\"\n"
            f"- **Result:** Corrupting the head causes attention to INCREASE on `{ex['increase_tokens']}` and DECREASE on `{ex['decrease_tokens']}`.\n\n"
        )

    system_prompt = (
        "**--- Key Concepts in the IOI Task (Crucial Background) ---**\n"
        "To understand the hypothesis, you must know these linguistic roles in the context of a sentence. However, in an IOI task, we may not provide the whole sentence, which means that IO at the end of the sentence may be hidden for prediction. For example, \"John and Mary went to the park. John gave a backpack to Mary.\"\n"
        "In this sentence:"
        "- **Subject (S):** The one performing the action. (e.g., 'John' gave the backpack).\n"
        "- **Indirect Object (IO):** The recipient of the action. Although the token is not provided directly at the end of the sentence, but you can infer from the context.(e.g., 'Mary' received the backpack).\n"
        "- **Direct Object (DO):** The object being transferred. (e.g., 'a backpack').\n"
    )
    user_prompt = (
        "You are an expert AI researcher tasked with discovering the function of an attention head in a transformer model (GPT2-small) for the Indirect Object Identification (IOI) task.\n\n"
        f"**Your Mission: Infer a general hypothesis for Sender Head {sender_head} based on the concrete experimental data below.**\n\n"
        
        "**Crucial Interpretability Context (Path Patching):**\n"
        "The data comes from a path patching experiment. You MUST interpret the results correctly:\n"
        "- An **INCREASE** in attention to a token proves the head's normal function is to **SUPPRESS** that token.\n"
        "- A **DECREASE** in attention to a token proves the head's normal function is to **PROMOTE** that token.\n\n"
        
        f"**Background on Downstream Heads:**\n{explanation}\n\n"
        
        "--- **Experimental Evidence (5 Random Samples)** ---\n"
        "Here are the results from 5 randomly selected sentences. Your task is to find a single, unified rule that explains all these examples.\n\n"
        f"{examples_str}"
        
        "**Your Task & Guidelines:**\n"
        "1.  **Synthesize, Don't Just List:** Analyze the pattern across all examples. What is the common linguistic or structural role of the suppressed tokens? What about the promoted tokens? Formulate a single, coherent hypothesis that explains this general mechanism.\n"
        "2.  **Explain the 'Why':** Your hypothesis must explain *why* the head performs this dual function (suppression and promotion) and how this behavior contributes to solving the IOI task.\n"
        "3.  **Be Precise & Falsifiable:** Use precise terminology (e.g., 'suppresses the subject token', 'promotes the indirect object token'). **AVOID** vague phrases like 'tracks entities' or 'maintains context'.\n"
        "4.  **Propose a Template-Based Hypothesis:** Your final hypothesis should clearly state both a suppression and a promotion effect. A good structure to follow is: 'The sender head (X, Y) suppresses attention to [CATEGORY A] while simultaneously promoting attention to [CATEGORY B] in order to [EXPLAIN THE GOAL].'\n\n"

        "**Response Format (Strict):**\n"
        "1.  **[REASONING]:** Start with this tag. Analyze the examples and explain your thought process for identifying the pattern. For example: 'In all examples, the token with increased attention (implying suppression) is consistently the subject of the sentence. The token with decreased attention (implying promotion) is the indirect object...'\n"
        "2.  **[HYPOTHESIS]:** Start with this tag. Provide the final, clean, standalone hypothesis based on your reasoning."
    )

    prompt = f"---SYSTEM PROMPT---\n{system_prompt}\n\n---USER PROMPT---\n{user_prompt}"
    
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_prompt}
    ]
    response = await open_router.generate(messages=messages, output_dir=output_dir)
    
    hypothesis = extract_hypothesis_text(response.text)
    print("初始假设已生成:", hypothesis)
    return hypothesis, prompt

# --- 阶段二：迭代验证与精炼 ---

async def predict_top_k_attenders_for_sentence(open_router, hypothesis, sid, sentence_data, sender_head, top_k, output_dir):
    sentence_text = sentence_data['sentence_text']

    prompt = (
        "You are a meticulous AI researcher tasked with predicting the direct attention pattern of a specific attention head.\n"
        "You are given a hypothesis about the head's function and a sentence. Your task is to predict which tokens the head will attend to most strongly **from the S2 token position** (the second subject in the IOI sentence structure. For example, in the following sentence 'Mary and Jack went to the park. Mary gave a book to ().', the second Mary is S2).\n\n"
        f"**Your Task & Guidelines:**\n"
        "1.  **Analyze the Hypothesis:** Carefully read the provided hypothesis for Head {sender_head}.\n"
        "2.  **Apply to Sentence:** Apply this hypothesis to the given sentence, considering the attention is cast **from the S2 position**.\n"
        "3.  **Predict Top-K Tokens:** Identify the **Top {top_k}** tokens that the head will attend to most strongly.\n"
        "4.  **Mark the Tokens:** You MUST highlight the most attended token with `<<token>>` and the second most attended token with `[[token]]`. If there is only one token to predict, use `<<token>>`.\n\n"
        "**MUST FOLLOW RULES:**\n"
        "- **Strict Formatting:** Your response MUST be the full, original sentence with the predicted tokens marked. Example: `ioi_9999: After the game, <<Mary>> and [[John]] played chess. Mary gave a book to`\n"
        "- **Exact Count:** You MUST predict and mark exactly {top_k} tokens.\n"
        "- **No Extra Text:** Do not include any reasoning, explanations, or any text other than the marked sentence.\n\n"
        "--- Example ---\n"
        "**Hypothesis:** 'This head attends to the Subject and the Indirect Object, with a strong preference for the Subject.'\n"
        "**Context:** Attention is from the S2 position ('John' in 'Mary and John...').\n"
        "**Sentence:** `ioi_9999: After the game, Mary and John played chess. Mary gave a book to`\n"
        "**Top_K:** 2\n"
        "**Correct Output:** `ioi_9999: After the game, <<Mary>> and [[John]] played chess. Mary gave a book to`\n"
        "--- End of Example ---\n\n"
        "**Now, it's your turn. Perform the real task below.**\n\n"
        f"**Hypothesis to Test for Head {sender_head}:**\n"
        f"\"{hypothesis}\"\n"
        "**Sentence to Analyze (Attention is from the S2 position):**\n"
        f"`{sid}: {sentence_text}`\n\n"
        f"**Top_K to Predict:** {top_k}\n\n"
        "**Your Prediction (full sentence with {top_k} marked tokens):**"
    )
    
    messages = [{"role": "user", "content": prompt}]
    response = await open_router.generate(messages=messages, output_dir=output_dir)
    
    marked_sentence = response.text.strip()
    
    # 我们将最重要的token视为"increase"，次重要的视为"decrease"来复用这个函数
    top1, top2 = _parse_and_suffix_tokens(sentence_text, marked_sentence)
    
    # 按顺序组合成预测列表
    predicted_tokens_ordered = top1 + top2
    
    return sid, {"predicted_tokens": predicted_tokens_ordered}

def _get_suffixed_word_map(original_text):
    """辅助函数：为句子的每个词生成带后缀的映射。"""
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

def evaluate_top_k_predictions(predictions, attention_ground_truth_map):
    """
    【重大重构】: 评估直接注意力预测。
    现在不再进行任何计算，而是直接从预处理好的 attention_ground_truth_map 中查找答案。
    """
    total_correct_tokens = 0
    total_predicted_tokens = 0
    total_real_tokens = 0
    
    feedback_details = []
    count = 0

    for sid, pred_data in predictions.items():
        if sid not in attention_ground_truth_map: continue
        
        # 1. 【核心修改】: 直接从预处理的地图中获取正确答案
        gt_entry = attention_ground_truth_map[sid]
        real_set = set(gt_entry.get('top_k_tokens', []))
        pred_set = set(pred_data.get('predicted_tokens', []))
        
        if not real_set: continue
        count += 1
        
        # 2. 计算F1分数 (逻辑不变)
        total_correct_tokens += len(pred_set.intersection(real_set))
        total_predicted_tokens += len(pred_set)
        total_real_tokens += len(real_set)

        # 3. 生成反馈字符串 (逻辑不变)
        original_sentence_text = gt_entry['sentence_text']
        words, suffixed_map = _get_suffixed_word_map(original_sentence_text)
        
        pred_words, real_words = [], []
        for i, word in enumerate(words):
            suffixed_word = suffixed_map.get(i)
            if not suffixed_word:
                pred_words.append(word)
                real_words.append(word)
                continue
            
            if suffixed_word in pred_set:
                symbol = "✓" if suffixed_word in real_set else "✗"
                pred_words.append(f"<<{word}>>({symbol})")
            else:
                pred_words.append(word)
            
            real_words.append(f"<<{word}>>" if suffixed_word in real_set else word)

        feedback_details.append(
            f"--- Sentence: {sid} ---\n"
            f"  [Your Prediction]: {' '.join(pred_words)}\n"
            f"  [Real Answer]:     {' '.join(real_words)}"
        )

    precision = total_correct_tokens / total_predicted_tokens if total_predicted_tokens > 0 else 0
    recall = total_correct_tokens / total_real_tokens if total_real_tokens > 0 else 0
    avg_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    final_feedback = (
        f"Overall Attention F1 Score for this batch: {avg_f1:.2f}\n\n" + 
        "\n".join(feedback_details)
    )
    
    return avg_f1, final_feedback

def sample_sentences_from_causal_dataset(causal_data, batch_size=5):
    """从causal_dataset中随机抽样句子及其相关数据"""
    sentence_ids = list(causal_data.keys())
    if len(sentence_ids) < batch_size:
        print(f"警告: 数据集中的句子总数({len(sentence_ids)})少于请求的batch_size({batch_size})。将使用所有句子。")
        sampled_ids = sentence_ids
    else:
        sampled_ids = random.sample(sentence_ids, batch_size)
    
    sampled_sentences = {sid: causal_data[sid] for sid in sampled_ids}
    return sampled_sentences

async def predict_for_single_sentence(open_router, hypothesis, sid, sentence_data, sender_head, top_k, output_dir, explanation):
    """
    为单个句子生成预测，包含完整的Prompt和推理过程。
    【已修改】新增 explanation 参数，并完全重写Prompt。
    【已修复】添加了system_prompt和响应解析逻辑。
    """
    sentence_text = sentence_data['sentence_info']['sentence_text']

    system_prompt = (
        "**--- Key Concepts in the IOI Task (Crucial Background) ---**\n"
        "To understand the hypothesis, you must know these linguistic roles in the context of a sentence. However, in an IOI task, we may not provide the whole sentence, which means that IO at the end of the sentence may be hidden for prediction. For example, \"John and Mary went to the park. John gave a backpack to Mary.\"\n"
        "In this sentence:"
        "- **Subject (S):** The one performing the action. (e.g., 'John' gave the backpack).\n"
        "- **Indirect Object (IO):** The recipient of the action. Although the token is not provided directly at the end of the sentence, but you can infer from the context.(e.g., 'Mary' received the backpack).\n"
        "- **Direct Object (DO):** The object being transferred. (e.g., 'a backpack').\n\n"
        "**--- Core Task & Causal Rules ---**\n"
        "You are a meticulous AI researcher applying a hypothesis to predict experimental outcomes in an IOI task. Your task is to predict the final outcome of a path patching experiment on a specific A->B->C circuit.\n"
        "You are given a hypothesis about a **Sender Head (A)**. You must predict how corrupting Head A changes the attention of the **final Receiver Head (C)**. This effect is mediated by an **Intermediate Head (B)**.\n\n"
        f"**The Specific Circuit Under Study:**\n{explanation}\n\n"
        "**Your Prediction Logic:**\n"
        "1.  Apply the hypothesis for Head A to the sentence.\n"
        "2.  Reason how corrupting Head A would change the behavior of Head B.\n"
        "3.  Reason how the change in Head B would, in turn, change the final attention pattern of Head C.\n"
        "4.  Mark the tokens on which Head C's attention changes:\n"
        "    - If corrupting Head A ultimately causes Head C to pay **MORE** attention to a token, mark it with `<<token>>`. (This implies Head A's normal function is to indirectly suppress it).\n"
        "    - If corrupting Head A ultimately causes Head C to pay **LESS** attention to a token, mark it with `[[token]]`. (This implies Head A's normal function is to indirectly promote it).\n\n"
        "**MUST FOLLOW RULES:**\n"
        f"- **Exact Count:** You **MUST** highlight exactly {top_k} token(s) for increase and {top_k} token(s) for decrease.\n"
        "- **Strict Formatting:** First, provide a step-by-step analysis in a `[REASONING]` block. Then, on a new line, provide the final marked sentence in a `[PREDICTION]` block.\n"
    )
    user_prompt = (
        "--- **Illustrative Example (How to perform the task):** ---\n"
        "To ensure you understand the method, here is a complete, fictional example. **DO NOT use the heads or functions from this example for your real task.** Your goal is to learn the *reasoning process*.\n\n"
        "**Fictional Scenario:**\n"
        "- **Circuit:** A(0.0) -> B(5.5) -> C(11.11)\n"
        "- **Head B (5.5) Function:** 'Positional Head'. It attends to tokens that are at the *end* of a clause (e.g., before a comma or period).\n"
        "- **Head C (11.11) Function:** 'Clause-End Mover'. It is *promoted* by Head B. When Head B attends strongly to a token, Head C's attention on that same token is *increased*.\n"
        "- **Hypothesis for Head A (0.0):** 'Head A is a verb inhibitor. It finds the main verb of a sentence and suppresses it.'\n\n"
        "**Example Sentence:** `ioi_9999: After the game, Mary and John played chess.`\n\n"
        "**Reasoning Walkthrough (Your thought process):**\n"
        "1.  **Analyze Head A's role:** The hypothesis says Head A suppresses the main verb, 'played'.\n"
        "2.  **Predict effect on Head B:** If we corrupt Head A, its suppression of 'played' is removed. Head B ('Positional Head') now sees 'played' more strongly. However, 'played' is not at the end of a clause, so Head B's attention pattern does not change.\n"
        "3.  **Predict effect on Head C:** Since Head B's behavior did not change, Head C's behavior also does not change. There is no effect.\n"
        "4.  **Apply Marking Rule:** Let's imagine a different hypothesis for A: 'Head A promotes the token before a comma'. Corrupting A would mean B sees 'game' less. Since B promotes C, C would also see 'game' less. The final marking would be `[[game]]`.\n\n"
        "**Correct Output for this Fictional Example (based on the second hypothesis):**\n"
        "`ioi_9999: After the [[game]], Mary and John played chess.`\n"
        "--- End of Example ---\n\n"
        "**Your Turn:**\n"
        f"**Hypothesis for Head {sender_head}:** {hypothesis}\n"
        "Now, using the **actual hypothesis and circuit information**, apply the same A->B->C reasoning process to the following sentence.\n\n"
        f"**Sentence to Analyze:**\n`{sid}: {sentence_text}`"
    )
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_prompt}
    ]
    response = await open_router.generate(messages=messages, output_dir=output_dir)

    # 【已修复】添加响应解析逻辑，以处理 [REASONING] 和 [PREDICTION]
    response_text = response.text
    prediction_match = re.search(r"\[PREDICTION\]:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
    
    marked_sentence = ""
    if prediction_match:
        marked_sentence = prediction_match.group(1).strip()
    else:
        # 作为后备，如果找不到[PREDICTION]，则尝试从响应中移除[REASONING]部分
        reasoning_removed = re.sub(r"\[REASONING\]:.*", "", response_text, flags=re.DOTALL | re.IGNORECASE)
        marked_sentence = reasoning_removed.strip()
        if not marked_sentence:
            print(f"警告: 在 {sid} 的响应中未找到 [PREDICTION] 标签，将使用整个响应进行解析。")
            marked_sentence = response_text # 最后的后备

    increase_tokens, decrease_tokens = _parse_and_suffix_tokens(sentence_text, marked_sentence)
    
    # 【已修复】返回三个值：sid, 解析后的预测, 以及原始响应文本
    return sid, {"increase": increase_tokens, "decrease": decrease_tokens}, response_text

async def predict_attention_changes_for_sentences(open_router, hypothesis, sentences_data, sender_head, top_k, output_dir, explanation):
    """
    让LLM根据假设，为一批句子并行预测注意力变化。
    【已修改】新增 explanation 参数。
    """
    print("正在根据假设并行预测每个句子的注意力变化...")
    
    tasks = []
    for sid, data in sentences_data.items():
        task = predict_for_single_sentence(
            open_router, hypothesis, sid, data, sender_head, top_k, output_dir, explanation
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    # 【已修复】正确地从三元组结果中解包
    predictions = {sid: pred_data for sid, pred_data, _ in results}
    raw_responses = {sid: raw_text for sid, _, raw_text in results}
    
    return predictions, raw_responses, None # 最后一个返回值保持兼容性

def evaluate_predictions(predictions, ground_truth_map, full_causal_dataset):
    """
    【重大重构】: 评估因果预测。
    【采纳用户建议】: 现在F1分数只基于 'decrease' (促进) 部分的预测准确率。
    """
    total_f1_decrease = 0
    feedback_details = []
    ground_truth_details = {sid: data["ground_truth"] for sid, data in ground_truth_map.items()}
    count = 0

    for sid, pred in predictions.items():
        if sid not in ground_truth_map: continue
        
        gt_entry = ground_truth_map[sid]
        real_inc_set = set(gt_entry['ground_truth'].get('increase', []))
        real_dec_set = set(gt_entry['ground_truth'].get('decrease', []))
        
        pred_inc_set = set(pred.get('increase', []))
        pred_dec_set = set(pred.get('decrease', []))
        
        # 【核心修改】: 只计算 decrease 部分的 F1 分数
        f1_decrease = (2 * len(pred_dec_set.intersection(real_dec_set))) / (len(pred_dec_set) + len(real_dec_set)) if (len(pred_dec_set) + len(real_dec_set)) > 0 else 0
        total_f1_decrease += f1_decrease
        count += 1

        # 生成反馈字符串的逻辑保持不变，以便提供完整的上下文
        original_sentence_text = full_causal_dataset[sid]['sentence_info']['sentence_text']
        words, suffixed_map = _get_suffixed_word_map(original_sentence_text)
        
        pred_words, real_words = [], []
        for i, word in enumerate(words):
            suffixed_word = suffixed_map.get(i)
            if not suffixed_word:
                pred_words.append(word)
                real_words.append(word)
                continue

            # 标记预测
            if suffixed_word in pred_inc_set:
                symbol = "✓" if suffixed_word in real_inc_set else "✗"
                pred_words.append(f"<<{word}>>({symbol})")
            elif suffixed_word in pred_dec_set:
                symbol = "✓" if suffixed_word in real_dec_set else "✗"
                pred_words.append(f"[[{word}]]({symbol})")
            else:
                pred_words.append(word)

            # 标记真实答案
            if suffixed_word in real_inc_set:
                real_words.append(f"<<{word}>>")
            elif suffixed_word in real_dec_set:
                real_words.append(f"[[{word}]]")
            else:
                real_words.append(word)

        feedback_details.append(
            f"--- Sentence: {sid} ---\n"
            f"  [Your Combined Prediction]: {' '.join(pred_words)}\n"
            f"  [Real Combined Answer]:   {' '.join(real_words)}"
        )

    avg_f1_decrease = total_f1_decrease / count if count > 0 else 0
    final_feedback = (
        f"Overall Causal F1 Score (Promote-Effect ONLY): {avg_f1_decrease:.2f}\n"
        "NOTE: Evaluation is now focused exclusively on correctly predicting the 'Promote' effect (marked with [[...]]).\n\n" 
        + "\n".join(feedback_details)
    )
    return avg_f1_decrease, final_feedback, ground_truth_details

async def refine_hypothesis_combined(open_router, old_hypothesis, sender_head, explanation, output_dir, f1_score, attention_f1_score, causal_feedback=None, attention_feedback=None):
    print("根据两个F1分数的组合情况，综合精炼假设...")

    # --- 1. 标准化分数 ---
    # 【已修改】: f1_score 现在直接代表 "Promote-Effect F1"
    causal_f1 = f1_score if f1_score is not None else 0.0
    attn_f1 = attention_f1_score if attention_f1_score is not None else 0.0
    
    # 综合分数逻辑简化
    composite_score = np.sqrt(causal_f1 * attn_f1) if attention_feedback else causal_f1
    print(f"因果F1 (Promote-Only): {causal_f1:.2f}, 注意力F1: {attn_f1:.2f}, 综合F1分数: {composite_score:.2f}")

    # --- 2. 【核心逻辑】根据双F1分数动态生成任务指令 ---
    task_guideline = ""
    # 定义阈值
    high_threshold = 0.7
    low_threshold = 0.2

    # 情况一：双高 -> 微调
    if causal_f1 >= high_threshold and attn_f1 >= high_threshold:
        task_guideline = (
            "**Your Task: Minor Refinement (High-Confidence State)**\n"
            "The current hypothesis is performing exceptionally well on both causal effect and direct attention prediction. **DO NOT make drastic changes.** Your goal is to make **minor, compatible adjustments** that might fix the remaining small errors without breaking what already works.\n"
            "- **Focus on Clarification:** Can you make the language more precise or concise?\n"
            "- **Focus on Exception Handling:** Can you add a clause that explains the few failed cases?\n"
            "**Preserve the core mechanism** of the old hypothesis. Your new hypothesis should be a slightly improved version, not a complete rewrite."
        )
    # 情况二：因果F1高，注意力F1低 -> 修复注意力预测
    elif causal_f1 >= high_threshold and attn_f1 < low_threshold:
        task_guideline = (
            "**CRITICAL TASK: Fix Flawed Attention Prediction**\n"
            f"The previous hypothesis is excellent at predicting the head's causal effect (Causal F1 = {causal_f1:.2f}), but it fails at predicting what the head actually LOOKS AT (Attention F1 = {attn_f1:.2f}).\n"
            "- **Analyze the 'Direct Attention Prediction' feedback (Source 1) above.** The 'Real Answer' shows what the head truly attends to.\n"
            "- **You MUST propose a new mechanism** that explains why the head attends to these real tokens, while **preserving the correct understanding of its causal effect**.\n"
            "- **Reconcile the conflict:** How can attending to *these* tokens lead to the causal effect you predicted correctly?"
        )
    # 情况三：注意力F1高，因果F1低 -> 修复因果预测
    elif attn_f1 >= high_threshold and causal_f1 < low_threshold:
        task_guideline = (
            "**TASK: Fix Flawed Causal Effect Prediction**\n"
            f"The previous hypothesis is excellent at predicting what the head LOOKS AT (Attention F1 = {attn_f1:.2f}), but it **completely fails** at predicting the head's causal effect (Causal F1 = {causal_f1:.2f}).\n"
            "- **Analyze the 'Causal Effect Prediction' feedback (Source 2) above.** The 'Real Answer' shows the head's true downstream impact.\n"
            "- **You MUST re-evaluate the head's purpose.** Given that it looks at these specific tokens, what is its true function? Why does it suppress/promote the tokens shown in the feedback?\n"
            "- **Your new hypothesis must explain this causal mechanism** without changing the correct understanding of what the head attends to."
        )
    # 情况四：双低或其他情况 -> 重大重构
    else:
        task_guideline = (
            "**Your Task: Major Refinement (Low-Confidence State)**\n"
            "The current hypothesis has evident flaws in one or both dimensions. Your goal is to propose a **single, unified hypothesis** that provides a better, more balanced explanation for BOTH phenomena.\n"
            "**Key Principle: Avoid Over-correction.** Do not become fixated on fixing one type of error to the extent that you abandon a correct understanding of the other. Your new hypothesis must be a robust improvement, not just a shift in focus."
        )

    # --- 3. 构建完整的Prompt (NDCG仅作为参考信息) ---
    system_prompt = (
        "**--- Key Concepts in the IOI Task (Crucial Background) ---**\n"
        "To understand the hypothesis, you must know these linguistic roles in the context of a sentence. However, in an IOI task, we may not provide the whole sentence, which means that IO at the end of the sentence may be hidden for prediction. For example, \"John and Mary went to the park. John gave a backpack to Mary.\"\n"
        "In this sentence:"
        "- **Subject (S):** The one performing the action. (e.g., 'John' gave the backpack).\n"
        "- **Indirect Object (IO):** The recipient of the action. Although the token is not provided directly at the end of the sentence, but you can infer from the context.(e.g., 'Mary' received the backpack).\n"
        "- **Direct Object (DO):** The object being transferred. (e.g., 'a backpack').\n"
        "You must analyze the discrepancies between predicted and real model behavior to understand how the prior hypothesis mistakenly interprets the head's real function. "
        "Moreover, you must realize that the fact that an attention head pays close attention to a certain token does not contradict the fact that one of its functions is to suppress the expression of this token in the final output. The attention to this token may have the effect of reminding other heads to block this token. At the same time, the attention head's attention to a certain token is not necessarily related to its influence on the downstream heads."
    )

    user_prompt_parts = [
        f"You are refining a hypothesis for Sender Head {sender_head}.\n"
        f"**Previous Hypothesis (Flawed):**\n{old_hypothesis}\n\n"
        "--- EXPERIMENTAL FEEDBACK ANALYSIS ---\n"
        "To formulate a better hypothesis, you must understand the **two different types of experiments** we ran and their distinct results.\n\n"
        "**Core Performance Metrics:**\n"
        f"- **Causal F1 Score (What it DOES): {causal_f1:.2f}**\n"
        f"- **Attention F1 Score (What it LOOKS AT): {attn_f1:.2f}**\n"
        f"- **Composite F1 Score (Geometric Mean): {composite_score:.2f}**\n\n"
        "Below is the detailed feedback from each experiment. Analyze all evidence carefully to formulate a better hypothesis."
    ]
    
    if attention_feedback:
        user_prompt_parts.append(
            "\n**Experiment 1: Direct Attention Analysis (What the head LOOKS AT at the S2 position)**\n"
            "In this experiment, we directly observed the attention patterns of Head {sender_head} itself, specifically when attending from the **S2 token position** (the second subject token in the IOI sentence structure). The feedback below shows how well your hypothesis predicted which tokens the head **pays attention to** from this specific position. A low score means the hypothesis is wrong about what the head is looking at.\n"
            f"{attention_feedback}\n"
        )

    if causal_feedback:
        user_prompt_parts.append(
            "\n**Experiment 2: Causal Effect Analysis (What the head DOES via a circuit)**\n"
            "In this experiment, we measured the head's **downstream causal effect**. We did this using a path patching experiment on the **A->B->C (0.1 -> 7.3 -> 9.6) circuit**. This means we corrupted Head {sender_head} (A) and measured the final attention change on Head 9.6 (C). The feedback below reveals the head's true function (suppression/promotion) within this specific circuit. A low score means the hypothesis is wrong about the head's actual effect.\n"
            f"{causal_feedback}\n"
        )
    
    user_prompt_parts.append(f"\n{task_guideline}\n")
    
    user_prompt_parts.append(
        "**Response Format (Strict):**\n"
        "1.  **[REASONING]:** Start with this tag. First, explicitly analyze the conflict and performance trade-offs: 'The direct attention feedback suggests the head looks at X, while the causal feedback proves its effect is Y. The previous hypothesis performed well on causal effect but poorly on attention. It failed because it only accounted for Y.' Then, propose a mechanism that reconciles this conflict in a balanced way. For example: 'A better explanation that preserves the correct causal understanding is that the head attends to X *in order to* gather information to perform action Y.'\n"
        "2.  **[HYPOTHESIS]:** Start with this tag. Provide the final, clean, standalone hypothesis that describes this unified, balanced mechanism.\n"
        "    - This paragraph must clearly and abstractly describe what this head is doing (functionally).\n"
        "    - Do not include tokens, examples, scores, or error context in the `[HYPOTHESIS]` paragraph.\n"
        "    - The hypothesis should sound like a standalone description of the head's role in the model with no concessions or negations.\n"
        "    - The hypothesis should describe the dominant functional behavior using precise linguistic, semantic, or structural terminology. Avoid overly abstract or generic phrasing like 'tracks entities' or 'maintains context'."
    )
    
    user_prompt = "\n".join(user_prompt_parts)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = await open_router.generate(messages=messages, output_dir=output_dir)
    
    response_text = response.text
    reasoning_match = re.search(r"(.*)\[HYPOTHESIS\]:", response_text, re.DOTALL | re.IGNORECASE)
    hypothesis_match = re.search(r"\[HYPOTHESIS\]:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning found in response."
    new_hypothesis = hypothesis_match.group(1).strip() if hypothesis_match else response_text.strip()
    
    prompt_log = f"---SYSTEM PROMPT---\n{system_prompt}\n\n---USER PROMPT---\n{user_prompt}"
    raw_response_log = response.text
    
    print("精炼后的假设:", new_hypothesis)
    return new_hypothesis, reasoning, prompt_log, raw_response_log


def _parse_and_suffix_tokens(original_sentence_text: str, marked_sentence: str, increase_markers: tuple = ("<<", ">>"), decrease_markers: tuple = ("[[", "]]")) -> tuple[list[str], list[str]]:
    """
    【最终修正版】采纳用户建议，使用更简洁、更稳健的last_token状态机逻辑。
    """
    # 1. 预计算全局token计数，用于判断是否需要加后缀
    original_tokens = word_tokenize(original_sentence_text)
    global_counts = defaultdict(int)
    for token in original_tokens:
        global_counts[normalize_token(token)] += 1

    # 2. 使用NLTK对带标记的句子进行分词
    marked_tokens = word_tokenize(marked_sentence)

    # 3. 初始化状态机和结果列表
    increase_suffixed, decrease_suffixed = [], []
    running_counts = defaultdict(int)
    
    in_increase = False
    in_decrease = False
    last_token = "" # 用于检测连续标记

    # 4. 【已修正】遍历分词后的token列表，使用更清晰的状态机逻辑
    for token in marked_tokens:
        # a. 更新状态机
        # 检查连续的两个token是否构成了开始标记
        if token == increase_markers[0][0] and last_token == increase_markers[0][0]: # '<<'
            in_increase = True
        elif token == decrease_markers[0][0] and last_token == decrease_markers[0][0]: # '[['
            in_decrease = True
        # 检查连续的两个token是否构成了结束标记
        elif token == increase_markers[1][0] and last_token == increase_markers[1][0]: # '>>'
            in_increase = False
        elif token == decrease_markers[1][0] and last_token == decrease_markers[1][0]: # ']]'
            in_decrease = False

        # b. 如果在标记内部，则处理当前token
        # 确保当前token不是标记本身
        is_marker_char = token in ['<', '>', '[', ']']
        if not is_marker_char and (in_increase or in_decrease):
            norm_token = normalize_token(token)
            running_counts[norm_token] += 1
            
            # c. 根据全局计数决定是否添加后缀
            if global_counts.get(norm_token, 0) > 1:
                suffixed_token = f"{token.strip()}_{running_counts[norm_token]}"
            else:
                suffixed_token = token.strip()
            
            # d. 将带后缀的token添加到对应的结果列表
            if in_increase:
                increase_suffixed.append(suffixed_token)
            elif in_decrease:
                decrease_suffixed.append(suffixed_token)
        
        last_token = token

    return increase_suffixed, decrease_suffixed


async def evaluate_single_hypothesis(hypothesis, open_router, validation_dataset, sender_head, top_k_causal, top_k_attention, attention_ground_truth, output_dir, explanation, causal_ground_truth_map): 
    """
    【新增辅助函数】在验证集上完整评估单个假设，用于并行化。
    【已修改】移除了NDCG相关的逻辑。
    """
    print(f"  - 开始评估假设: \"{hypothesis[:80]}...\"")
    # 在验证集上运行因果预测
    val_causal_preds, _, _ = await predict_attention_changes_for_sentences(open_router, hypothesis, validation_dataset, sender_head, top_k_causal, output_dir, explanation)
    val_f1, _, val_causal_gt = evaluate_predictions(
        val_causal_preds, 
        causal_ground_truth_map, # 直接使用全局加载的答案地图
        validation_dataset       # 传入原始数据以生成反馈字符串
    )

    # 在验证集上运行注意力预测
    val_f1_att = 0.0
    val_att_preds = None
    val_att_gt = None
    if attention_ground_truth:
        val_sids_att = {sid for sid in validation_dataset if sid in attention_ground_truth}
        if val_sids_att:
            tasks = [predict_top_k_attenders_for_sentence(open_router, hypothesis, sid, attention_ground_truth[sid], sender_head, top_k_attention, output_dir) for sid in val_sids_att]
            results = await asyncio.gather(*tasks)
            val_att_preds = {sid: result for sid, result in results}
            val_f1_att, _ = evaluate_top_k_predictions(val_att_preds, attention_ground_truth)
            val_att_gt = {sid: attention_ground_truth[sid] for sid in val_sids_att}
    
    print(f"  - 评估完成: Causal F1={val_f1:.2f}, Attn F1={val_f1_att:.2f}")
    
    # 返回该假设的完整验证结果
    return {
        "hypothesis": hypothesis,
        "validation_scores": {
            "causal_f1": val_f1, 
            "direct_attention_f1": val_f1_att
        },
        "validation_details": {
            "causal_predictions": val_causal_preds, "causal_ground_truth": val_causal_gt,
            "attention_predictions": val_att_preds, "attention_ground_truth": val_att_gt
        }
    }
    
async def main():
    # 1. 初始化和加载数据
    args = await parse_arguments()
    sender_head = (args.layer, args.head)
    output_dir = args.output_dir
    MAX_ITERATIONS = 7 
    top_k_attention = 2
    top_k_causal = 1

    # --- 变量初始化，用于try/finally块 ---
    full_run_log = []
    prediction_log_content = []
    all_hypotheses = set()
    validation_results = []
    initial_hypothesis_prompt = "未生成初始假设。"
    current_hypothesis = ""
    
    # 将所有文件加载和主逻辑都放入try块，以便在中断时也能安全保存
    try:
        # --- 步骤 1: 加载原始因果数据 (仅用于初始抽样) ---
        print("--- 正在加载数据文件 ---")
        try:
            with open(args.causal_effects_file, "r") as f:
                causal_data_list = json.load(f)
                full_causal_dataset = {item['sentence_id']: item for item in causal_data_list}
            print(f"[1/3] 成功加载 {len(full_causal_dataset)} 条原始因果数据 (来自: {args.causal_effects_file})")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"错误: 无法加载原始因果数据集 {args.causal_effects_file}: {e}")
            return

        # --- 步骤 2: 加载预处理好的因果效应答案 ---
        try:
            with open(args.ground_truth_file, "r") as f:
                gt_data_list = json.load(f)
                causal_ground_truth_map = {item['sentence_id']: item for item in gt_data_list}
            print(f"[2/3] 成功加载 {len(causal_ground_truth_map)} 条预处理好的因果效应答案 (来自: {args.ground_truth_file})")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"错误: 无法加载预处理的因果答案文件 {args.ground_truth_file}: {e}")
            return
        
        # --- 步骤 3: 加载预处理好的直接注意力答案 (可选) ---
        attention_ground_truth_map = None
        if args.attention_ground_truth_file:
            try:
                with open(args.attention_ground_truth_file, "r") as f:
                    gt_data_list = json.load(f)
                    attention_ground_truth_map = {item['sentence_id']: item for item in gt_data_list}
                print(f"[3/3] 成功加载 {len(attention_ground_truth_map)} 条预处理好的直接注意力答案 (来自: {args.attention_ground_truth_file})")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"警告: 无法加载注意力答案文件 {args.attention_ground_truth_file}: {e}。将跳过直接注意力评估。")
        else:
            print("[3/3] 警告: 未提供 --attention_ground_truth_file，将跳过直接注意力评估。")

        # --- 数据集划分和API初始化 ---
        train_dataset, validation_dataset = split_dataset(full_causal_dataset)
        open_router = initialize_openrouter()
        
        # --- 生成初始假设 ---
        print("\n--- 生成初始假设 ---")
        explanation = f"""
            Background Context: We are studying Sender Head {sender_head}. Its causal effect is measured by how it affects the attention pattern of a downstream head (9.6) via an intermediate head (7.3).
            - **Known function of intermediate head (7.3):** This head's primary role is to attend to the Subject (S) token and suppress its consideration by later heads.
            - **Known function of downstream (9.6):** This head is crucial for the final answer. It attends to the correct Indirect Object (IO) token and copies it to the final position.
            
            Your task is to hypothesize the function of Sender Head {sender_head} by explaining how it influences this A->B->C ({sender_head[0]}.{sender_head[1]} -> 7.3 -> 9.6) circuit. The data shows you the final outcome on Head 9.6.
            """
        
        sampled_sentences = sample_sentences_from_causal_dataset(train_dataset, batch_size=5)
        causal_examples = get_causal_effects_for_sampling(sampled_sentences, sender_head, top_k=top_k_causal)
        
        if not causal_examples:
            print("无法为抽样句子计算因果效应，程序终止。")
            return

        current_hypothesis, initial_hypothesis_prompt = await generate_initial_hypothesis(open_router, sender_head, causal_examples, explanation, output_dir)
        
        if not current_hypothesis:
            print("无法生成初始假设，程序终止。")
            return
        
        all_hypotheses = {current_hypothesis}

        # --- 统一的并行迭代精炼循环 ---
        print("\n--- 开始统一的并行迭代精炼循环 ---")
        for i in range(1, MAX_ITERATIONS + 1):
            print(f"\n--- Iteration {i}/{MAX_ITERATIONS} ---")
            iteration_log = {"iteration": i, "hypothesis_before": current_hypothesis}

            prediction_log_content.append(f"--- Iteration {i} ---\n")
            prediction_log_content.append(f"Hypothesis: {current_hypothesis}\n")
            
            # 1. 在测试集上进行因果预测
            test_sentences_batch = sample_sentences_from_causal_dataset(train_dataset, batch_size=5)
            causal_predictions, raw_causal_responses, _ = await predict_attention_changes_for_sentences(
                open_router, current_hypothesis, test_sentences_batch, sender_head, top_k_causal, output_dir, explanation
            )
            
            # 2. 在测试集上进行直接注意力预测 (如果提供了答案文件)
            attention_predictions = None
            if attention_ground_truth_map:
                # 只对那些在答案地图中存在的句子进行预测
                test_sids_for_attention = {sid for sid in test_sentences_batch if sid in attention_ground_truth_map}
                if test_sids_for_attention:
                    tasks = [predict_top_k_attenders_for_sentence(open_router, current_hypothesis, sid, attention_ground_truth_map[sid], sender_head, top_k_attention, output_dir) for sid in test_sids_for_attention]
                    results = await asyncio.gather(*tasks)
                    attention_predictions = {sid: result for sid, result in results}

            # 3. 评估因果预测
            f1_score, causal_feedback, causal_ground_truth_details = evaluate_predictions(
                causal_predictions, 
                causal_ground_truth_map, 
                full_causal_dataset # 传入原始数据以生成反馈字符串
            )
            print(f"Iteration {i} Causal F1 Score: {f1_score:.2f}")
            iteration_log["causal_f1"] = f1_score
            iteration_log["causal_feedback"] = causal_feedback

            # 记录因果预测日志
            prediction_log_content.append("\n** Causal Effect Predictions **\n")
            for sid in test_sentences_batch:
                pred = causal_predictions.get(sid, {})
                gt = causal_ground_truth_map.get(sid, {}).get("ground_truth", {})
                prediction_log_content.append(f"  Sentence: {sid}: {full_causal_dataset[sid]['sentence_info']['sentence_text']}")
                prediction_log_content.append(f"    - LLM Prediction: Promote={pred.get('decrease', [])}, Suppress={pred.get('increase', [])}")
                prediction_log_content.append(f"    - Ground Truth:   Promote={gt.get('decrease', [])}, Suppress={gt.get('increase', [])}\n")
                
            # 4. 评估直接注意力预测
            f1_score_att = 0.0
            attention_feedback = None
            if attention_ground_truth_map and attention_predictions:
                f1_score_att, attention_feedback = evaluate_top_k_predictions(attention_predictions, attention_ground_truth_map)
                print(f"Iteration {i} Direct Attention F1: {f1_score_att:.2f}")
                iteration_log.update({"attention_f1": f1_score_att, "attention_feedback": attention_feedback})
                
                # 记录直接注意力预测日志
                prediction_log_content.append("\n** Direct Attention Predictions **\n")
                for sid, pred_data in attention_predictions.items():
                    gt_data = attention_ground_truth_map.get(sid, {})
                    prediction_log_content.append(f"  Sentence: {sid}: {gt_data.get('sentence_text', '')}")
                    prediction_log_content.append(f"    - LLM Prediction: {pred_data.get('predicted_tokens', [])}")
                    prediction_log_content.append(f"    - Ground Truth:   {gt_data.get('top_k_tokens', [])}\n")
                    
            # 5. 检查收敛或终止条件
            if f1_score > 0.85 and (not attention_ground_truth_map or f1_score_att > 0.85):
                print("假设收敛成功。")
                full_run_log.append(iteration_log)
                break
            
            if i == MAX_ITERATIONS:
                full_run_log.append(iteration_log)
                break

            # 6. 精炼假设
            current_hypothesis, reasoning, prompt, raw_response = await refine_hypothesis_combined(
                open_router, current_hypothesis, sender_head, explanation, output_dir, 
                f1_score=f1_score,
                attention_f1_score=f1_score_att,
                causal_feedback=causal_feedback, 
                attention_feedback=attention_feedback
            )
            all_hypotheses.add(current_hypothesis)
            iteration_log.update({
                "hypothesis_after": current_hypothesis, 
                "refinement_reasoning": reasoning, 
                "refinement_prompt": prompt, 
                "refinement_raw_response": raw_response
            })
            full_run_log.append(iteration_log)

        # --- 最终验证 ---
        print(f"\n--- 迭代完成，共产生 {len(all_hypotheses)} 个独立假设 ---")
        print("--- 开始在验证集上并行评估所有候选假设 (这可能需要一些时间)... ---")

        validation_tasks = [
            evaluate_single_hypothesis(
                hypothesis, open_router, validation_dataset, sender_head, 
                top_k_causal, top_k_attention, attention_ground_truth_map, 
                output_dir, explanation, causal_ground_truth_map
            ) 
            for hypothesis in list(all_hypotheses)
        ]
        
        validation_results = await asyncio.gather(*validation_tasks)

    except KeyboardInterrupt:
        print("\n\n检测到用户中断 (Ctrl+C)。正在保存当前所有进度...")
    
    finally:
        # --- 文件保存 ---
        print("\n--- 正在保存当前进度到文件 (即使被中断)... ---")
        
        final_log_entry = {
            "head": f"{sender_head[0]}.{sender_head[1]}",
            "typename": args.typename,
            "status": "Interrupted by user" if 'KeyboardInterrupt' in str(sys.exc_info()) else "Completed",
            "last_hypothesis": current_hypothesis,
            "initial_hypothesis_prompt": initial_hypothesis_prompt,
            "full_run_log": full_run_log,
            "all_hypotheses_validation_results": validation_results
        }
        
        prediction_log_path = os.path.join(output_dir, f"prediction_log_round_{args.rounds}.txt")
        try:
            with open(prediction_log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(prediction_log_content))
            print(f"详细的逐轮预测日志已保存至: {prediction_log_path}")
        except Exception as e:
            print(f"错误: 无法写入预测日志文件: {e}")
                
        final_log_path = os.path.join(output_dir, f"final_result_round_{args.rounds}.json")
        try:
            with open(final_log_path, "w") as f:
                json.dump(final_log_entry, f, indent=2, ensure_ascii=False)
            print(f"\n所有运行数据和最终的并行验证结果已统一保存至: {final_log_path}")
        except Exception as e:
            print(f"错误: 无法写入最终结果JSON文件: {e}")

if __name__ == "__main__":
    asyncio.run(main())
