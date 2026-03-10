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

sys.path.append("/data31/private/wangziran/eap_auto/")
from api import OpenRouter

async def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Run automated hypothesis generation and refinement for a given attention head.")
    parser.add_argument("--layer", type=int, required=True, help="The sender head's layer.")
    parser.add_argument("--head", type=int, required=True, help="The sender head's number.")
    parser.add_argument("--rounds", type=int, required=True, help="The round number for saving results.")
    parser.add_argument("--typename", type=str, required=True, help="The typename of the head (e.g., s_inhibition_head).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
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
def initialize_openrouter(model: str = "gpt-4o"):
    """初始化OpenRouter API"""
    api_key = "sk-t9ooUIrk73Zg4s72dFCf2QAYWrNsobTW1gT8P7AG7m1r4Wbd"
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
    【新函数】取代 get_overall_causal_effect。
    为一批抽样的句子，逐句计算其真实的因果效应。
    """
    causal_examples = []
    print(f"正在为 {len(sampled_sentences)} 个抽样句子，逐句计算真实的因果效应...")

    for sid, data in sampled_sentences.items():
        real_diffs = data.get('diff_vectors', {})
        tokens = data.get('tokens', [])
        if not real_diffs or not tokens:
            continue

        # 计算该句子上sender到所有receiver的平均效应
        sentence_vectors = [np.array(v) for k, v in real_diffs.items() if k.startswith(f"{sender_head[0]}.{sender_head[1]}->") and v]
        if not sentence_vectors:
            continue

        # 确保长度一致
        min_len = min(len(v) for v in sentence_vectors)
        avg_diff_vector = np.mean([v[:min_len] for v in sentence_vectors], axis=0)
        
        tokens_padded = tokens[:min_len]
        token_changes = sorted(zip(tokens_padded, avg_diff_vector), key=lambda x: x[1], reverse=True)
        
        # 找出效应最强的token
        increase_tokens = [normalize_token(t) for t, s in token_changes[:top_k]]
        decrease_tokens = [normalize_token(t) for t, s in token_changes[-top_k:]]
        
        causal_examples.append({
            "sentence_text": data['sentence_text'],
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
        "You are given a hypothesis about the head's function and a sentence. Your task is to predict which tokens the head will attend to most strongly from the final token position of the sentence.\n\n"
        f"**Your Task & Guidelines:**\n"
        "1.  **Analyze the Hypothesis:** Carefully read the provided hypothesis for Head {sender_head}.\n"
        "2.  **Apply to Sentence:** Apply this hypothesis to the given sentence.\n"
        "3.  **Predict Top-K Tokens:** Identify the **Top {top_k}** tokens that the head will attend to most strongly from the final position.\n"
        "4.  **Mark the Tokens:** You MUST highlight the most attended token with `<<token>>` and the second most attended token with `[[token]]`. If there is only one token to predict, use `<<token>>`.\n\n"
        "**MUST FOLLOW RULES:**\n"
        "- **Strict Formatting:** Your response MUST be the full, original sentence with the predicted tokens marked. Example: `ioi_9999: After the game, <<Mary>> and [[John]] played chess. Mary gave a book to`\n"
        "- **Exact Count:** You MUST predict and mark exactly {top_k} tokens.\n"
        "- **No Extra Text:** Do not include any reasoning, explanations, or any text other than the marked sentence.\n\n"
        "--- Example ---\n"
        "**Hypothesis:** 'This head attends to the Subject and the Indirect Object, with a strong preference for the Subject.'\n"
        "**Sentence:** `ioi_9999: After the game, Mary and John played chess. Mary gave a book to`\n"
        "**Top_K:** 2\n"
        "**Correct Output:** `ioi_9999: After the game, <<Mary>> and [[John]] played chess. Mary gave a book to`\n"
        "--- End of Example ---\n\n"
        "**Now, it's your turn. Perform the real task below.**\n\n"
        f"**Hypothesis to Test for Head {sender_head}:**\n"
        f"\"{hypothesis}\"\n"
        "**Sentence to Analyze:**\n"
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

def evaluate_top_k_predictions(predictions, attention_ground_truth, top_k):
    """
    评估直接注意力预测，并生成新的、更直观的反馈格式。
    """
    total_ndcg = 0
    feedback_details = []
    count = 0

    for sid, pred_data in predictions.items():
        if sid not in attention_ground_truth: continue
        
        gt_data = attention_ground_truth[sid]
        
        # --- 1. 计算真实的Top-K token (逻辑不变) ---
        if not gt_data.get('tokens') or not gt_data.get('scores'): continue
        all_tokens_with_scores = sorted(zip(gt_data['tokens'], gt_data['scores']), key=lambda x: x[1], reverse=True)
        real_top_k_tokens = [token for token, score in all_tokens_with_scores[:top_k]]
        
        predicted_tokens = pred_data.get('predicted_tokens', [])
        if not real_top_k_tokens: continue

        # --- 2. 计算NDCG分数 (逻辑不变) ---
        relevance_map = {token: (top_k - i) for i, token in enumerate(real_top_k_tokens)}
        true_relevance = np.asarray([[relevance_map.get(t, 0) for t in real_top_k_tokens]])
        pred_relevance = np.asarray([[relevance_map.get(t, 0) for t in predicted_tokens]])
        max_len = max(true_relevance.shape[1], pred_relevance.shape[1])
        if max_len > 0:
            true_relevance = np.pad(true_relevance, ((0, 0), (0, max_len - true_relevance.shape[1])))
            pred_relevance = np.pad(pred_relevance, ((0, 0), (0, max_len - pred_relevance.shape[1])))
            ndcg = ndcg_score(true_relevance, pred_relevance, k=top_k)
        else:
            ndcg = 0.0
        total_ndcg += ndcg
        count += 1
        
        # --- 3. 【重大改变】生成新的、更直观的反馈字符串 ---
        original_sentence_text = gt_data['sentence_text']
        words, suffixed_map = _get_suffixed_word_map(original_sentence_text)
        
        pred_words, real_words = [], []
        
        # 为了标记，将列表转为集合以便快速查找
        pred_set = set(predicted_tokens)
        real_set = set(real_top_k_tokens)
        
        for i, word in enumerate(words):
            suffixed_word = suffixed_map[i]
            
            # 构建预测字符串
            if suffixed_word in pred_set:
                symbol = "✓" if suffixed_word in real_set else "✗"
                pred_words.append(f"<<{word}>>({symbol})")
            else:
                pred_words.append(word)
            
            # 构建真实答案字符串
            real_words.append(f"<<{word}>>" if suffixed_word in real_set else word)

        feedback_details.append(
            f"--- Sentence: {sid} ---\n"
            f"  [Your Prediction]: {' '.join(pred_words)}\n"
            f"  [Real Answer]:     {' '.join(real_words)}"
        )

    avg_ndcg = total_ndcg / count if count > 0 else 0
    final_feedback = f"Overall Attention NDCG@{top_k} for this batch: {avg_ndcg:.2f}\n\n" + "\n".join(feedback_details)
    return avg_ndcg, "\n".join(feedback_details)

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

async def predict_for_single_sentence(open_router, hypothesis, sid, sentence_data, sender_head, top_k, output_dir):
    """
    为单个句子生成预测，包含完整的Prompt和推理过程。
    """
    sentence_text = sentence_data['sentence_text']

    system_prompt = (
        "You are a meticulous AI researcher applying a hypothesis to predict experimental outcomes in an IOI task. Your task is to predict how corrupting a 'Sender Head' will change the attention of a 'Receiver Head'.\n\n"
        "**--- Key Concepts in the IOI Task (Crucial Background) ---**\n"
        "To understand the hypothesis, you must know these linguistic roles in the context of a sentence. However, in an IOI task, we may not provide the whole sentence, which means that IO at the end of the sentence may be hidden for prediction. For example, \"John and Mary went to the park. John gave a backpack to Mary.\"\n"
        "In this sentence:"
        "- **Subject (S):** The one performing the action. (e.g., 'John' gave the backpack).\n"
        "- **Indirect Object (IO):** The recipient of the action. Although the token is not provided directly at the end of the sentence, but you can infer from the context.(e.g., 'Mary' received the backpack).\n"
        "- **Direct Object (DO):** The object being transferred. (e.g., 'a backpack').\n"
        "The IOI task is about correctly identifying the **Indirect Object (IO)**.\n\n"
        "--- **Core Task & Causal Rules** ---\n"
        "Your task is to predict how **corrupting Sender Head {sender_head}** will change the attention of downstream receiver heads according to the hypothesis.\n\n"
        "This means you must predict:\n"
        "1. When Sender Head {sender_head} is corrupted, which token(s) the receiver heads will pay **MORE** attention to. This happens because the Sender Head's normal **SUPPRESSION** of that token is removed. You will mark these with `<<token>>`.\n"
        "2. When Sender Head {sender_head} is corrupted, which token(s) the receiver heads will pay **LESS** attention to. This happens because the Sender Head's normal **PROMOTION** of that token is removed. You will mark these with `[[token]]`.\n\n"
        "**MUST FOLLOW RULES:**\n"
        "- **Logical Consistency:** A token CANNOT be marked for both increase `<< >>` and decrease `[[ ]]`.\n"
        "- **No Opposite Markings:** You MUST mark `<< >>` for increase and `[[ ]]` for decrease.\n"
        f"- **Exact Count:** You must highlight exactly {top_k} token(s) for increase and {top_k} token(s) for decrease.\n"
        "- **No Modification:** Do not omit, add, or change any part of the original sentence text.\n"
        "- **Strict Formatting:** You must first provide a step-by-step analysis in a `[REASONING]` block. Then, on a new line, provide the final marked sentence in a `[PREDICTION]` block.\n\n"
    )
    user_prompt = (
        "--- **Illustrative Example (How to perform the task):** ---\n"
        "To ensure you understand the method, here is a complete, self-contained example. **DO NOT use the functions or entities from this example for your real task.** Your goal is to learn the *reasoning process*.\n\n"
        "**Fictional Scenario:**\n"
        "- We are studying how a fictional **Sender Head 0.0** affects a fictional **Receiver Head 0.1**.\n"
        "- **Known function of Receiver Head 0.1:** It is a 'Verb Detector Head'. It tries to pay high attention to verbs in a sentence.\n"
        "- **Hypothetical Hypothesis for Sender Head 0.0:** 'Sender Head 0.0 helps the Verb Detector by **promoting** attention to action verbs (like 'gave', 'ran') and **suppressing** attention to linking verbs (like 'is', 'was', 'were').'\n\n"
        "**Example Sentence:** `ioi_9999: After the game, he was tired but she gave him a water bottle.`\n\n"
        "**Reasoning Walkthrough (Your thought process):**\n"
        "1.  **Analyze Suppression:** The hypothesis for Sender Head 0.0 says it **suppresses** 'linking verbs'. In the sentence, the linking verb is 'was'.\n"
        "2.  **Predict Effect of Suppression Removal:** If we corrupt Sender Head 0.0, its suppression of 'was' is removed. Therefore, Receiver Head 0.1 (the Verb Detector) will pay **MORE** attention to 'was'.\n"
        "3.  **Apply Marking Rule:** The rule for INCREASED attention is `<<token>>`. So, I will mark `<<was>>`.\n\n"
        "4.  **Analyze Promotion:** The hypothesis for Sender Head 0.0 says it **promotes** 'action verbs'. In the sentence, the action verb is 'gave'.\n"
        "5.  **Predict Effect of Promotion Removal:** If we corrupt Sender Head 0.0, its promotion of 'gave' is removed. Therefore, Receiver Head 0.1 (the Verb Detector) will pay **LESS** attention to 'gave'.\n"
        "6.  **Apply Marking Rule:** The rule for DECREASED attention is `[[token]]`. So, I will mark `[[gave]]`.\n\n"
        "**Correct Output for this Fictional Example:**\n"
        "`ioi_9999: After the game, he <<was>> tired but she [[gave]] him a water bottle.`\n"
        "--- End of Example ---\n\n"
        "**Your Turn:"
        f"**Hypothesis:** {hypothesis}\n"
        "Now, using the **actual hypothesis for Head {sender_head}**, apply the same reasoning process to the following sentence.\n\n"
        f"**Sentence to Analyze:**\n`{sid}: {sentence_text}`"
    )
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_prompt}
    ]
    response = await open_router.generate(messages=messages, output_dir=output_dir)
    
    response_text = response.text.strip()
    
    reasoning_match = re.search(r"\[REASONING\]\s*(.*?)\s*\[PREDICTION\]", response_text, re.DOTALL | re.IGNORECASE)
    prediction_match = re.search(r"\[PREDICTION\]\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning found."
    marked_sentence = prediction_match.group(1).strip() if prediction_match else ""
    
    increase, decrease = _parse_and_suffix_tokens(sentence_text, marked_sentence)
    
    return sid, {
        "increase": increase,
        "decrease": decrease,
        "reasoning": reasoning,
        "raw_response": response_text
    }

async def predict_attention_changes_for_sentences(open_router, hypothesis, sentences_data, sender_head, top_k, output_dir):
    """
    让LLM根据假设，为一批句子并行预测注意力变化。
    """
    print("正在根据假设并行预测每个句子的注意力变化...")
    
    tasks = []
    for sid, data in sentences_data.items():
        task = predict_for_single_sentence(
            open_router, hypothesis, sid, data, sender_head, top_k, output_dir
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    predictions = {sid: result_dict for sid, result_dict in results}
        
    return predictions, None, None # 不再返回统一的prompt和raw_response

def evaluate_predictions(predictions, ground_truth_data, sender_head, top_k=1):
    """
    【重大重构】根据用户反馈，彻底重构评估函数。
    1. 修复了重复token被多次标记为正确答案的bug。
    2. 将Increase/Decrease反馈合并到同一行，减少上下文并增强对比。
    """
    total_f1_increase, total_f1_decrease = 0, 0
    feedback_details = []
    ground_truth_details = {}
    count = 0

    for sid, pred in predictions.items():
        if sid not in ground_truth_data: continue
        
        gt_sentence_data = ground_truth_data[sid]
        pred_inc_set = set(pred.get('increase', []))
        pred_dec_set = set(pred.get('decrease', []))
        
        # --- 1. 【已修正】精确定位真实答案的唯一 suffixed token ---
        real_diffs = gt_sentence_data['diff_vectors']
        tokens = gt_sentence_data['tokens']
        sentence_vectors = [np.array(v) for k, v in real_diffs.items() if k.startswith(f"{sender_head[0]}.{sender_head[1]}->") and v]
        if not sentence_vectors: continue
        
        min_len = min(len(v) for v in sentence_vectors)
        avg_diff_vector = np.mean([v[:min_len] for v in sentence_vectors], axis=0)
        
        original_sentence_text = gt_sentence_data['sentence_text']
        words, suffixed_map = _get_suffixed_word_map(original_sentence_text)
        
        # 将分数与唯一的、带后缀的token绑定
        suffixed_token_changes = []
        for i in range(min_len):
            suffixed_token = suffixed_map.get(i)
            if suffixed_token:
                suffixed_token_changes.append((suffixed_token, avg_diff_vector[i]))

        suffixed_token_changes.sort(key=lambda x: x[1], reverse=True)
        
        # 现在我们得到的是精确的、唯一的token集合
        real_inc_set = {t for t, s in suffixed_token_changes[:top_k]}
        real_dec_set = {t for t, s in suffixed_token_changes[-top_k:]}
        
        ground_truth_details[sid] = {"increase": list(real_inc_set), "decrease": list(real_dec_set)}
        
        # --- 2. 计算F1分数 (逻辑不变) ---
        f1_increase = (2 * len(pred_inc_set.intersection(real_inc_set))) / (len(pred_inc_set) + len(real_inc_set)) if (len(pred_inc_set) + len(real_inc_set)) > 0 else 0
        f1_decrease = (2 * len(pred_dec_set.intersection(real_dec_set))) / (len(pred_dec_set) + len(real_dec_set)) if (len(pred_dec_set) + len(real_dec_set)) > 0 else 0
        total_f1_increase += f1_increase
        total_f1_decrease += f1_decrease
        count += 1

        # --- 3. 【已修正】生成新的、合并的反馈字符串 ---
        pred_words, real_words = [], []
        for i, word in enumerate(words):
            suffixed_word = suffixed_map.get(i)
            if not suffixed_word:
                pred_words.append(word)
                real_words.append(word)
                continue

            # 构建合并的预测字符串
            if suffixed_word in pred_inc_set:
                symbol = "✓" if suffixed_word in real_inc_set else "✗"
                pred_words.append(f"<<{word}>>({symbol})")
            elif suffixed_word in pred_dec_set:
                symbol = "✓" if suffixed_word in real_dec_set else "✗"
                pred_words.append(f"[[{word}]]({symbol})")
            else:
                pred_words.append(word)

            # 构建合并的真实答案字符串
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

    avg_f1 = ((total_f1_increase / count) + (total_f1_decrease / count)) / 2 if count > 0 else 0
    final_feedback = f"Overall Causal F1 Score for this batch: {avg_f1:.2f}\n\n" + "\n".join(feedback_details)
    return avg_f1, final_feedback, ground_truth_details

async def refine_hypothesis_combined(open_router, old_hypothesis, sender_head, explanation, output_dir, f1_score, ndcg_score, attention_f1_score, causal_feedback=None, attention_feedback=None):
    """
    【已修正】恢复了旧版(auto_s_att_old.py)中完整且正确的Prompt逻辑，并正确集成了新的attention_f1_score。
    """
    print("根据两种并行反馈，综合精炼假设...")

    # --- 1. 计算综合分数 ---
    if ndcg_score is None or ndcg_score < 0:
        ndcg_score = 0.0
    if attention_f1_score is None or attention_f1_score < 0:
        attention_f1_score = 0.0
    
    # 使用几何平均数，如果注意力任务没跑，就只用因果F1
    composite_score = np.sqrt(f1_score * attention_f1_score) if attention_feedback else f1_score
    print(f"综合性能分数 (几何平均): {composite_score:.2f}")

    # --- 2. 【已恢复】根据分数动态生成任务指令 ---
    task_guideline = ""
    if composite_score > 0.75: # 阈值可以调整
        task_guideline = (
            "**Your Task: Minor Refinement (High-Confidence State)**\n"
            "The current hypothesis is performing well, with a high composite score. **DO NOT make drastic changes.** Your goal is to make **minor, compatible adjustments** that might fix the remaining small errors without breaking what already works.\n"
            "- **Focus on Clarification:** Can you make the language more precise?\n"
            "- **Focus on Exception Handling:** Can you add a clause that explains the few failed cases?\n"
            "**Preserve the core mechanism** of the old hypothesis. Your new hypothesis should be a slightly improved version, not a complete rewrite."
        )
    else:
        task_guideline = (
            "**Your Task: Major Refinement (Low-Confidence State)**\n"
            "The current hypothesis has evident flaws. Your goal is to propose a **single, unified hypothesis** that provides a better, more balanced explanation for BOTH phenomena.\n"
            "**Key Principle: Avoid Over-correction.** Do not become fixated on fixing one type of error (e.g., attention prediction) to the extent that you abandon a correct understanding of the other type (e.g., causal effect). Your new hypothesis must be a robust improvement, not just a shift in focus."
        )

    # --- 3. 【已恢复】构建完整的、包含详细引导的Prompt ---
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
        "--- PERFORMANCE & FEEDBACK ANALYSIS ---\n"
        f"**Performance Summary:**\n"
        f"- Causal F1 Score: {f1_score:.2f}\n"
        f"- Attention NDCG Score: {ndcg_score:.2f}\n"
        f"- Attention F1 Score: {attention_f1_score:.2f}\n" # <-- 新增的分数
        f"- **Composite Score (Geometric Mean): {composite_score:.2f}**\n\n"
        "Below is the detailed feedback. Analyze all evidence carefully to formulate a better hypothesis."
    ]
    
    if attention_feedback:
        user_prompt_parts.append(
            "\n**Feedback Source 1: Direct Attention Prediction (What the head LOOKS AT)**\n"
            "This feedback shows how well the hypothesis predicted the head's own Top-K attention targets. A low score means the hypothesis is wrong about what the head is paying attention to.\n"
            f"{attention_feedback}\n"
        )

    if causal_feedback:
        user_prompt_parts.append(
            "\n**Feedback Source 2: Causal Effect Prediction (What the head DOES)**\n"
            "This feedback reveals the head's true downstream function (suppression/promotion). A low score means the hypothesis is wrong about the head's actual effect.\n"
            f"{causal_feedback}\n"
        )
    
    user_prompt_parts.append(f"\n{task_guideline}\n") # <-- 恢复了动态指令
    
    # 【已恢复】对REASONING和HYPOTHESIS的详细要求
    user_prompt_parts.append(
        "**Response Format (Strict):**\n"
        "1.  **[REASONING]:** Start with this tag. First, explicitly analyze the conflict and performance trade-offs: 'The direct attention feedback suggests the head looks at X, while the causal feedback proves its effect is Y. The previous hypothesis performed well on causal effect (F1 score) but poorly on attention (NDCG). It failed because it only accounted for Y.' Then, propose a mechanism that reconciles this conflict in a balanced way. For example: 'A better explanation that preserves the correct causal understanding is that the head attends to X *in order to* gather information to perform action Y.'\n"
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
    reasoning_match = re.search(r"\[REASONING\]:\s*(.*?)\s*\[HYPOTHESIS\]:", response_text, re.DOTALL | re.IGNORECASE)
    hypothesis_match = re.search(r"\[HYPOTHESIS\]:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning found in response."
    new_hypothesis = hypothesis_match.group(1).strip() if hypothesis_match else response_text.strip()
    
    prompt_log = f"---SYSTEM PROMPT---\n{system_prompt}\n\n---USER PROMPT---\n{user_prompt}"
    raw_response_log = response.text
    
    print("精炼后的假设:", new_hypothesis)
    return new_hypothesis, reasoning, prompt_log, raw_response_log

def _parse_and_suffix_tokens(original_sentence_text: str, marked_sentence: str, increase_markers: tuple = ("<<", ">>"), decrease_markers: tuple = ("[[", "]]")) -> tuple[list[str], list[str]]:
    """
    从LLM标记的句子中解析token，并根据其在原始句子中的位置智能地添加后缀。
    这是一个更健壮的实现，借鉴了auto_logit_two.py的思想。
    """
    # 1. 对原始句子进行分词并建立全局计数，以确定哪些词是重复的
    # 使用正则表达式来更好地处理各种token
    original_tokens = re.findall(r"\w+|[^\w\s]", original_sentence_text)
    global_counts = defaultdict(int)
    for token in original_tokens:
        global_counts[normalize_token(token)] += 1

    # 2. 从标记的句子中提取出被标记的token（不带后缀）
    increase_targets = {normalize_token(t) for t in re.findall(f"{re.escape(increase_markers[0])}(.*?){re.escape(increase_markers[1])}", marked_sentence)}
    decrease_targets = {normalize_token(t) for t in re.findall(f"{re.escape(decrease_markers[0])}(.*?){re.escape(decrease_markers[1])}", marked_sentence)}

    # 3. 遍历原始token列表，重建带后缀的标记token
    increase_suffixed, decrease_suffixed = [], []
    running_counts = defaultdict(int)
    for token in original_tokens:
        norm_token = normalize_token(token)
        running_counts[norm_token] += 1
        
        # 检查这个token是否是我们的目标之一
        is_increase = norm_token in increase_targets
        is_decrease = norm_token in decrease_targets

        if is_increase or is_decrease:
            # 如果全局计数大于1，说明是重复词，需要加后缀
            if global_counts[norm_token] > 1:
                suffixed_token = f"{token.strip()}_{running_counts[norm_token]}"
            else:
                suffixed_token = token.strip()
            
            if is_increase:
                increase_suffixed.append(suffixed_token)
            if is_decrease:
                decrease_suffixed.append(suffixed_token)

    return increase_suffixed, decrease_suffixed


async def main():
    # 1. 初始化和加载数据
    args = await parse_arguments()
    sender_head = (args.layer, args.head)
    output_dir = args.output_dir
    # 为两个阶段分别设置迭代次数
    MAX_ITERATIONS = 10 
    
    top_k_attention = 2
    top_k_causal = 1

    # --- 加载双份Ground Truth数据 ---
    # 数据源1：因果效应数据
    full_causal_dataset_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ioi", "path_patching", "causal_dataset.json")
    try:
        with open(full_causal_dataset_path, "r") as f:
            full_causal_dataset = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法加载因果数据集 {full_causal_dataset_path}: {e}")
        return

    # 数据源2：直接注意力分数数据
    attention_scores_dataset_path = os.path.join(output_dir, "attention_scores_ground_truth.jsonl")
    attention_ground_truth = {}
    try:
        # 【已修正】修改加载方式，从逐行加载JSONL改为一次性加载标准JSON
        with open(attention_scores_dataset_path, "r") as f:
            # 直接加载整个文件，因为它是一个JSON列表
            all_data = json.load(f) 
            
            # 遍历加载后的列表来构建字典
            for data in all_data:
                if "key" in data and "attention_scores" in data:
                    attention_ground_truth[data["key"]] = {
                        "sentence_text": data.get("original_sentence", ""),
                        "tokens": [item.get("token", "") for item in data["attention_scores"]],
                        "scores": [item.get("score", 0.0) for item in data["attention_scores"]]
                    }
                else:
                    print(f"警告: 在 {attention_scores_dataset_path} 中发现缺少 'key' 或 'attention_scores' 的条目，已跳过。")

        if not attention_ground_truth:
             print(f"警告: 从 {attention_scores_dataset_path} 加载了 0 条有效的注意力数据。")
             attention_ground_truth = None
        else:
             print(f"成功加载 {len(attention_ground_truth)} 条直接注意力基准数据。")

    except FileNotFoundError:
        print(f"警告: 未找到直接注意力基准数据文件 {attention_scores_dataset_path}。\n第一阶段的精炼将被跳过。")
        attention_ground_truth = None
    except json.JSONDecodeError as e:
        print(f"错误: 解析JSON文件 {attention_scores_dataset_path} 时失败: {e}\n第一阶段的精炼将被跳过。")
        attention_ground_truth = None
    except Exception as e:
        print(f"加载 {attention_scores_dataset_path} 时发生未知错误: {e}\n第一阶段的精炼将被跳过。")
        attention_ground_truth = None

    train_dataset, validation_dataset = split_dataset(full_causal_dataset)
    open_router = initialize_openrouter()
    full_run_log = []
    
    print("生成初始假设...")
    explanation = f"""
        Background Context: The sender head {sender_head} is believed to function by influencing a circuit of downstream heads. The primary role of these downstream heads is to attend to the indirect object (IO) token in a sentence. Here is a summary of what is known about the key heads (9.6 and 9.9) that we are measuring the effect on:
        - **Head 9.6**: ...
        - **Head 9.9**: ...
        Your task is to hypothesize the function of the sender head {sender_head} mainly based on how it causally affects the attention of these downstream heads.
        """
    
    # 【重大改变】采用新的、基于样本的假设生成流程
    # 1. 随机抽样5个句子
    sampled_sentences = sample_sentences_from_causal_dataset(train_dataset, batch_size=5)
    
    # 2. 为这5个句子计算真实的因果效应
    causal_examples = get_causal_effects_for_sampling(sampled_sentences, sender_head, top_k=1)
    
    if not causal_examples:
        print("无法为抽样句子计算因果效应，程序终止。")
        return

    # 3. 让LLM根据这些具体例子进行归纳
    current_hypothesis, initial_hypothesis_prompt = await generate_initial_hypothesis(open_router, sender_head, causal_examples, explanation, output_dir)
    
    if not current_hypothesis:
        print("无法生成初始假设，程序终止。")
        return
    
    attention_feedback = None
    # --- 【恢复为】统一的并行迭代精炼循环 ---
    print("\n--- 开始统一的并行迭代精炼循环 ---")
    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {i}/{MAX_ITERATIONS} ---")
        iteration_log = {"iteration": i, "hypothesis_before": current_hypothesis}

        # --- 1. 并行预测：同时进行两种预测 ---
        test_sentences_batch = sample_sentences_from_causal_dataset(train_dataset, batch_size=5)
        
        causal_predictions, _, _ = await predict_attention_changes_for_sentences(
            open_router, current_hypothesis, test_sentences_batch, sender_head, top_k_causal, output_dir
        )
        
        attention_predictions = None
        if attention_ground_truth:
            test_sentences_attention = {sid: data for sid, data in attention_ground_truth.items() if sid in test_sentences_batch}
            tasks = [predict_top_k_attenders_for_sentence(open_router, current_hypothesis, sid, data, sender_head, top_k_attention, output_dir) for sid, data in test_sentences_attention.items()]
            results = await asyncio.gather(*tasks)
            attention_predictions = {sid: result for sid, result in results}

        # --- 2. 并行评估：同时计算两种分数和反馈 ---
        f1_score, causal_feedback, _ = evaluate_predictions(causal_predictions, train_dataset, sender_head)
        print(f"Iteration {i} Causal F1 Score: {f1_score:.2f}")
        iteration_log["causal_f1"] = f1_score
        iteration_log["causal_feedback"] = causal_feedback

        ndcg_score_val = 0
        attention_feedback = None
        if attention_ground_truth and attention_predictions:
            ndcg_score_val, attention_feedback = evaluate_top_k_predictions(attention_predictions, attention_ground_truth, top_k_attention)
            print(f"Iteration {i} Direct Attention NDCG@{top_k_attention}: {ndcg_score_val:.2f}")
            iteration_log["attention_ndcg"] = ndcg_score_val
            iteration_log["attention_feedback"] = attention_feedback

        # --- 3. 检查终止条件 (如果两个分数都足够高) ---
        if f1_score > 0.85 and ndcg_score_val > 0.85:
            print(f"假设收敛成功，F1={f1_score:.2f}, NDCG={ndcg_score_val:.2f}。")
            full_run_log.append(iteration_log)
            break
        
        if i == MAX_ITERATIONS:
            full_run_log.append(iteration_log)
            break

        # --- 4. 统一精炼：使用两种反馈进行修正 ---
        current_hypothesis, reasoning, prompt, raw_response = await refine_hypothesis_combined(
            open_router, current_hypothesis, sender_head, explanation, output_dir, 
            f1_score=f1_score,
            ndcg_score=ndcg_score_val,
            causal_feedback=causal_feedback, 
            attention_feedback=attention_feedback
        )
        iteration_log.update({
            "hypothesis_after": current_hypothesis, 
            "refinement_reasoning": reasoning, 
            "refinement_prompt": prompt, 
            "refinement_raw_response": raw_response
        })
        full_run_log.append(iteration_log)

    final_hypothesis = current_hypothesis
    print(f"\n--- 迭代精炼完成。最终假设: {final_hypothesis} ---")

    # --- 最终验证与日志记录 ---
    print("\n--- Final Validation and Logging ---")
    val_causal_preds, _, _ = await predict_attention_changes_for_sentences(open_router, final_hypothesis, validation_dataset, sender_head, top_k_causal, output_dir)
    val_f1, _, val_causal_gt = evaluate_predictions(val_causal_preds, validation_dataset, sender_head, top_k=top_k_causal)
    print(f"最终验证集 F1 分数: {val_f1:.2f}")

    val_ndcg = 0; val_att_preds = None; val_att_gt = None
    if attention_ground_truth:
        val_sids_att = {sid for sid in validation_dataset if sid in attention_ground_truth}
        tasks = [predict_top_k_attenders_for_sentence(open_router, final_hypothesis, sid, attention_ground_truth[sid], sender_head, top_k_attention, output_dir) for sid in val_sids_att]
        results = await asyncio.gather(*tasks)
        val_att_preds = {sid: result for sid, result in results}
        val_ndcg, _ = evaluate_top_k_predictions(val_att_preds, attention_ground_truth, top_k_attention)
        val_att_gt = {sid: attention_ground_truth[sid] for sid in val_sids_att}
        print(f"最终验证集 NDCG@{top_k_attention} 分数: {val_ndcg:.2f}")

    final_log_entry = {
        "head": f"{sender_head[0]}.{sender_head[1]}",
        "typename": args.typename,
        "initial_hypothesis_prompt": initial_hypothesis_prompt,
        "full_run_log": full_run_log,
        "final_hypothesis": final_hypothesis,
        "validation_scores": {"causal_f1": val_f1, "direct_attention_ndcg": val_ndcg},
        "validation_details": {
            "causal_predictions": val_causal_preds, "causal_ground_truth": val_causal_gt,
            "attention_predictions": val_att_preds, "attention_ground_truth": val_att_gt
        }
    }
    
    final_log_path = os.path.join(output_dir, f"final_result_round_{args.rounds}.json")
    with open(final_log_path, "w") as f:
        json.dump(final_log_entry, f, indent=2, ensure_ascii=False)
    
    print(f"\n所有运行数据和最终结果已统一保存至: {final_log_path}")

if __name__ == "__main__":
    asyncio.run(main())