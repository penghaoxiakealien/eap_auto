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
sys.path.append("/home/wangziran/eap_auto")
from tests.experiments.summarize_receiver_heads import summarize_head_group

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
    parser.add_argument("--rounds", type=int, default=1, help="(Deprecated) Kept for compatibility; no longer controls iterations.")
    parser.add_argument("--typename", type=str, default="", help="The typename of the head (e.g., s_inhibition_head).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument(
        "--causal_dataset",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "garden", "path_patching", "causal_dataset.json"),
        help="Path to the causal dataset JSON containing diff_vectors (default: global causal_dataset.json).",
    )
    parser.add_argument(
        "--data-source-dir",
        type=str,
        default="",
        help="Directory containing causal_dataset.json and attention_scores_ground_truth.jsonl (optional).",
    )
    parser.add_argument(
        "--receiver_heads",
        type=str,
        default="",
        help="Comma-separated downstream heads (e.g., '10.7,11.10') for contextual prompts.",
    )
    parser.add_argument(
        "--receiver_descriptions_file",
        type=str,
        default="",
        help="Optional JSON file mapping receiver head strings to textual descriptions.",
    )
    parser.add_argument(
        "--with-reasoning",
        action="store_true",
        help="Include [REASONING] blocks in LLM outputs (default: only [PREDICTION]).",
    )
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
        "--test-sample-size",
        type=int,
        default=0,
        help="Test sentences for final evaluation (0 means full test split).",
    )
    parser.add_argument(
        "--test-all-validations",
        action="store_true",
        help="Evaluate every validation checkpoint hypothesis on the test split.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2-2025-12-11",
        help="OpenRouter model name.",
    )
    return parser.parse_args()

def format_garden_sid(sid) -> str:
    """Format sentence id for display/parsing."""
    sid_str = str(sid)
    if sid_str.startswith("garden_"):
        return sid_str
    if sid_str.isdigit():
        return f"garden_{int(sid_str):04d}"
    return f"garden_{sid_str}"

def split_dataset(full_dataset, train_split=0.4, validation_split=0.2, seed=42):
    """将完整数据集确定性地划分为训练/验证/测试集"""
    test_split = 1.0 - train_split - validation_split
    if test_split < 0:
        raise ValueError("Split ratios must sum to <= 1.0")
    print(
        f"将数据集划分为 {train_split:.0%} 训练集、{validation_split:.0%} 验证集、{test_split:.0%} 测试集..."
    )
    sentence_ids = list(full_dataset.keys())
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

def initialize_openrouter(model: str = "gpt-5.2-2025-12-11"):
    """初始化OpenRouter API（默认使用 Claude Sonnet）。"""
    # api_key = "sk-Z3pwy4dD8WY2XZlbzch66NP5hQIoFKeU7KvI2XD8bQSyFVGO"
    # api_key = "sk-99F0IFe53pSHOPQ3phWbEAEx86ZDOqkE58Ov9aYCS9AOQ2C7"
    api_key = "sk-ssCRXZzlj8qhPNs6Ps2BxTXZQXq97vJvKATpFXdwYV0E0gUO"
    return OpenRouter(model=model, api_key=api_key)

def extract_hypothesis_text(response_text):
    """从LLM的响应中提取[HYPOTHESIS]部分"""
    tag_re = re.compile(r"(?:\*\*\s*)?\[HYPOTHESIS\]\s*:?(?:\s*\*\*)?", re.IGNORECASE)
    tags_re = re.compile(r"(?:\*\*\s*)?\[[A-Z_]+\]\s*:?(?:\s*\*\*)?", re.IGNORECASE)
    match = tag_re.search(response_text)
    if not match:
        alt_re = re.compile(r"(?:\*\*\s*)?HYPOTHESIS\s*:?(?:\s*\*\*)?", re.IGNORECASE)
        match = alt_re.search(response_text)
        if not match:
            print("警告: 在LLM响应中未找到 '[HYPOTHESIS]' 标签。")
            # Fallback: strip any leading reasoning block if present.
            cleaned = re.sub(
                r"(?:\*\*\s*)?\[REASONING\]\s*:?.*?$",
                "",
                response_text,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
            return cleaned
    tail = response_text[match.end() :].strip()
    next_tag = tags_re.search(tail)
    if next_tag:
        tail = tail[: next_tag.start()].strip()
    # Remove any stray reasoning tag lines inside the hypothesis block.
    tail = re.sub(r"^\s*\[REASONING\]\s*:?.*?$", "", tail, flags=re.IGNORECASE | re.MULTILINE).strip()
    return tail

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


async def generate_initial_hypothesis(open_router, sender_head, causal_examples, explanation, output_dir, hop_info):
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
        "You are a meticulous AI researcher conducting an important investigation into patterns found in language. "
        "The text is based on the Garden Path NP/Z v-trans task, where the model is asked to choose the correct disambiguation token (\"was\" or \"for\") at the end of a sentence.\n\n"
        "**Key Garden positions you must use:**\n"
        "- **SUBORD:** the clause introducer (e.g., As/While/After/When)\n"
        "- **SUBJ:** the main clause subject\n"
        "- **VERB:** the ambiguous verb whose transitivity drives the NP/Z ambiguity\n"
        "- **OBJ_HEAD:** the head noun of the ambiguous object phrase\n"
        "- **REL_PRON:** the relative pronoun (e.g., who)\n"
        "- **REL_VERB:** the verb in the relative clause\n"
        "- **END/DISAMB:** the final disambiguation position (\"was\"/\"for\")\n"
        "Also consider clause boundaries and modifiers (e.g., relative clauses) when identifying roles.\n\n"
        "Example (for grounding only):\n"
        "\"As the criminal shot the woman who told bad jokes [was/for]\"\n"
        "- SUBJ: criminal\n"
        "- VERB: shot\n"
        "- OBJ_HEAD: woman\n"
        "- REL_PRON: who\n"
        "- REL_VERB: told\n"
        "- END/DISAMB: [was/for]\n\n"
        "Important: The head may help OR hurt the task; do not assume it is beneficial by default."
    )
    user_prompt = (
        f"**Your task is to provide a rational hypothesis that thoroughly explains the function of the given attention head {sender_head} in this specific task based on the concrete experimental evidence below.**\n\n"
        "The samples may not explicitly annotate sentence roles. You should infer SUBJ / VERB / OBJ_HEAD / REL_PRON / REL_VERB from context and sentence structure.\n\n"
        f"**Downstream Head Context (may include summaries for intermediate/target groups):**\n{explanation}\n\n"
        "**Hop Type:**\n"
        "- If intermediate heads are provided, treat this as a TWO-HOP setting (A→B→C).\n"
        "- Otherwise, treat this as a SINGLE-HOP setting (A→B).\n\n"
        "**Crucial Interpretability Context (Path Patching — read carefully):**\n"
        "The data below comes from path patching experiments that isolate the causal effect of a specific path.\n\n"
        "What path patching means here:\n"
        "- We run the model on a CLEAN input and on a CORRUPTED input.\n"
        "- We cache activations from both runs.\n"
        "- We then re-run the clean input, but along a specific path we REPLACE the clean activations with the corresponding CORRUPTED activations.\n"
        "- This replacement isolates the causal effect of that path.\n\n"
        "What we measure:\n"
        "- We measure how DOWNSTREAM receiver/target heads' attention patterns change after this replacement.\n"
        "- The reported INCREASE/DECREASE refers to changes in the downstream heads' attention, not the sender head's own attention.\n\n"
        "Direction interpretation (use this exactly):\n"
        "- An **INCREASE** in downstream attention to a token implies the sender head normally **SUPPRESSES** the downstream head's attention to that token (its suppression was removed by corruption/patching).\n"
        "- A **DECREASE** in downstream attention to a token implies the sender head normally **PROMOTES** the downstream head's attention to that token (its promotion was removed by corruption/patching).\n\n"
        "Single-hop vs two-hop interpretation:\n"
        "- SINGLE-HOP (A→B): we patch the A→B path and measure the change at B.\n"
        "- TWO-HOP (A→B→C): we patch the A→B path but measure the change at C, i.e., A affects C THROUGH B.\n\n"
        "**Sample Format (important):**\n"
        "Each sample is a causal observation in this exact form:\n"
        "- Sentence: \"...\"\n"
        "- Result: Corrupting/patching the sender path causes downstream attention to INCREASE on [...] and DECREASE on [...].\n"
        "Treat the INCREASE/DECREASE token sets as the primary evidence.\n\n"
        "--- **Experimental Evidence (5 Random Samples)** ---\n"
        "Here are the results from 5 randomly selected sentences. Your task is to find a single, unified rule that explains all these examples.\n\n"
        f"{examples_str}"
        "**Your Task & Guidelines:**\n"
        "- Synthesize, don't just list examples.\n"
        "- Use SUBJ / VERB / OBJ_HEAD / REL_PRON / REL_VERB and clause structure explicitly.\n"
        "- Connect what the head likely attends to with its downstream causal impact on Garden disambiguation behavior/logit differences.\n"
        "- The head may help OR hurt the task; do not assume it is beneficial.\n"
        "- Keep it compact: the final hypothesis should be 3–5 sentences in one paragraph.\n\n"
        "**Response Format (Strict):**\n"
        "1. **[REASONING]:** Briefly explain how you identified the structural pattern. The colon is mandatory (must be `[REASONING]:`).\n"
        "2. **[HYPOTHESIS]:** Start with this exact tag. The colon is mandatory (must be `[HYPOTHESIS]:`).\n"
        "   The hypothesis must include both the attention pattern and the causal effect on the task."
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

async def predict_top_k_attenders_for_sentence(open_router, hypothesis, sid, sentence_data, sender_head, top_k, output_dir, attention_position, with_reasoning):
    sentence_text = sentence_data['sentence_text']
    sid_str = str(sid)
    if sid_str.isdigit():
        prefixed_sid = f"garden_{int(sid_str):04d}"
    else:
        prefixed_sid = f"garden_{sid_str}"

    prompt = (
        "--- **Illustrative Example (How to perform the task):** ---\n"
        "To ensure you understand the method, here is a complete, self-contained example. **DO NOT use the functions or entities from this example for your real task.**\n\n"
        "**Fictional Scenario:**\n"
        "- We are studying a Head that attends to 'colors'.\n"
        "- **Hypothesis:** 'Head 9.9 attends to color adjectives.'\n\n"
        "**Example Sentence:** `example_1: The <<red>> apple is next to the <<green>> pear.`\n\n"
        "**Reasoning:**\n"
        "1. The hypothesis says the head attends to color adjectives.\n"
        "2. In this sentence, 'red' and 'green' are color adjectives.\n"
        "3. I will highlight them with `<< >>`.\n\n"
        "--- End of Example ---\n\n"
        "**Your Turn:**\n"
        f"**Hypothesis:** {hypothesis}\n"
        "Now, using the **actual hypothesis for Head {sender_head}**, predict the tokens this head attends to.\n\n"
        f"**Sentence to Analyze:**\n`{prefixed_sid}: {sentence_text}`"
    )

    system_prompt = (
        "You are a meticulous AI researcher applying a hypothesis to predict the attention pattern of a specific head.\n"
        "Your task is to identify which tokens the head pays the MOST attention to, based on the provided hypothesis.\n\n"
        "**--- Key Concepts ---**\n"
        "To understand the hypothesis, you must know these linguistic roles in the context of a sentence. However, in this task, we may not provide the whole sentence, which means that the disambiguation token at the end of the sentence may be hidden for prediction. For example, \"As the criminal shot the woman who told bad jokes [was/for].\"\n"
        "In this sentence:\n"
        "- **Subordinator (SUBORD):** the clause introducer (e.g., 'As', 'While', 'After', 'When').\n"
        "- **Subject (SUBJ):** the agent of the main clause (e.g., 'criminal').\n"
        "- **Ambiguous Verb (VERB):** the main verb whose transitivity drives the NP/Z ambiguity (e.g., 'shot' vs an intransitive verb).\n"
        "- **Object NP Head (OBJ_HEAD):** the head noun of the ambiguous object phrase (e.g., 'woman').\n"
        "- **Relative Pronoun / Verb (REL_PRON / REL_VERB):** the relative clause that follows the object (e.g., 'who told').\n"
        "The Garden task is about correctly predicting the **disambiguation token** at the sentence end (\"was\" or \"for\").\n\n"
        f"- **Query Position:** {attention_position}\n"
        "  (You are predicting where the head attends TO, from this query position)\n\n"
        "**Task:**\n"
        f"Highlight exactly {top_k} tokens that the head attends to heavily, using `<<token>>`.\n"
        "Do NOT use `[[ ]]`. Only use `<< >>` for high attention.\n\n"
        "**MUST FOLLOW RULES:**\n"
        f"- **Exact Count:** You must highlight exactly {top_k} token(s).\n"
        "- **No Modification:** Do not change the sentence text.\n"
        "- **Format:** `[PREDICTION]` block with the marked sentence.\n"
    )
    
    if with_reasoning:
        system_prompt += (
            "You must first provide a step-by-step analysis in a `[REASONING]` block. Then, on a new line, provide the final marked sentence in a `[PREDICTION]` block.\n"
        )
    else:
        system_prompt += (
            "Output only a `[PREDICTION]` block with the marked sentence. No reasoning.\n"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    response = await open_router.generate(messages=messages, output_dir=output_dir)
    
    marked_sentence = response.text.strip()
    prediction_matches = re.findall(r"\[PREDICTION\]\s*(.*)", marked_sentence, re.DOTALL | re.IGNORECASE)
    if prediction_matches:
        marked_sentence = prediction_matches[-1].strip()

    def _extract_target_line(text: str) -> str:
        candidates = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("`") and line.endswith("`"):
                line = line[1:-1].strip()
            if line.startswith(f"{prefixed_sid}:"):
                candidates.append(line)
        marked_candidates = [c for c in candidates if "<<" in c]
        if marked_candidates:
            return marked_candidates[-1]
        if candidates:
            return candidates[-1]
        return ""

    marked_sentence = _extract_target_line(marked_sentence)
    
    # 我们只关心被 << >> 标记的 token，这被 _parse_and_suffix_tokens 解析为 top1 (increase)
    top1, _ = _parse_and_suffix_tokens(sentence_text, marked_sentence)
    
    # 返回列表
    predicted_tokens_ordered = top1
    
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

def _suffix_tokens_in_order(tokens):
    """为注意力真值的 token 列表按出现顺序添加 _1/_2 后缀。"""
    global_counts = defaultdict(int)
    for tok in tokens:
        global_counts[normalize_token(tok)] += 1
    running_counts = defaultdict(int)
    suffixed = []
    for tok in tokens:
        norm_tok = normalize_token(tok)
        running_counts[norm_tok] += 1
        if global_counts[norm_tok] > 1:
            suffixed.append(f"{tok.strip()}_{running_counts[norm_tok]}")
        else:
            suffixed.append(tok.strip())
    return suffixed


def load_receiver_group_meta(output_dir: str) -> dict:
    """Load receiver group metadata if available."""
    meta_path = os.path.join(output_dir, "receiver_group_meta.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {}

def evaluate_top_k_predictions(predictions, attention_ground_truth, top_k):
    """
    【已修改】评估直接注意力预测，同时计算NDCG和F1分数。
    """
    total_ndcg = 0
    # --- 新增F1分数计算所需变量 ---
    total_correct_tokens = 0
    total_predicted_tokens = 0
    total_real_tokens = 0
    
    feedback_details = []
    count = 0

    for sid, pred_data in predictions.items():
        if sid not in attention_ground_truth: continue
        
        gt_data = attention_ground_truth[sid]
        
        tokens = gt_data.get('tokens_suffixed') or gt_data.get('tokens')
        if not tokens or not gt_data.get('scores'):
            continue
        all_tokens_with_scores = sorted(zip(tokens, gt_data['scores']), key=lambda x: x[1], reverse=True)
        real_top_k_tokens = [token for token, score in all_tokens_with_scores[:top_k]]
        
        predicted_tokens = pred_data.get('predicted_tokens', [])
        if not real_top_k_tokens: continue

        # 计算NDCG分数
        relevance_map = {token: (top_k - i) for i, token in enumerate(real_top_k_tokens)}
        true_relevance = np.asarray([[relevance_map.get(t, 0) for t in real_top_k_tokens]])
        pred_relevance = np.asarray([[relevance_map.get(t, 0) for t in predicted_tokens]])
        max_len = max(true_relevance.shape[1], pred_relevance.shape[1])
        if max_len > 0:
            true_relevance_padded = np.pad(true_relevance, ((0, 0), (0, max_len - true_relevance.shape[1])))
            pred_relevance_padded = np.pad(pred_relevance, ((0, 0), (0, max_len - pred_relevance.shape[1])))
            ndcg = ndcg_score(true_relevance_padded, pred_relevance_padded, k=top_k)
        else:
            ndcg = 0.0
        total_ndcg += ndcg
        count += 1
        
        # 为计算F1分数累加计数
        pred_set = set(predicted_tokens)
        real_set = set(real_top_k_tokens)
        
        total_correct_tokens += len(pred_set.intersection(real_set))
        total_predicted_tokens += len(pred_set)
        total_real_tokens += len(real_set)

        # 生成反馈字符串
        original_sentence_text = gt_data['sentence_text']
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
            f"--- Sentence: {format_garden_sid(sid)} ---\n"
            f"  [Your Prediction]: {' '.join(pred_words)}\n"
            f"  [Real Answer]:     {' '.join(real_words)}"
        )

    avg_ndcg = total_ndcg / count if count > 0 else 0
    
    # --- 新增：计算最终的F1分数 ---
    precision = total_correct_tokens / total_predicted_tokens if total_predicted_tokens > 0 else 0
    recall = total_correct_tokens / total_real_tokens if total_real_tokens > 0 else 0
    avg_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    final_feedback = (
        f"Overall Attention NDCG@{top_k} for this batch: {avg_ndcg:.2f}\n"
        f"Overall Attention F1 Score for this batch: {avg_f1:.2f}\n\n" + 
        "\n".join(feedback_details)
    )
    
    # --- 【已修改】返回三个值 ---
    return avg_ndcg, avg_f1, final_feedback

def compute_composite_score(causal_f1, attention_f1, attention_ndcg):
    """统一综合分数口径：优先用 causal_f1 与 attention_f1 的几何平均。"""
    if causal_f1 and attention_f1 and causal_f1 > 0 and attention_f1 > 0:
        return float(np.sqrt(causal_f1 * attention_f1))
    if causal_f1 and causal_f1 > 0:
        return float(causal_f1)
    return 0.0


def chunk_dict(data_dict, chunk_size):
    items = list(data_dict.items())
    return [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]


def sample_sentences_from_causal_dataset(causal_data, batch_size=5, start_idx=0):
    """按固定顺序抽样句子，超出长度后循环。"""
    sentence_ids = list(causal_data.keys())
    if not sentence_ids:
        return {}, start_idx
    sampled_ids = [sentence_ids[(start_idx + i) % len(sentence_ids)] for i in range(batch_size)]
    sampled_sentences = {sid: causal_data[sid] for sid in sampled_ids}
    return sampled_sentences, start_idx + batch_size

async def predict_for_single_sentence(open_router, hypothesis, sid, sentence_data, sender_head, top_k, output_dir, receiver_attention_position, hop_info, with_reasoning):
    """
    为单个句子生成预测，包含完整的Prompt和推理过程。
    """
    sentence_text = sentence_data['sentence_text']
    display_sid = format_garden_sid(sid)

    if with_reasoning:
        system_prompt = (
            "You are a meticulous AI researcher applying a hypothesis to predict experimental outcomes in a Garden Path NP/Z v-trans task. Your task is to predict how corrupting a 'Sender Head' will change the attention of a 'Receiver Head'.\n\n"
            "**--- Key Concepts in the Garden Task (Crucial Background) ---**\n"
            "To understand the hypothesis, you must know these linguistic roles in the context of a sentence. However, in this task, we may not provide the whole sentence, which means that the disambiguation token at the end of the sentence may be hidden for prediction. For example, \"As the criminal shot the woman who told bad jokes [was/for].\"\n"
            "In this sentence:\n"
            "- **Subordinator (SUBORD):** the clause introducer (e.g., 'As', 'While', 'After', 'When').\n"
            "- **Subject (SUBJ):** the agent of the main clause (e.g., 'criminal').\n"
            "- **Ambiguous Verb (VERB):** the main verb whose transitivity drives the NP/Z ambiguity (e.g., 'shot' vs an intransitive verb).\n"
            "- **Object NP Head (OBJ_HEAD):** the head noun of the ambiguous object phrase (e.g., 'woman').\n"
            "- **Relative Pronoun / Verb (REL_PRON / REL_VERB):** the relative clause that follows the object (e.g., 'who told').\n"
            "The Garden task is about correctly predicting the **disambiguation token** at the sentence end (\"was\" or \"for\").\n\n"
            f"- **Receiver Query Position:** {receiver_attention_position} (the token position whose attention row we evaluate).\n"
            "- This means we look at the receiver head's attention distribution from that specific token in the sentence.\n\n"
            "--- **Core Task & Causal Rules** ---\n"
            f"Your task is to predict how **corrupting Sender Head {sender_head}** will change the attention of downstream receiver heads according to the hypothesis.\n\n"
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
    else:
        system_prompt = (
            "You are a meticulous AI researcher applying a hypothesis to predict experimental outcomes in a Garden Path NP/Z v-trans task. Your task is to predict how corrupting a 'Sender Head' will change the attention of a 'Receiver Head'.\n\n"
            "**--- Key Concepts in the Garden Task (Crucial Background) ---**\n"
            "To understand the hypothesis, you must know these linguistic roles in the context of a sentence. However, in this task, we may not provide the whole sentence, which means that the disambiguation token at the end of the sentence may be hidden for prediction. For example, \"As the criminal shot the woman who told bad jokes [was/for].\"\n"
            "In this sentence:\n"
            "- **Subordinator (SUBORD):** the clause introducer (e.g., 'As', 'While', 'After', 'When').\n"
            "- **Subject (SUBJ):** the agent of the main clause (e.g., 'criminal').\n"
            "- **Ambiguous Verb (VERB):** the main verb whose transitivity drives the NP/Z ambiguity (e.g., 'shot' vs an intransitive verb).\n"
            "- **Object NP Head (OBJ_HEAD):** the head noun of the ambiguous object phrase (e.g., 'woman').\n"
            "- **Relative Pronoun / Verb (REL_PRON / REL_VERB):** the relative clause that follows the object (e.g., 'who told').\n"
            "The Garden task is about correctly predicting the **disambiguation token** at the sentence end (\"was\" or \"for\").\n\n"
            f"- **Receiver Query Position:** {receiver_attention_position} (the token position whose attention row we evaluate).\n"
            "- This means we look at the receiver head's attention distribution from that specific token in the sentence.\n\n"
            "--- **Core Task & Causal Rules** ---\n"
            f"Your task is to predict how **corrupting Sender Head {sender_head}** will change the attention of downstream receiver heads according to the hypothesis.\n\n"
            "This means you must predict:\n"
            "1. When Sender Head {sender_head} is corrupted, which token(s) the receiver heads will pay **MORE** attention to. This happens because the Sender Head's normal **SUPPRESSION** of that token is removed. You will mark these with `<<token>>`.\n"
            "2. When Sender Head {sender_head} is corrupted, which token(s) the receiver heads will pay **LESS** attention to. This happens because the Sender Head's normal **PROMOTION** of that token is removed. You will mark these with `[[token]]`.\n\n"
            "**MUST FOLLOW RULES:**\n"
            "- **No Extra Text:** Do not include any reasoning or explanations.\n"
            "- **Logical Consistency:** A token CANNOT be marked for both increase `<< >>` and decrease `[[ ]]`.\n"
            "- **No Opposite Markings:** You MUST mark `<< >>` for increase and `[[ ]]` for decrease.\n"
            f"- **Exact Count:** You must highlight exactly {top_k} token(s) for increase and {top_k} token(s) for decrease.\n"
            "- **No Modification:** Do not omit, add, or change any part of the original sentence text.\n"
            "- **Strict Formatting:** Output only a `[PREDICTION]` block with the marked sentence.\n\n"
        )
    user_prompt = (
        "--- **Illustrative Example (How to perform the task):** ---\n"
        "To ensure you understand the method, here is a complete, self-contained example. **DO NOT use the functions or entities from this example for your real task.** Your goal is to learn the *reasoning process*.\n\n"
        "**Fictional Scenario:**\n"
        "- We are studying how a fictional **Sender Head 0.0** affects a fictional **Receiver Head 0.1**.\n"
        "- **Known function of Receiver Head 0.1:** It is a 'Verb Detector Head'. It tries to pay high attention to verbs in a sentence.\n"
        "- **Hypothetical Hypothesis for Sender Head 0.0:** 'Sender Head 0.0 helps the Verb Detector by **promoting** attention to action verbs (like 'shot', 'told') and **suppressing** attention to linking verbs (like 'is', 'was', 'were').'\\n\n"
        "**Example Sentence:** `garden_9999: As the criminal was tired but the woman told jokes [was/for].`\\n\n"
        "**Reasoning Walkthrough (Your thought process):**\n"
        "1.  **Analyze Suppression:** The hypothesis for Sender Head 0.0 says it **suppresses** 'linking verbs'. In the sentence, the linking verb is 'was'.\n"
        "2.  **Predict Effect of Suppression Removal:** If we corrupt Sender Head 0.0, its suppression of 'was' is removed. Therefore, Receiver Head 0.1 (the Verb Detector) will pay **MORE** attention to 'was'.\n"
        "3.  **Apply Marking Rule:** The rule for INCREASED attention is `<<token>>`. So, I will mark `<<was>>`.\n\n"
        "4.  **Analyze Promotion:** The hypothesis for Sender Head 0.0 says it **promotes** 'action verbs'. In the sentence, the action verb is 'told'.\n"
        "5.  **Predict Effect of Promotion Removal:** If we corrupt Sender Head 0.0, its promotion of 'told' is removed. Therefore, Receiver Head 0.1 (the Verb Detector) will pay **LESS** attention to 'told'.\n"
        "6.  **Apply Marking Rule:** The rule for DECREASED attention is `[[token]]`. So, I will mark `[[told]]`.\n\n"
        "**Correct Output for this Fictional Example:**\n"
        "`garden_9999: As the criminal <<was>> tired but the woman [[told]] jokes [was/for].`\n"
        "--- End of Example ---\n\n"
        "**Your Turn:**\n"
        f"**Hypothesis:** {hypothesis}\n"
        "Now, using the **actual hypothesis for Head {sender_head}**, apply the same reasoning process to the following sentence.\n\n"
        f"**Sentence to Analyze:**\n`{display_sid}: {sentence_text}`"
    )
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_prompt}
    ]
    response = await open_router.generate(messages=messages, output_dir=output_dir)
    
    response_text = response.text.strip()
    
    reasoning_match = re.search(r"\[REASONING\]\s*(.*?)\s*\[PREDICTION\]", response_text, re.DOTALL | re.IGNORECASE)
    prediction_matches = re.findall(r"\[PREDICTION\]\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning found."
    marked_sentence = prediction_matches[-1].strip() if prediction_matches else ""
    if marked_sentence:
        candidate_line = None
        for line in marked_sentence.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("`") and line.endswith("`"):
                line = line[1:-1].strip()
            if line.startswith(f"{display_sid}:"):
                candidate_line = line
        if candidate_line:
            marked_sentence = candidate_line
    
    increase, decrease = _parse_and_suffix_tokens(sentence_text, marked_sentence)
    
    return sid, {
        "increase": increase,
        "decrease": decrease,
        "reasoning": reasoning,
        "raw_response": response_text
    }

async def predict_attention_changes_for_sentences(open_router, hypothesis, sentences_data, sender_head, top_k, output_dir, receiver_attention_position, hop_info, with_reasoning):
    """
    让LLM根据假设，为一批句子并行预测注意力变化。
    """
    print("正在根据假设并行预测每个句子的注意力变化...")
    
    tasks = []
    for sid, data in sentences_data.items():
        task = predict_for_single_sentence(
            open_router, hypothesis, sid, data, sender_head, top_k, output_dir, receiver_attention_position, hop_info, with_reasoning
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
            f"--- Sentence: {format_garden_sid(sid)} ---\n"
            f"  [Your Combined Prediction]: {' '.join(pred_words)}\n"
            f"  [Real Combined Answer]:   {' '.join(real_words)}"
        )

    avg_f1 = ((total_f1_increase / count) + (total_f1_decrease / count)) / 2 if count > 0 else 0
    final_feedback = f"Overall Causal F1 Score for this batch: {avg_f1:.2f}\n\n" + "\n".join(feedback_details)
    return avg_f1, final_feedback, ground_truth_details

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


async def refine_hypothesis_combined(open_router, old_hypothesis, sender_head, explanation, output_dir, f1_score, ndcg_score, attention_f1_score, causal_feedback=None, attention_feedback=None, hop_info=""):
    """按 attention -> causal -> synthesis 三步精炼假设。"""
    print("按 attention -> causal -> synthesis 三步精炼假设...")

    causal_f1 = f1_score if f1_score is not None else 0.0
    attn_f1 = attention_f1_score if attention_f1_score is not None else 0.0
    composite_score = np.sqrt(causal_f1 * attn_f1) if attention_feedback else causal_f1

    print(f"因果F1: {causal_f1:.2f}, 注意力F1: {attn_f1:.2f}, 综合F1分数: {composite_score:.2f}")

    common_system_prompt = (
        "**--- Key Concepts in the Garden Task (Crucial Background) ---**\n"
        "To understand the hypothesis, you must know these linguistic roles in the context of a sentence. However, in this task, we may not provide the whole sentence, which means that the disambiguation token at the end of the sentence may be hidden for prediction. For example, \"As the criminal shot the woman who told bad jokes [was/for].\"\n"
        "In this sentence:\n"
        "- **Subordinator (SUBORD):** the clause introducer (e.g., 'As', 'While', 'After', 'When').\n"
        "- **Subject (SUBJ):** the agent of the main clause (e.g., 'criminal').\n"
        "- **Ambiguous Verb (VERB):** the main verb whose transitivity drives the NP/Z ambiguity (e.g., 'shot' vs an intransitive verb).\n"
        "- **Object NP Head (OBJ_HEAD):** the head noun of the ambiguous object phrase (e.g., 'woman').\n"
        "- **Relative Pronoun / Verb (REL_PRON / REL_VERB):** the relative clause that follows the object (e.g., 'who told').\n"
        "You must analyze the discrepancies between predicted and real model behavior to understand how the prior hypothesis mistakenly interprets the head's real function. "
        "Moreover, you must realize that the fact that an attention head pays close attention to a certain token does not contradict the fact that one of its functions is to suppress the expression of this token in the final output. The attention to this token may have the effect of reminding other heads to block this token. At the same time, the attention head's attention to a certain token is not necessarily related to its influence on the downstream heads."
        "\nTwo-hop clarification (A→B→C): A is patched into B (intermediate), and the measured change is at C (target). "
        "This means A affects C *through* B (A changes B's behavior, which in turn changes B's effect on C). "
        "Path patching here replaces/patches A's activations into B, then measures how C's attention changes. "
        "Your refined hypothesis must explain how A influences B and how that influence propagates to C, using the downstream-head descriptions when provided."
        "\n\nReminder: The head may help OR hurt the task; avoid assuming it is beneficial."
    )

    attention_reasoning = ""
    attention_prompt_log = ""
    attention_raw_response = ""
    if attention_feedback:
        attention_user_prompt = (
            f"You are refining a hypothesis for Sender Head {sender_head}.\n"
            f"**Previous Hypothesis (Flawed):**\n{old_hypothesis}\n\n"
            "--- PERFORMANCE & FEEDBACK ANALYSIS ---\n"
            "**Core Metrics:**\n"
            f"- **Attention F1 Score (What it LOOKS AT): {attn_f1:.2f}**\n"
            f"(Reference only: Attention NDCG Score: {ndcg_score:.2f})\n\n"
            "**Feedback Source: Direct Attention Prediction (What the head LOOKS AT)**\n"
            "This feedback shows how well the hypothesis predicted the head's own Top-K attention targets. A low score means the hypothesis is wrong about what the head is paying attention to.\n"
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
    if causal_feedback:
        causal_user_prompt = (
            f"You are refining a hypothesis for Sender Head {sender_head}.\n"
            f"**Previous Hypothesis (Flawed):**\n{old_hypothesis}\n\n"
            "--- PERFORMANCE & FEEDBACK ANALYSIS ---\n"
            "**Core Metrics:**\n"
            f"- **Causal F1 Score (What it DOES): {causal_f1:.2f}**\n\n"
            "**Feedback Source: Causal Effect Prediction (What the head DOES)**\n"
            "This feedback reveals the head's true downstream function (suppression/promotion). A low score means the hypothesis is wrong about the head's actual effect.\n"
            f"{causal_feedback}\n\n"
            "**Your Task:**\n"
            "- Analyze the causal-effect errors and trade-offs shown above.\n"
            "- Analyze what the previous hypothesis got right about the head's causal-side effect.\n"
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

    synthesis_user_prompt = (
        f"You are refining a hypothesis for Sender Head {sender_head}.\n"
        f"**Previous Hypothesis (Flawed):**\n{old_hypothesis}\n\n"
        "--- PERFORMANCE & FEEDBACK ANALYSIS ---\n"
        "**Core Metrics:**\n"
        f"- **Causal F1 Score (What it DOES): {causal_f1:.2f}**\n"
        f"- **Attention F1 Score (What it LOOKS AT): {attn_f1:.2f}**\n"
        f"- **Composite F1 Score (Geometric Mean): {composite_score:.2f}**\n"
        f"(Reference only: Attention NDCG Score: {ndcg_score:.2f})\n\n"
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
        "- You may reframe the mechanism, but do not make the sign of the head's effect ambiguous.\n"
        f"- Keep the multi-hop setting in view: {hop_info or 'Use the provided receiver/intermediate context.'}\n\n"
        "**Response Format (Strict):**\n"
        "1. **[REASONING]:** Start with this tag. Explicitly analyze how you reconciled the attention-side and causal-side evidence into one coherent mechanism.\n"
        "2. **[HYPOTHESIS]:** Start with this tag. Provide one clean standalone hypothesis describing this unified mechanism.\n"
        "    - This paragraph must clearly and abstractly describe what this head is doing (functionally).\n"
        "    - Do not include tokens, examples, scores, or error context in the `[HYPOTHESIS]` paragraph.\n"
        "    - The hypothesis should sound like a standalone description of the head's role in the model with no concessions or negations.\n"
        "    - The hypothesis should describe the dominant functional behavior using precise linguistic, semantic, or structural terminology. Avoid overly abstract or generic phrasing like 'tracks entities' or 'maintains context'."
    )
    synthesis_raw_response, synthesis_prompt_log = await _generate_refine_substep(
        open_router, common_system_prompt, synthesis_user_prompt, output_dir
    )

    synthesis_reasoning = _extract_tagged_section(synthesis_raw_response, "REASONING") or "No reasoning found in response."
    new_hypothesis = _extract_tagged_section(synthesis_raw_response, "HYPOTHESIS") or extract_hypothesis_text(synthesis_raw_response) or old_hypothesis

    reasoning = (
        "=== ATTENTION REASONING ===\n"
        f"{attention_reasoning or 'N/A'}\n\n"
        "=== CAUSAL REASONING ===\n"
        f"{causal_reasoning or 'N/A'}\n\n"
        "=== SYNTHESIS REASONING ===\n"
        f"{synthesis_reasoning}"
    )
    prompt_log = (
        "=== ATTENTION STEP ===\n"
        f"{attention_prompt_log}\n\n"
        "=== CAUSAL STEP ===\n"
        f"{causal_prompt_log}\n\n"
        "=== SYNTHESIS STEP ===\n"
        f"---SYSTEM PROMPT---\n{common_system_prompt}\n\n---USER PROMPT---\n{synthesis_user_prompt}"
    )
    raw_response_log = (
        "=== ATTENTION STEP ===\n"
        f"{attention_raw_response}\n\n"
        "=== CAUSAL STEP ===\n"
        f"{causal_raw_response}\n\n"
        "=== SYNTHESIS STEP ===\n"
        f"{synthesis_raw_response}"
    )

    print("精炼后的假设:", new_hypothesis)
    return new_hypothesis, reasoning, prompt_log, raw_response_log


def _parse_and_suffix_tokens(original_sentence_text: str, marked_sentence: str, increase_markers: tuple = ("<<", ">>"), decrease_markers: tuple = ("[[", "]]")) -> tuple[list[str], list[str]]:
    """
    Extract marked tokens with suffixes, matching their first valid occurrence order.
    """
    original_tokens = word_tokenize(original_sentence_text)
    if not original_tokens:
        return [], []

    global_counts = defaultdict(int)
    for token in original_tokens:
        global_counts[normalize_token(token)] += 1

    running_counts = defaultdict(int)
    suffixed_tokens = []
    for token in original_tokens:
        norm_token = normalize_token(token)
        running_counts[norm_token] += 1
        if global_counts.get(norm_token, 0) > 1:
            suffixed_tokens.append(f"{token.strip()}_{running_counts[norm_token]}")
        else:
            suffixed_tokens.append(token.strip())

    pattern = re.compile(r"<<([^<>]+)>>|\[\[([^\[\]]+)\]\]")
    matches = []
    for m in pattern.finditer(marked_sentence):
        inc = m.group(1)
        dec = m.group(2)
        if inc is not None:
            matches.append(("inc", inc))
        elif dec is not None:
            matches.append(("dec", dec))

    increase_suffixed, decrease_suffixed = [], []
    search_start = 0
    for kind, content in matches:
        tokens = word_tokenize(content)
        if not tokens:
            continue
        token = tokens[0]
        norm_token = normalize_token(token)
        found_idx = None
        for i in range(search_start, len(original_tokens)):
            if normalize_token(original_tokens[i]) == norm_token:
                found_idx = i
                search_start = i + 1
                break
        if found_idx is None:
            continue
        if kind == "inc":
            increase_suffixed.append(suffixed_tokens[found_idx])
        else:
            decrease_suffixed.append(suffixed_tokens[found_idx])

    return increase_suffixed, decrease_suffixed


async def evaluate_single_hypothesis(hypothesis, open_router, validation_dataset, sender_head, top_k_causal, top_k_attention, attention_ground_truth, output_dir, sender_attention_position, receiver_attention_position, hop_info, with_reasoning):
    """
    【新增辅助函数】在验证集上完整评估单个假设，用于并行化。
    """
    print(f"  - 开始评估假设: \"{hypothesis[:80]}...\"")
    # 在验证集上运行因果预测
    val_causal_preds, _, _ = await predict_attention_changes_for_sentences(
        open_router, hypothesis, validation_dataset, sender_head, top_k_causal, output_dir, receiver_attention_position, hop_info, with_reasoning
    )
    val_f1, _, val_causal_gt = evaluate_predictions(val_causal_preds, validation_dataset, sender_head, top_k=top_k_causal)

    # 在验证集上运行注意力预测
    val_ndcg = 0.0
    val_f1_att = 0.0
    val_att_preds = None
    val_att_gt = None
    if attention_ground_truth:
        val_sids_att = {sid for sid in validation_dataset if sid in attention_ground_truth}
        if val_sids_att:
            tasks = [
                predict_top_k_attenders_for_sentence(
                    open_router, hypothesis, sid, attention_ground_truth[sid], sender_head, top_k_attention, output_dir, sender_attention_position, with_reasoning
                )
                for sid in val_sids_att
            ]
            results = await asyncio.gather(*tasks)
            val_att_preds = {sid: result for sid, result in results}
            val_ndcg, val_f1_att, _ = evaluate_top_k_predictions(val_att_preds, attention_ground_truth, top_k_attention)
            val_att_gt = {sid: attention_ground_truth[sid] for sid in val_sids_att}
    
    print(f"  - 评估完成: Causal F1={val_f1:.2f}, Attn NDCG={val_ndcg:.2f}, Attn F1={val_f1_att:.2f}")
    
    # 返回该假设的完整验证结果
    return {
        "hypothesis": hypothesis,
        "validation_scores": {
            "causal_f1": val_f1, 
            "direct_attention_ndcg": val_ndcg,
            "direct_attention_f1": val_f1_att
        },
        "validation_details": {
            "causal_predictions": val_causal_preds, "causal_ground_truth": val_causal_gt,
            "attention_predictions": val_att_preds, "attention_ground_truth": val_att_gt
        }
    }
    
async def main():
    args = await parse_arguments()
    with_reasoning = args.with_reasoning
    optimize_only = args.optimize_only
    validate_every = max(1, int(args.validate_every))
    validation_sample_size = max(0, int(args.validation_sample_size))
    test_sample_size = max(0, int(args.test_sample_size))
    test_all_validations = bool(args.test_all_validations)
    sender_head = (args.layer, args.head)
    output_dir = args.output_dir
    receiver_heads = [h.strip() for h in args.receiver_heads.split(",") if h.strip()] if args.receiver_heads else []
    receiver_descriptions = {}
    if args.receiver_descriptions_file:
        try:
            with open(args.receiver_descriptions_file, "r", encoding="utf-8") as f:
                receiver_descriptions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"警告: 无法加载 receiver 描述文件 {args.receiver_descriptions_file}: {e}")

    epochs = 8
    train_batch_size = 10
    init_batch_size = 5
    top_k_attention = 2
    top_k_causal = 1

    meta = load_receiver_group_meta(output_dir)
    sender_attention_position = (meta.get("attention_position") or "end").upper()
    receiver_attention_position = (meta.get("receiver_attention_position") or "end").upper()
    run_type = (meta.get("run_type") or "middle").strip().lower()
    intermediate_heads = [h for h in (meta.get("intermediate_heads") or []) if h]
    target_head = (meta.get("target_head") or "").strip()

    data_source_dir = (args.data_source_dir or "").strip()
    full_causal_dataset_path = os.path.join(data_source_dir, "causal_dataset.json") if data_source_dir else args.causal_dataset
    try:
        with open(full_causal_dataset_path, "r") as f:
            full_causal_dataset = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法加载因果数据集 {full_causal_dataset_path}: {e}")
        return

    attention_scores_dataset_path = (
        os.path.join(data_source_dir, "attention_scores_ground_truth.jsonl")
        if data_source_dir
        else os.path.join(output_dir, "attention_scores_ground_truth.jsonl")
    )
    attention_ground_truth = {}
    try:
        with open(attention_scores_dataset_path, "r") as f:
            all_data = json.load(f)
            for data in all_data:
                if "key" in data and "attention_scores" in data:
                    raw_tokens = [item.get("token", "") for item in data["attention_scores"]]
                    suffixed_tokens = _suffix_tokens_in_order(raw_tokens)
                    attention_ground_truth[data["key"]] = {
                        "sentence_text": data.get("original_sentence", ""),
                        "tokens": raw_tokens,
                        "tokens_suffixed": suffixed_tokens,
                        "scores": [item.get("score", 0.0) for item in data["attention_scores"]],
                    }
        if not attention_ground_truth:
            attention_ground_truth = None
    except Exception as e:
        print(f"警告: 无法加载直接注意力基准数据 {attention_scores_dataset_path}: {e}")
        attention_ground_truth = None

    if attention_ground_truth:
        summary_file_path = os.path.join(output_dir, "attention_top2_summary.txt")
        try:
            with open(summary_file_path, "w", encoding="utf-8") as f:
                for sid in sorted(attention_ground_truth.keys()):
                    data = attention_ground_truth[sid]
                    tokens = data.get("tokens_suffixed") or data.get("tokens", [])
                    scores = data.get("scores", [])
                    if not tokens or not scores:
                        continue
                    ranked = sorted(zip(tokens, scores), key=lambda x: x[1], reverse=True)
                    f.write(f"Sentence ID: {format_garden_sid(sid)}\n")
                    f.write(f"Sentence: {data.get('sentence_text', 'N/A')}\n")
                    f.write(f"Top-1 Token: {ranked[0][0] if len(ranked) > 0 else 'N/A'}\n")
                    f.write(f"Top-2 Token: {ranked[1][0] if len(ranked) > 1 else 'N/A'}\n")
                    f.write("-" * 50 + "\n")
        except Exception as e:
            print(f"错误：生成注意力规律总览文件时失败: {e}")

    train_dataset, validation_dataset, test_dataset = split_dataset(full_causal_dataset)
    open_router = initialize_openrouter(model=args.model)

    receiver_heads_text = ", ".join(receiver_heads) if receiver_heads else "unknown"
    if run_type == "middle_plus" and intermediate_heads and target_head:
        hop_info = (
            f"TWO-HOP (A→B→C). A=Sender Head {sender_head}. "
            f"B=Intermediate Head(s) {', '.join(intermediate_heads)}. "
            f"C=Target Head {target_head}. Causal effects are measured at C."
        )
    else:
        hop_info = (
            f"SINGLE-HOP (A→B). A=Sender Head {sender_head}. "
            f"B=Receiver Head(s) {receiver_heads_text}. Causal effects are measured at B."
        )

    downstream_text = ", ".join(receiver_heads) if receiver_heads else "the downstream heads of interest"
    receiver_desc_lines = []
    if receiver_descriptions:
        receiver_desc_lines.append("Known behaviors of the downstream attention heads:")
        ordered_heads = receiver_heads if receiver_heads else sorted(receiver_descriptions.keys())
        group_summary = None
        if len(ordered_heads) > 1:
            heads_with_desc = [h for h in ordered_heads if receiver_descriptions.get(h)]
            if len(heads_with_desc) > 1:
                group_summary = await summarize_head_group(
                    open_router, receiver_descriptions, ordered_heads, "target_heads", output_dir, task="garden"
                )
        if group_summary:
            receiver_desc_lines.append(f"- **Heads {', '.join(ordered_heads)} (summary):** {group_summary}")
        else:
            for head in ordered_heads:
                desc = receiver_descriptions.get(head)
                if desc:
                    receiver_desc_lines.append(f"- **Head {head}**: {desc}")
    else:
        receiver_desc_lines.append("- The downstream heads attend to disambiguation cues; the sender head changes their attention and thereby changes the garden task outcome.")
    receiver_desc_block = "\n".join(receiver_desc_lines)

    explanation = (
        f"Background Context:\n"
        f"- Sender head {sender_head} is believed to function by influencing downstream heads ({downstream_text}).\n"
        f"- These downstream heads attend to disambiguation cues; any disruption introduced by {sender_head} will change their attention patterns and therefore the garden task outcome.\n"
        f"{receiver_desc_block}\n\n"
        f"Your task is to hypothesize the function of the sender head {sender_head} mainly based on how it causally affects the attention of these downstream heads."
    )

    iteration_results_dir = os.path.join(output_dir, "iteration_results")
    validation_results_dir = os.path.join(output_dir, "validation_results")
    os.makedirs(iteration_results_dir, exist_ok=True)
    os.makedirs(validation_results_dir, exist_ok=True)
    full_run_log = []
    validation_history = []

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
            best = max(validation_history, key=lambda r: r.get("validation_scores", {}).get("composite_score", 0.0))
            with open(os.path.join(output_dir, "best_hypothesis.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "head": f"{sender_head[0]}.{sender_head[1]}",
                        "iteration": best.get("label"),
                        "best_hypothesis": best.get("hypothesis"),
                        "validation_scores": best.get("validation_scores"),
                        "source": "validation_history",
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    async def evaluate_hypothesis_on_ids(hypothesis, sentence_ids, label):
        if not sentence_ids:
            return None
        subset = {sid: validation_dataset.get(sid, test_dataset.get(sid, train_dataset.get(sid))) for sid in sentence_ids}
        subset = {sid: data for sid, data in subset.items() if data is not None}
        if not subset:
            return None
        result = await evaluate_single_hypothesis(
            hypothesis,
            open_router,
            subset,
            sender_head,
            top_k_causal,
            top_k_attention,
            attention_ground_truth,
            output_dir,
            sender_attention_position,
            receiver_attention_position,
            hop_info,
            with_reasoning,
        )
        if result:
            scores = result.get("validation_scores", {})
            composite = compute_composite_score(
                scores.get("causal_f1", 0.0),
                scores.get("direct_attention_f1", 0.0),
                scores.get("direct_attention_ndcg", 0.0),
            )
            result["label"] = label
            result["sentence_ids"] = sentence_ids
            result["validation_scores"]["composite_score"] = composite
            result["validation_scores"]["composite_f1"] = composite
        return result

    train_cursor = 0
    initial_hypothesis_candidates = []
    for candidate_idx in range(1, 4):
        sampled_sentences, train_cursor = sample_sentences_from_causal_dataset(train_dataset, batch_size=init_batch_size, start_idx=train_cursor)
        causal_examples = get_causal_effects_for_sampling(sampled_sentences, sender_head, top_k=1)
        if not causal_examples:
            print(f"无法为初始候选 {candidate_idx} 计算因果效应，程序终止。")
            return
        hypothesis, prompt = await generate_initial_hypothesis(open_router, sender_head, causal_examples, explanation, output_dir, hop_info)
        if not hypothesis:
            print(f"无法生成初始候选 {candidate_idx} 假设，程序终止。")
            return
        initial_hypothesis_candidates.append(
            {
                "candidate_index": candidate_idx,
                "hypothesis": hypothesis,
                "sampled_sentence_ids": list(sampled_sentences.keys()),
                "prompt": prompt,
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
            candidate_result["initial_hypothesis_prompt"] = candidate.get("prompt", "")
            candidate_result["decision"] = "candidate_evaluated"
            init_candidate_results.append(candidate_result)
            validation_history.append(candidate_result)

    if not init_candidate_results:
        print("错误: 无法评估任何初始候选假设，程序终止。")
        return

    best_init_candidate = max(init_candidate_results, key=lambda r: r.get("validation_scores", {}).get("composite_score", 0.0))
    init_val_result = copy.deepcopy(best_init_candidate)
    init_val_result["label"] = "validation_epoch_0_initial"
    init_val_result["decision"] = "selected_as_initial_start"
    validation_history.append(init_val_result)
    persist_validation_artifacts()

    current_hypothesis = init_val_result.get("hypothesis", "")
    initial_hypothesis = current_hypothesis
    initial_hypothesis_prompt = init_val_result.get("initial_hypothesis_prompt", "")
    previous_val_result = init_val_result

    for epoch in range(1, epochs + 1):
        iteration_log = {"epoch": epoch, "hypothesis_before": current_hypothesis}
        test_sentences_batch, train_cursor = sample_sentences_from_causal_dataset(train_dataset, batch_size=train_batch_size, start_idx=train_cursor)

        causal_predictions = {}
        causal_feedback = None
        causal_ground_truth = {}
        causal_f1 = 0.0
        if optimize_only != "attention":
            causal_predictions, _, _ = await predict_attention_changes_for_sentences(
                open_router, current_hypothesis, test_sentences_batch, sender_head, top_k_causal, output_dir, receiver_attention_position, hop_info, with_reasoning
            )
            causal_f1, causal_feedback, causal_ground_truth = evaluate_predictions(causal_predictions, test_sentences_batch, sender_head, top_k=top_k_causal)

        attention_predictions = None
        ndcg_score_val = 0.0
        attention_f1 = 0.0
        attention_feedback = None
        attention_ground_truth_batch = None
        if attention_ground_truth and optimize_only != "causal":
            test_sentences_attention = {sid: data for sid, data in attention_ground_truth.items() if sid in test_sentences_batch}
            attention_predictions = {}
            for batch in chunk_dict(test_sentences_attention, 5):
                tasks = [
                    predict_top_k_attenders_for_sentence(
                        open_router, current_hypothesis, sid, data, sender_head, top_k_attention, output_dir, sender_attention_position, with_reasoning
                    )
                    for sid, data in batch.items()
                ]
                results = await asyncio.gather(*tasks)
                attention_predictions.update({sid: result for sid, result in results})
            ndcg_score_val, attention_f1, attention_feedback = evaluate_top_k_predictions(attention_predictions, attention_ground_truth, top_k_attention)
            attention_ground_truth_batch = {sid: attention_ground_truth[sid] for sid in attention_predictions if sid in attention_ground_truth}

        iteration_log.update(
            {
                "causal_f1": causal_f1,
                "attention_ndcg": ndcg_score_val,
                "attention_f1": attention_f1,
                "causal_feedback": causal_feedback,
                "attention_feedback": attention_feedback,
            }
        )

        current_hypothesis, reasoning, prompt, raw_response = await refine_hypothesis_combined(
            open_router,
            current_hypothesis,
            sender_head,
            explanation,
            output_dir,
            f1_score=causal_f1,
            ndcg_score=ndcg_score_val,
            attention_f1_score=attention_f1,
            causal_feedback=causal_feedback if optimize_only != "attention" else None,
            attention_feedback=attention_feedback if optimize_only != "causal" else None,
            hop_info=hop_info,
        )
        iteration_log.update(
            {
                "hypothesis_after": current_hypothesis,
                "refinement_reasoning": reasoning,
                "refinement_prompt": prompt,
                "refinement_raw_response": raw_response,
            }
        )
        full_run_log.append(iteration_log)

        with open(os.path.join(output_dir, "iteration_results", f"iteration_{epoch}.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "hypothesis": current_hypothesis,
                    "scores": {
                        "causal_f1": causal_f1,
                        "attention_f1": attention_f1,
                        "attention_ndcg": ndcg_score_val,
                    },
                    "predicted_causal": causal_predictions,
                    "causal_ground_truth": causal_ground_truth,
                    "predicted_attention": attention_predictions,
                    "attention_ground_truth": attention_ground_truth_batch,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        if epoch % validate_every == 0:
            val_result = await evaluate_hypothesis_on_ids(current_hypothesis, val_ids, f"validation_epoch_{epoch}")
            if val_result:
                composite = float(val_result.get("validation_scores", {}).get("composite_score", 0.0))
                prev_score = float(previous_val_result.get("validation_scores", {}).get("composite_score", 0.0)) if previous_val_result else None
                if previous_val_result is not None and composite < prev_score:
                    rollback_hypothesis = previous_val_result.get("hypothesis", "")
                    if rollback_hypothesis:
                        current_hypothesis = rollback_hypothesis
                    val_result["decision"] = "rollback_to_previous_validation"
                    val_result["previous_validation_composite_score"] = prev_score
                    val_result["previous_validation_composite_f1"] = prev_score
                else:
                    val_result["decision"] = "accept_current"
                    previous_val_result = val_result
                validation_history.append(val_result)
                persist_validation_artifacts()

    hypothesis_for_test = current_hypothesis
    test_ids_all = list(test_dataset.keys())
    test_ids = test_ids_all[:test_sample_size] if test_sample_size > 0 and len(test_ids_all) > test_sample_size else test_ids_all
    test_result = await evaluate_hypothesis_on_ids(hypothesis_for_test, test_ids, "test_final")
    if test_result:
        with open(os.path.join(output_dir, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump(test_result, f, ensure_ascii=False, indent=2)

    same_as_final = (initial_hypothesis or "").strip() == (hypothesis_for_test or "").strip()
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

    final_log_entry = {
        "head": f"{sender_head[0]}.{sender_head[1]}",
        "typename": args.typename,
        "initial_hypothesis_candidates": initial_hypothesis_candidates,
        "initial_hypothesis_prompt": initial_hypothesis_prompt,
        "full_run_log": full_run_log,
        "all_hypotheses_validation_results": validation_history,
        "selected_validation_last": validation_history[-1] if validation_history else None,
        "hypothesis_used_for_test": hypothesis_for_test,
        "initial_hypothesis": initial_hypothesis,
        "initial_test_results": initial_test_result,
        "test_results": test_result,
    }
    final_log_path = os.path.join(output_dir, "final_result_all_rounds.json")
    with open(final_log_path, "w", encoding="utf-8") as f:
        json.dump(final_log_entry, f, ensure_ascii=False, indent=2)
    print(f"\n所有运行数据和最终的并行验证结果已统一保存至: {final_log_path}")

if __name__ == "__main__":
    asyncio.run(main())
