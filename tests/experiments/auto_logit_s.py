import asyncio
import sys
import json
import os
import numpy as np
import re
import random
from collections import defaultdict
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

# --- 阶段一：从数据生成初始假设 ---

def get_overall_causal_effect(full_causal_dataset, sender_head, top_k=5):
    """
    直接从原始的、逐句的因果数据集中动态计算平均因果效应。
    """
    sender_key_prefix = f"{sender_head[0]}.{sender_head[1]}->"
    
    all_diff_vectors = []
    representative_tokens = None

    print(f"正在从 {len(full_causal_dataset)} 条句子中动态计算 Sender {sender_head} 的平均因果效应...")
    for sentence_id, data in full_causal_dataset.items():
        if "diff_vectors" not in data or "tokens" not in data:
            continue
        
        if representative_tokens is None:
            representative_tokens = data["tokens"]

        for path_key, diff_vector in data["diff_vectors"].items():
            if path_key.startswith(sender_key_prefix):
                # diff_vector 已经是最后一行的注意力差异
                if diff_vector: # 确保列表不为空
                    all_diff_vectors.append(np.array(diff_vector))

    if not all_diff_vectors:
        print(f"错误: 未能在数据集中找到 Sender {sender_head} 的任何有效因果路径数据。")
        return None, None

    # 确保所有向量长度一致，以最短的为准
    min_len = min(len(v) for v in all_diff_vectors)
    all_diff_vectors_padded = [v[:min_len] for v in all_diff_vectors]
    
    avg_diff_vector = np.mean(all_diff_vectors_padded, axis=0)
    
    str_tokens = representative_tokens[:min_len]

    token_changes = sorted(zip(str_tokens, avg_diff_vector), key=lambda x: x[1], reverse=True)
    
    top_increase = [{"token": token, "score": float(score)} for token, score in token_changes[:top_k]]
    top_decrease = [{"token": token, "score": float(score)} for token, score in token_changes[-top_k:]]
    
    return {"increase": top_increase, "decrease": top_decrease}, str_tokens

async def generate_initial_hypothesis(open_router, sender_head, causal_effects, explanation, output_dir):
    """
    根据聚合后的因果效应数据，生成初始假设。
    """
    print(f"为头 {sender_head} 生成初始假设...")
    
    increase_tokens = ", ".join([f"'{item['token']}' ({item['score']:.3f})" for item in causal_effects['increase']])
    decrease_tokens = ", ".join([f"'{item['token']}' ({item['score']:.3f})" for item in causal_effects['decrease']])
    
    prompt = (
        "You are a meticulous AI researcher investigating the function of a specific attention head in a transformer model for the Indirect Object Identification (IOI) task. "
        "The IOI task requires the model to correctly identify the recipient of an action in a sentence (e.g., in 'John gave a book to Mary', 'Mary' is the indirect object).\n\n"
        
        f"**Experimental Data for Sender Head {sender_head}:**\n\n"
        
        "**Crucial Interpretability Context (Path Patching):**\n"
        "A path patching experiment was conducted. This involves corrupting the output of the sender head and measuring the causal effect on downstream heads known to be important for the IOI task. You MUST interpret the results correctly:\n"
        "- An **INCREASE** in attention to a token means the sender head normally **SUPPRESSES** attention to that token. Its original function is **inhibitory** towards this token.\n"
        "- A **DECREASE** in attention to a token means the sender head normally **PROMOTES** attention to that token. Its original function is **contributory** towards this token.\n\n"

        f"**Background Information on Downstream Heads:**\n{explanation}\n\n"
        
        "**Average Causal Effects Observed:**\n"
        "When Sender Head {sender_head}'s output is corrupted:\n"
        f"- Attention to these tokens **INCREASES** (implying the head normally SUPPRESSES them): {increase_tokens}.\n"
        f"- Attention to these tokens **DECREASES** (implying the head normally PROMOTES them): {decrease_tokens}.\n\n"
        
        "**Your Task & Guidelines for Hypothesis Formulation:**\n"
        "1. **Synthesize, Don't List:** Do not just describe the data. Synthesize a clear, coherent hypothesis that explains the head's fundamental mechanism.\n"
        "2. **Explain the 'Why':** Your hypothesis must explain *why* the head promotes attention to certain tokens while suppressing others, and how this behavior contributes to or hinders the overall IOI task.\n"
        "3. **Be Precise & Falsifiable:** Use precise linguistic, semantic, or structural terminology (e.g., 'attends to the first name token', 'suppresses the subject token'). Your hypothesis must be testable and falsifiable. **AVOID** vague phrases like 'tracks entities', 'maintains context', 'semantic relevance', or 'contextual cues'.\n"
        "4. **Categorical Analysis:** Before writing the hypothesis, think about how the head might behave differently across various sentence structures or entity roles. This analysis should inform your final hypothesis.\n"
        "5. **Standalone & Clear:** The final hypothesis should be a single, standalone paragraph that clearly articulates the head's function without referencing the experiment itself.\n\n"

        "**Example Thought Process (for you to follow internally):**\n"
        "1.  *Reasoning*: First, I analyze the data. The head seems to promote attention to tokens that are potential indirect objects and suppress attention to tokens that are subjects. This suggests a role in differentiating between the agent and the recipient.\n"
        "2.  *Hypothesis Formulation*: Based on this, I will write a concise paragraph describing this function.\n\n"

        "**Response Format:**\n"
        "Begin with a reasoning paragraph analyzing the data and your thought process. Then, write a clean summary paragraph starting with `[HYPOTHESIS]:`."
    )

    messages = [{"role": "user", "content": prompt}]
    response = await open_router.generate(messages=messages, output_dir=output_dir)
    
    hypothesis = extract_hypothesis_text(response.text)
    print("初始假设已生成:", hypothesis)
    return hypothesis, prompt

# --- 阶段二：迭代验证与精炼 ---

def sample_sentences_from_causal_dataset(causal_data, batch_size=10):
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
    
    increase = [normalize_token(t) for t in re.findall(r"<<(.*?)>>", marked_sentence)]
    decrease = [normalize_token(t) for t in re.findall(r"\[\[(.*?)\]\]", marked_sentence)]
    
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

def evaluate_predictions(predictions, ground_truth_data, sender_head):
    """评估LLM的预测与真实数据的匹配度"""
    total_f1_increase, total_f1_decrease = 0, 0
    feedback_details = []
    ground_truth_details = {}
    count = 0

    for sid, pred in predictions.items():
        if sid not in ground_truth_data: continue
        
        # 【修正点 2】增加对旧日志格式的兼容性，进行二次解析
        pred_inc_set = set(pred.get('increase', []))
        pred_dec_set = set(pred.get('decrease', []))

        # 如果increase/decrease列表为空，但存在raw_response，则尝试从中解析
        if not pred_inc_set and not pred_dec_set and 'raw_response' in pred:
            raw_response = pred['raw_response']
            prediction_match = re.search(r"\[PREDICTION\]:\s*(.*)", raw_response, re.DOTALL | re.IGNORECASE)
            marked_sentence = prediction_match.group(1).strip() if prediction_match else ""
            pred_inc_set = {normalize_token(t) for t in re.findall(r"<<(.*?)>>", marked_sentence)}
            pred_dec_set = {normalize_token(t) for t in re.findall(r"\[\[(.*?)\]\]", marked_sentence)}
        
        # 提取该句子的真实因果效应
        real_diffs = ground_truth_data[sid]['diff_vectors']
        tokens = ground_truth_data[sid]['tokens']
        
        # 计算该句子上sender到所有receiver的平均效应
        sentence_vectors = []
        for path_key, diff_vector in real_diffs.items():
            if path_key.startswith(f"{sender_head[0]}.{sender_head[1]}->"):
                if diff_vector:
                    sentence_vectors.append(np.array(diff_vector))
        
        if not sentence_vectors: continue
        
        # 确保长度一致
        min_len = min(len(v) for v in sentence_vectors)
        sentence_vectors_padded = [v[:min_len] for v in sentence_vectors]
        avg_diff_vector = np.mean(sentence_vectors_padded, axis=0)
        
        # 确保token列表长度一致
        tokens_padded = tokens[:min_len]
        token_changes = sorted(zip(tokens_padded, avg_diff_vector), key=lambda x: x[1], reverse=True)
        
        real_inc_set = {normalize_token(t) for t, s in token_changes[:1]} # Top-1 increase
        real_dec_set = {normalize_token(t) for t, s in token_changes[-1:]} # Top-1 decrease
        
        ground_truth_details[sid] = {
            "increase": list(real_inc_set),
            "decrease": list(real_dec_set)
        }
        
     #   pred_inc_set = set(pred.get('increase', []))
     #   pred_dec_set = set(pred.get('decrease', []))
        
        # 计算F1分数
        f1_increase = (2 * len(pred_inc_set.intersection(real_inc_set))) / (len(pred_inc_set) + len(real_inc_set)) if (len(pred_inc_set) + len(real_inc_set)) > 0 else 0
        f1_decrease = (2 * len(pred_dec_set.intersection(real_dec_set))) / (len(pred_dec_set) + len(real_dec_set)) if (len(pred_dec_set) + len(real_dec_set)) > 0 else 0
        
        total_f1_increase += f1_increase
        total_f1_decrease += f1_decrease
        count += 1
        
        feedback_details.append(
            f"- For sentence '{ground_truth_data[sid]['sentence_text']}', you predicted increase in {pred_inc_set} (real: {real_inc_set}) and decrease in {pred_dec_set} (real: {real_dec_set}). F1 score (inc/dec): {f1_increase:.2f}/{f1_decrease:.2f}"
        )

    avg_f1 = ((total_f1_increase / count) + (total_f1_decrease / count)) / 2 if count > 0 else 0
    return avg_f1, "\n".join(feedback_details), ground_truth_details

async def refine_hypothesis(open_router, old_hypothesis, feedback_text, sender_head, explanation, output_dir):
    """根据预测失败的反馈来精炼假设"""
    print("根据反馈精炼假设...")
    system_prompt = (
        "**--- Key Concepts in the IOI Task (Crucial Background) ---**\n"
        "To understand the hypothesis, you must know these linguistic roles in the context of a sentence. However, in an IOI task, we may not provide the whole sentence, which means that IO at the end of the sentence may be hidden for prediction. For example, \"John and Mary went to the park. John gave a backpack to Mary.\"\n"
        "In this sentence:"
        "- **Subject (S):** The one performing the action. (e.g., 'John' gave the backpack).\n"
        "- **Indirect Object (IO):** The recipient of the action. Although the token is not provided directly at the end of the sentence, but you can infer from the context.(e.g., 'Mary' received the backpack).\n"
        "- **Direct Object (DO):** The object being transferred. (e.g., 'a backpack').\n"
    )
    user_prompt = (
        f"You are a meticulous AI researcher refining a hypothesis about Sender Head {sender_head}'s function in the IOI task.\n\n"
        
        f"**Background & Context:**\n"
        f"- **Task:** Indirect Object Identification (IOI).\n"
        f"- **Experiment Details (Path Patching):** The data comes from a **path patching** experiment, a causal intervention technique. Here’s how it works: We run the model on a clean, original sentence. We then 'patch in' (i.e., replace) the output of only **Sender Head {sender_head}** with its output from a different, corrupted sentence (e.g., one with swapped names). We then measure how this surgical change affects the attention of important downstream heads. This isolates the precise causal role of Sender Head {sender_head}.\n\n"
        
        f"- **Crucial Causal Interpretation:** You must interpret the results of this experiment correctly. The feedback you receive shows the *change* in attention when the sender head is corrupted:\n"
        f"  - **An INCREASE in attention to a token proves the head's normal function is to SUPPRESS that token.**\n"
        f"  - **A DECREASE in attention to a token proves the head's normal function is to PROMOTE that token.**\n\n"
        
        f"- **Downstream Heads Info:**\n{explanation}\n\n"
        
        f"**Previous Hypothesis (Flawed):**\n{old_hypothesis}\n\n"
        
        "**Evidence for Refinement: Prediction Performance**\n"
        "The previous hypothesis was tested against real data. The feedback below details its performance, showing both correct and incorrect predictions. Your task is to analyze this feedback to refine the hypothesis.\n"
        f"**Feedback Data:**\n{feedback_text}\n\n"
        
        "**Your Task (Structured Refinement):**\n"
        "1.  **Analyze the Evidence:** Look at the `(real: ...)` part of the feedback. This is the ground truth. Based on the causal rules, what does this data tell you the head's *actual* function is? For example, if the feedback says `(real: {'increase': {'subject'}, 'decrease': {'object'}})` it proves the head's true function is to **suppress the subject** and **promote the object**.\n"
        "2.  **Identify the Flaw:** Compare this true function to the `Previous Hypothesis`. Where did the old hypothesis go wrong? Did it reverse the roles? Did it miss one part of the function?\n"
        "3.  **Formulate a Complete Revision:** Propose a new, more precise hypothesis. **Your new hypothesis MUST describe both the suppression and promotion roles of the head to be considered complete.** It must explain what the head suppresses and what it promotes, and how this dual action helps solve the IOI task.\n\n"
        
        "**Response Format (Strict):**\n"
        "1.  **[REASONING]:** Start with this tag. First, state the head's true function as revealed by the feedback data. Second, explain the specific flaw in the old hypothesis. For example: 'The feedback shows that corrupting the head leads to increased attention on subjects and decreased attention on objects. This proves the head's true function is to suppress subjects and promote objects. The previous hypothesis was flawed because it incorrectly stated the opposite.'\n"
        "2.  **[HYPOTHESIS]:** Start with this tag. Provide a single, clean, standalone paragraph for your new, complete hypothesis. It must describe both a suppression and a promotion effect. For example: 'Sender Head (X, Y) suppresses attention to subject tokens while simultaneously promoting attention to indirect object tokens...'"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = await open_router.generate(messages=messages, output_dir=output_dir)
    
    response_text = response.text

    reasoning_match = re.search(r"\[REASONING\].*?:\s*(.*?)(?=\n.*\[HYPOTHESIS\])", response_text, re.DOTALL | re.IGNORECASE)
    hypothesis_match = re.search(r"\[HYPOTHESIS\]:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning found in response."
    new_hypothesis = hypothesis_match.group(1).strip() if hypothesis_match else response_text.strip()
    prompt = f"---SYSTEM PROMPT---\n{system_prompt}\n\n---USER PROMPT---\n{user_prompt}"
    print("精炼后的假设:", new_hypothesis)
    # 同时返回新假设和推理过程
    return new_hypothesis, reasoning, prompt


def find_best_hypothesis_from_logs(output_dir):
    """从所有旧的causal_results_*.json日志中找到F1分数最高的假设"""
    best_hypothesis = None
    best_f1 = -1.0
    
    try:
        # 寻找所有匹配 'causal_results_*.json' 的文件
        log_files = [f for f in os.listdir(output_dir) if re.match(r'causal_results_\d+\.json', f)]
        
        for log_file in log_files:
            path = os.path.join(output_dir, log_file)
            with open(path, 'r') as f:
                data = json.load(f)
                for iteration_data in data:
                    if iteration_data.get('f1_score', -1) > best_f1:
                        best_f1 = iteration_data['f1_score']
                        best_hypothesis = iteration_data['hypothesis']
                        
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"读取日志时出错: {e}。将从头开始生成假设。")
        return None, -1.0
        
    if best_hypothesis:
        print(f"从过往日志中找到最佳假设 (F1 Score: {best_f1:.2f})")
    
    return best_hypothesis, best_f1


async def main():
    # 1. 初始化和加载数据
    args = await parse_arguments()
    sender_head = (args.layer, args.head)
    output_dir = args.output_dir
    max_iterations = 5
    top_k = 1 # 我们专注于预测最显著的变化

    # 现在只加载唯一的原始数据源
    full_causal_dataset_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ioi", "path_patching", "causal_dataset.json")
    try:
        with open(full_causal_dataset_path, "r") as f:
            full_causal_dataset = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法加载 {full_causal_dataset_path}: {e}\n请先运行 run_path_patching.py。")
        return

    train_dataset, validation_dataset = split_dataset(full_causal_dataset)
    
    open_router = initialize_openrouter()
    print(f"模型和数据加载完毕。开始为头 {sender_head} 进行自动化假设检验...")
    
    full_run_log = []
    
    initial_hypothesis_prompt = None
    initial_hypothesis, best_f1_so_far = find_best_hypothesis_from_logs(output_dir)
    if not initial_hypothesis or best_f1_so_far < 0.1: # 如果历史最佳F1太低，也重新开始
        print("未找到有效的历史假设或历史分数过低。重新生成初始假设...")
        explanation = f"""
            Background Context: The sender head {sender_head} is believed to function by influencing a circuit of downstream heads. The primary role of these downstream heads is to attend to the indirect object (IO) token in a sentence. Here is a summary of what is known about the key heads (9.6 and 9.9) that we are measuring the effect on:
            - **Head 9.6**: ...
            - **Head 9.9**: ...
            Your task is to hypothesize the function of the sender head {sender_head} mainly based on how it causally affects the attention of these downstream heads.
            """
        overall_causal_effect, _ = get_overall_causal_effect(train_dataset, sender_head, top_k=1)
        
        if not overall_causal_effect:
            print("无法生成初始假设，程序终止。")
            return

        current_hypothesis, initial_hypothesis_prompt = await generate_initial_hypothesis(open_router, sender_head, overall_causal_effect, explanation, output_dir)
        
        if not current_hypothesis:
            print("无法生成初始假设，程序终止。")
            return
    else:
        print("使用历史最佳假设作为本次运行的起点。")
        current_hypothesis = initial_hypothesis
        
    refinement_reasoning = "N/A (This is the initial hypothesis or loaded from previous best)"
    
    for i in range(1, max_iterations + 1):
        print(f"\n--- Iteration {i}/{max_iterations} ---")
        
        test_sentences = sample_sentences_from_causal_dataset(train_dataset, batch_size=10)
        
        predictions, _, _ = await predict_attention_changes_for_sentences(
            open_router, current_hypothesis, test_sentences, sender_head, top_k, output_dir
        )
        
        f1_score, feedback_text, ground_truth_details = evaluate_predictions(predictions, train_dataset, sender_head)
        print(f"Iteration {i} Training F1 Score: {f1_score:.2f}")
        
        iteration_log_entry = {
            "type": "iteration",
            "iteration_number": i,
            "hypothesis": current_hypothesis,
            "refinement_reasoning_for_this_hypothesis": refinement_reasoning,
            "f1_score": f1_score,
            "training_batch_sentences": {sid: data['sentence_text'] for sid, data in test_sentences.items()},
            "predictions": predictions,
            "ground_truth": ground_truth_details,
            "feedback_to_llm": feedback_text,
            "prompts": {}
        }
        
        if i == 1 and initial_hypothesis_prompt:
            iteration_log_entry["prompts"]["initial_hypothesis_prompt"] = initial_hypothesis_prompt

        full_run_log.append(iteration_log_entry)
        
        if f1_score > 0.8:
            print(f"假设收敛成功，F1分数为 {f1_score:.2f}，满足阈值 > 0.8。")
            break
        
        if i == max_iterations:
            print("已达到最大迭代次数，停止。")
            break
            
        current_hypothesis, refinement_reasoning, refinement_prompt = await refine_hypothesis(
            open_router, current_hypothesis, feedback_text, sender_head, explanation, output_dir
        )
        
        full_run_log[-1]["prompts"]["refinement_prompt_for_next_iteration"] = refinement_prompt

        if not current_hypothesis:
            print("精炼失败，无法生成新假设，程序终止。")
            break

    print("\n--- Final Validation Phase ---")
    best_hypothesis_from_run = None
    best_f1_from_run = -1.0
    best_iteration_num = -1

    for entry in full_run_log:
        if entry["type"] == "iteration" and entry["f1_score"] > best_f1_from_run:
            best_f1_from_run = entry["f1_score"]
            best_hypothesis_from_run = entry["hypothesis"]
            best_iteration_num = entry["iteration_number"]

    if best_hypothesis_from_run:
        print(f"本次运行中最佳假设来自第 {best_iteration_num} 轮 (F1: {best_f1_from_run:.2f})。正在验证集上评估...")
        
        validation_predictions, _, _ = await predict_attention_changes_for_sentences(
            open_router, best_hypothesis_from_run, validation_dataset, sender_head, top_k, output_dir
        )
        
        validation_f1, validation_feedback, validation_ground_truth = evaluate_predictions(validation_predictions, validation_dataset, sender_head)
        print(f"最终验证集 F1 分数: {validation_f1:.2f}")

        validation_log_entry = {
            "type": "final_validation",
            "best_hypothesis_from_run": best_hypothesis_from_run,
            "best_hypothesis_iteration_source": best_iteration_num,
            "best_hypothesis_training_f1": best_f1_from_run,
            "validation_f1_score": validation_f1,
            "validation_sentences": {sid: data['sentence_text'] for sid, data in validation_dataset.items()},
            "validation_predictions": validation_predictions, # 验证预测现在也包含推理和原始回复
            "validation_ground_truth": validation_ground_truth,
            "validation_feedback": validation_feedback,
            "validation_prediction_prompt": "N/A - Prompts are now per-sentence within predictions."
        }
        full_run_log.append(validation_log_entry)
    else:
        print("本次运行未能产生有效假设进行验证。")

    final_log_path = os.path.join(output_dir, f"full_run_log_{args.rounds}.json")
    with open(final_log_path, "w") as f:
        json.dump(full_run_log, f, indent=2)
    print(f"\n所有运行数据已统一保存至: {final_log_path}")
    
    # 清理旧的、分散的文件
    old_causal_results = os.path.join(output_dir, f"causal_results_{args.rounds}.json")
    old_best_result = os.path.join(output_dir, f"best_result_{args.rounds}.jsonl")
    if os.path.exists(old_causal_results):
        os.remove(old_causal_results)
    if os.path.exists(old_best_result):
        os.remove(old_best_result)


if __name__ == "__main__":
    asyncio.run(main())