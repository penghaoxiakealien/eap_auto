import asyncio
import sys
import json
import os
import re
import random
from collections import defaultdict
import argparse
from typing import List, Dict, Any, Tuple
import nltk

# 确保可以导入api模块
sys.path.append("/home/wangziran/eap_auto/")
from api import OpenRouter
from nltk.tokenize import word_tokenize

# --- 1. 参数解析与初始化 ---

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Automated hypothesis generation for a backup attention head under ablation conditions.")
    parser.add_argument("--target_head", type=str, required=True, help="The target head to analyze (e.g., '11.2').")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the attention analysis JSON file from ablated_att.py.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results and logs.")
    parser.add_argument("--rounds", type=int, default=5, help="Number of refinement rounds.")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of samples per round.")
    return parser.parse_args()

def initialize_openrouter(model: str = "gpt-4o"):
    """初始化OpenRouter API"""
    api_key = "sk-Z3pwy4dD8WY2XZlbzch66NP5hQIoFKeU7KvI2XD8bQSyFVGO" # 请替换为您的API Key
    return OpenRouter(model=model, api_key=api_key)

def extract_hypothesis_from_response(response_text: str) -> Tuple[str, str]:
    """从LLM响应中提取推理和假设"""
    reasoning_match = re.search(r"\[REASONING\]\s*(.*?)\s*\[HYPOTHESIS\]", response_text, re.DOTALL | re.IGNORECASE)
    hypothesis_match = re.search(r"\[HYPOTHESIS\]:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning found."
    hypothesis = hypothesis_match.group(1).strip() if hypothesis_match else response_text.strip()
    
    return reasoning, hypothesis

# --- 2. 数据加载与处理 (适配ablated_att.py的输出) ---

def load_attention_analysis_data(input_file: str) -> List[Dict[str, Any]]:
    """加载ablated_att.py生成的分析文件"""
    try:
        with open(input_file, "r") as f:
            data = json.load(f)
        results = data.get('results', [])
        print(f"✅ 成功从 {input_file} 加载了 {len(results)} 条分析样本。")
        return results
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ 错误: 无法加载或解析文件 {input_file}: {e}")
        return []

def _get_suffixed_word_map(original_text):
    """辅助函数：为句子的每个词生成带后缀的映射"""
    words = word_tokenize(original_text)
    global_counts = defaultdict(int)
    for w in words:
        global_counts[w.lower()] += 1
    
    running_counts = defaultdict(int)
    suffixed_map = {}
    for i, word in enumerate(words):
        norm_word = word.lower()
        running_counts[norm_word] += 1
        if global_counts[norm_word] > 1:
            suffixed_map[i] = f"{word}_{running_counts[norm_word]}"
        else:
            suffixed_map[i] = word
    return words, suffixed_map

def random_sample_from_analysis_data(all_samples: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
    """从已加载的分析数据中随机采样"""
    if len(all_samples) < batch_size:
        print(f"警告: 样本总数 ({len(all_samples)}) 小于批次大小 ({batch_size})。将使用所有样本。")
        return all_samples
    return random.sample(all_samples, batch_size)

def format_examples_for_prediction(samples: List[Dict[str, Any]]) -> str:
    """将采样出的数据格式化，用于后续的token预测"""
    prompt_lines = []
    for i, sample in enumerate(samples):
        key = f"{i+1}_test"
        sentence = sample.get("sentence_text", "")
        io_token = sample.get("context_info", {}).get("io_token", "N/A")
        
        # 在这个新流程中，我们让LLM预测所有重要的token，所以不预设数量
        # 我们可以根据真实数据中的IO和S来设定一个动态的数量
        analysis = sample.get("attention_analysis", {})
        num_important = 0
        if analysis.get("io_attention"): num_important += 1
        if analysis.get("s_attention"): num_important += 1
        
        prompt_lines.append(f"Example: {key}: {sentence} {{{{{io_token}}}}}")
        prompt_lines.append(f"Number of important tokens: {num_important}")
        prompt_lines.append(f"indirect object: {io_token}")
        prompt_lines.append("")
    return "\n".join(prompt_lines)

# --- 3. 核心：假设生成与精炼的Prompt (继承自auto_NMH.py并改造) ---

def get_initial_hypothesis_prompt(target_head: str, ablated_heads: List[str], example_prompt: str) -> List[Dict[str, str]]:
    """构建生成初始假设的Prompt (BNMH版本)"""
    ablated_heads_function = "are active at the final token of a sentence, attend to previous names, and copy the name they attend to, effectively selecting one name over others."

    system_prompt = (
        "You are a world-class AI researcher specializing in transformer interpretability. Your task is to deduce the function of a specific attention head that has become active *only* under special circumstances.\n\n"
        "**--- CRUCIAL CONTEXT: READ CAREFULLY ---**\n"
        f"You are analyzing **Head {target_head}**. This head's behavior is being observed in a model where several other critical heads ({', '.join(ablated_heads)}) have been **artificially disabled (ablated)**.\n\n"
        f"1.  **The Disabled Heads ({', '.join(ablated_heads)})**: These heads normally perform a key function: they {ablated_heads_function}. Their failure is the reason we are studying Head {target_head}.\n"
        f"2.  **Head {target_head}'s Normal State (Before Ablation)**: Under normal conditions, this head is **unimportant**. Its attention patterns are weak and show no clear function. It does not contribute significantly to the model's performance on the IOI task.\n"
        f"3.  **Head {target_head}'s Current State (After Ablation)**: With the main heads disabled, Head {target_head} has **stepped up**. It now shows a clear, strong attention pattern and its activation now **positively contributes** to the model's ability to correctly identify the Indirect Object (IO).\n\n"
        "Your mission is to formulate a hypothesis that explains **what Head {target_head} is doing now** to compensate for the failure of the main heads.\n\n"
        "**--- Key Concepts in the IOI Task ---**\n"
        "- **Subject (S):** The entity performing the action (e.g., 'Jane' gave the snack).\n"
        "- **Indirect Object (IO):** The recipient of the action (e.g., 'Victoria' received the snack).\n"
        "The goal is to correctly identify the IO.\n\n"
        "**--- Response Guidelines ---**\n"
        "- You will be given examples of Head {target_head}'s attention patterns **after ablation**. Tokens with high attention are marked with `<< >>`.\n"
        "- Your reasoning should focus on how this new attention pattern helps the model distinguish the IO from the S, now that the primary mechanism is broken.\n"
        "- Your final response must contain two parts:\n"
        "  1. **[REASONING]:** A detailed analysis of the provided examples in light of the ablation context.\n"
        "  2. **[HYPOTHESIS]:** A single, concise paragraph describing the emergent function of Head {target_head}."
    )

    user_prompt = (
        f"Based on the critical context provided, analyze the following examples of Head {target_head}'s behavior **after the ablation of heads {', '.join(ablated_heads)}** and formulate a hypothesis explaining its new, compensatory function.\n\n"
        f"--- Examples of Head {target_head}'s Compensatory Behavior ---\n"
        f"{example_prompt}"
    )

    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

async def refine_hypothesis(open_router: OpenRouter, old_hypothesis: str, feedback: str, target_head: str, ablated_heads: List[str], output_dir: str) -> Tuple[str, str]:
    """根据反馈精炼假设 (BNMH版本)"""
    ablated_heads_function = "are active at the final token, attend to previous names, and copy the name they attend to."

    system_prompt = (
        "You are a world-class AI researcher refining a hypothesis about an attention head's compensatory behavior.\n\n"
        f"**--- CRUCIAL CONTEXT (Reminder) ---**\n"
        f"You are analyzing **Head {target_head}**. This head is only active because other heads ({', '.join(ablated_heads)}) which normally {ablated_heads_function} have been **disabled**.\n\n"
        "**--- Your Task ---**\n"
        "You are given the previous hypothesis and feedback showing where it succeeded or failed to predict Head {target_head}'s attention. Your goal is to **refine the hypothesis** to better explain the observed behavior.\n"
        "- Analyze the `INCORRECT` examples in the feedback. Why did the old hypothesis fail for them?\n"
        "- Look for a common pattern in the failures. Does the head behave differently for certain sentence structures (e.g., ABBA vs. BABA)?\n"
        "- Propose a new, more accurate hypothesis that accounts for these failures while still explaining the successes.\n\n"
        "**--- Response Guidelines ---**\n"
        "1. **[REASONING]:** First, analyze the feedback. Explain *why* the old hypothesis failed, referencing specific examples.\n"
        "2. **[HYPOTHESIS]:** Then, provide the new, improved hypothesis in a single, concise paragraph."
    )

    user_prompt = (
        f"**Previous Hypothesis:**\n{old_hypothesis}\n\n"
        "**--- Performance Feedback ---**\n"
        "The previous hypothesis was tested, and here are the results. Analyze the failures to improve it.\n\n"
        f"{feedback}\n\n"
        "Based on this feedback, provide your reasoning for the refinement and the new hypothesis."
    )
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response = await open_router.generate(messages=messages, output_dir=output_dir)
    
    return extract_hypothesis_from_response(response.text)

# --- 4. 预测与评估 (继承自auto_NMH.py并改造) ---

async def underscore_important_tokens(open_router, target_head, hypothesis_text, examples, output_dir):
    """让LLM根据假设，标记出重要的token"""
    print(f"Predicting important tokens for head {target_head} based on hypothesis...")

    def validate_highlighted_output(text):
        # 验证逻辑可以保持不变
        pattern = r"Number of important tokens: (\d+)\s+(.+?: .+?)\s+indirect object:"
        blocks = re.findall(pattern, text)
        if not blocks: return False
        for expected_str, sentence in blocks:
            expected = int(expected_str)
            actual = len(re.findall(r"<<[^<>]+?>>", sentence))
            if expected != actual:
                print(f"Mismatch found: expected {expected}, got {actual} in: {sentence}")
                return False
        return True
    
    def extract_token_sentences(text):
        return "\n".join(re.findall(r"^.+?: .+?{{.+?}}", text, re.MULTILINE))
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a meticulous AI researcher working on interpreting the function of a specific attention head in GPT2-small. "
                "You are given:\n"
                "- A hypothesis describing what this attention head might be doing in the Indirect Object Identification (IOI) task.\n"
                "- A set of example sentences with indirect objects annotated using {{...}}.\n"
                "- For each sentence, a line explicitly states the required number of important tokens to highlight, written as 'Number of important tokens: N'.\n\n"
                "Your job is to:\n"
                "1. Predict exactly N important tokens in each sentence, based on the hypothesis. This importance is from the perspective of the token right before the indirect object.\n"
                "2. Highlight the predicted tokens by enclosing them in double angle brackets like <<this>>.\n"
                "3. Ensure the number of highlighted tokens (<< >>) matches **exactly** the number given for that sentence.\n"
                "4. Treat each occurrence of a word as a unique token (e.g., 'David' at the beginning and 'David' at the end are different).\n"
                "5. The tokens you highlight must be **before** the indirect object in {{}}.\n\n"
                "MUST FOLLOW:\n"
                "- Your response must strictly follow this structure for each example:\n"
                "  Line 1: Number of important tokens: N (unchanged)\n"
                "  Line 2: sentence_key: the sentence with exactly N <<highlighted>> tokens\n"
                "  Line 3: indirect object: actual indirect object (unchanged)\n"
                "- Do not insert any extra lines. Do not omit or add to the sentence. Do not guess metadata."
            )
        },
        {
            "role": "user",
            "content": (
                f"You are given a hypothesis about attention head {target_head} and several examples. "
                "For each example, identify exactly the number of important tokens stated, based on the hypothesis. "
                "Highlight those tokens with << >> and strictly follow the output format.\n"
                f"\n{hypothesis_text}"
                f"\n{examples}"
            )
        }
    ]
    
    for attempt in range(5): # 减少重试次数以加快调试
        highlighted = await open_router.generate(messages=messages, output_dir=output_dir)
        output = highlighted.text.strip()
        print(f"Attempt {attempt + 1}: Checking highlighted token counts...")
        if validate_highlighted_output(output):
            print("All token counts are correct.")
            highlighted_sentences = extract_token_sentences(output)
            print("Highlighted sentences:", highlighted_sentences)
            return highlighted_sentences
        else:
            print("Mismatch detected. Retrying...\n")

    raise ValueError("Failed to generate valid highlighted tokens after multiple retries.")

def convert_prediction_to_json(predict_attention_text: str, real_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将LLM的预测文本转换为结构化的JSON，并添加后缀"""
    results = []
    lines = [line.strip() for line in predict_attention_text.split("\n") if line.strip()]
    
    real_samples_map = {f"{i+1}_test": sample for i, sample in enumerate(real_samples)}

    for line in lines:
        if ":" not in line: continue
        key, sentence_with_markers = line.split(":", 1)
        key = key.strip()

        if key not in real_samples_map: continue
        
        real_sample = real_samples_map[key]
        original_sentence = real_sample["sentence_text"]
        
        # 使用正则表达式从标记句子中提取出被高亮的词
        predicted_words_no_suffix = re.findall(r"<<(.*?)>>", sentence_with_markers)
        
        # 为提取出的词添加后缀
        words, suffixed_map = _get_suffixed_word_map(original_sentence)
        
        predicted_tokens_with_suffix = []
        temp_predicted = predicted_words_no_suffix.copy()
        
        for i, word in enumerate(words):
            if temp_predicted and word == temp_predicted[0]:
                predicted_tokens_with_suffix.append(suffixed_map[i])
                temp_predicted.pop(0)

        results.append({
            "sample_id": real_sample["sample_id"],
            "key": key,
            "predicted_tokens": predicted_tokens_with_suffix,
        })
    return results

def evaluate_attention_f1(predictions: List[Dict[str, Any]], ground_truth_samples: List[Dict[str, Any]]) -> Tuple[float, str]:
    """评估预测的F1分数并生成反馈"""
    gt_map = {sample["sample_id"]: sample for sample in ground_truth_samples}
    
    total_f1 = 0
    feedback_lines = []
    valid_comparisons = 0

    for pred in predictions:
        sample_id = pred["sample_id"]
        if sample_id not in gt_map:
            continue
        
        gt_sample = gt_map[sample_id]
        pred_tokens = set(pred.get("predicted_tokens", []))
        
        # 从真实数据中提取IO和S token作为ground truth
        analysis = gt_sample.get("attention_analysis", {})
        io_att = analysis.get("io_attention")
        s_att = analysis.get("s_attention")
        
        real_tokens = set()
        if io_att:
            # 需要为真实token也添加后缀
            _, suffixed_map = _get_suffixed_word_map(gt_sample["sentence_text"])
            real_tokens.add(suffixed_map.get(io_att["position"]))
        if s_att:
            _, suffixed_map = _get_suffixed_word_map(gt_sample["sentence_text"])
            real_tokens.add(suffixed_map.get(s_att["position"]))
        
        real_tokens.discard(None) # 移除可能产生的None

        intersection = len(pred_tokens.intersection(real_tokens))
        precision = intersection / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = intersection / len(real_tokens) if len(real_tokens) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        total_f1 += f1
        valid_comparisons += 1

        # 生成反馈
        feedback_lines.append(f"--- Sample {sample_id} ({'CORRECT' if f1 > 0.99 else 'INCORRECT'}) ---")
        feedback_lines.append(f"Sentence: {gt_sample['sentence_text']}")
        feedback_lines.append(f"Hypothesis Predicted: {sorted(list(pred_tokens))}")
        feedback_lines.append(f"Real Attention:       {sorted(list(real_tokens))}")
        feedback_lines.append("")

    avg_f1 = total_f1 / valid_comparisons if valid_comparisons > 0 else 0
    feedback_text = "\n".join(feedback_lines)
    
    print(f"Attention F1 Score: {avg_f1:.2f}")
    return avg_f1, feedback_text

# --- 5. 主程序 ---

async def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 Starting automated hypothesis generation for Backup Head {args.target_head}...")
    
    # 初始化
    open_router = initialize_openrouter()
    all_samples = load_attention_analysis_data(args.input_file)
    if not all_samples:
        return
        
    ablated_heads = all_samples[0].get("heads_intervened", ["9.6", "9.9", "10.0"])

    # 1. 生成初始假设
    print("\n--- 📝 Generating Initial Hypothesis ---")
    initial_samples = random.sample(all_samples, 5)
    # 为初始prompt格式化样本
    example_prompt_lines = []
    for sample in initial_samples:
        analysis = sample.get("attention_analysis", {})
        io_token = sample.get("context_info", {}).get("io_token", "N/A")
        token_strs = [t["token_text"] for t in analysis.get("all_tokens", [])]
        
        highlighted_sentence = []
        for i, token in enumerate(token_strs):
            is_io = analysis.get("io_attention") and i == analysis["io_attention"]["position"]
            is_s = analysis.get("s_attention") and i == analysis["s_attention"]["position"]
            if is_io or is_s:
                highlighted_sentence.append(f"<<{token.strip()}>>")
            else:
                highlighted_sentence.append(token)
        
        example_prompt_lines.append(f"Example: {' '.join(highlighted_sentence)} {{{{{io_token}}}}}")
    example_prompt = "\n".join(example_prompt_lines)

    initial_prompt_messages = get_initial_hypothesis_prompt(args.target_head, ablated_heads, example_prompt)
    
    response = await open_router.generate(messages=initial_prompt_messages, output_dir=args.output_dir)
    reasoning, hypothesis = extract_hypothesis_from_response(response.text)
    
    print(f"Initial Reasoning: {reasoning}")
    print(f"Initial Hypothesis: {hypothesis}")

    results_log = []

    # 2. 迭代精炼循环
    for i in range(1, args.rounds + 1):
        print(f"\n--- 🔄 Iteration {i}/{args.rounds} ---")
        
        # a. 采样
        batch_samples = random_sample_from_analysis_data(all_samples, args.batch_size)
        print(f"Sampled {len(batch_samples)} sentences for this round.")
        
        # b. 格式化用于预测
        examples_for_prediction = format_examples_for_prediction(batch_samples)
        
        # c. 预测重要token
        highlighted_text = await underscore_important_tokens(open_router, args.target_head, hypothesis, examples_for_prediction, args.output_dir)
        
        # d. 将预测转换为JSON
        predictions_json = convert_prediction_to_json(highlighted_text, batch_samples)
        
        # e. 评估
        f1_score, feedback = evaluate_attention_f1(predictions_json, batch_samples)
        
        # f. 记录日志
        results_log.append({
            "iteration": i,
            "hypothesis": hypothesis,
            "f1_score": f1_score,
            "feedback_generated": feedback,
            "predictions": predictions_json
        })

        # g. 检查终止条件
        if f1_score > 0.95:
            print("✅ Hypothesis converged with F1 score > 0.95. Stopping.")
            break
        if i == args.rounds:
            print("🏁 Reached max iterations.")
            break

        # h. 精炼
        print("Refining hypothesis based on feedback...")
        reasoning, hypothesis = await refine_hypothesis(open_router, hypothesis, feedback, args.target_head, ablated_heads, args.output_dir)
        print(f"New Reasoning: {reasoning}")
        print(f"Refined Hypothesis: {hypothesis}")

    # 3. 保存最终结果
    final_output_path = os.path.join(args.output_dir, f"backup_hypothesis_results_{args.target_head.replace('.', '_')}.json")
    with open(final_output_path, "w") as f:
        json.dump(results_log, f, indent=2)
        
    print(f"\n\n🎉 Experiment complete! Final results saved to {final_output_path}")
    print(f"Final Hypothesis for Head {args.target_head}: {hypothesis}")


if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    asyncio.run(main())