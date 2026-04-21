import asyncio
import sys
import json
import os
import numpy as np
import re
import math
import random
from scipy.stats import kendalltau
from itertools import groupby
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt_tab')
from collections import defaultdict
import argparse
sys.path.append("/home/wangziran/eap_auto/")
# sys.path.append("/data63/private/chensiyuan/EAP-IG/")
from api import OpenRouter
from attention_score_by_head import run

async def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Run OpenRouter with specified layer and head.")
    parser.add_argument("--layer", type=int, required=True, help="The layer number to analyze.")
    parser.add_argument("--head", type=int, required=True, help="The head number to analyze.")
    parser.add_argument("--rounds", type=int, required=True, help="The rounds number for the analysis.")
    parser.add_argument("--typename", type=str, required=True, help="The typename of head.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory path.")  # 添加这行
    return parser.parse_args()

def load_examples(json_path):
      """从JSON文件加载示例句子和激活值"""
      example_sentence, example_activations = [], []
      with open(json_path, "r") as f:
            data = json.load(f)
      example_sentence = [item["example_sentence"] for item in data]
      example_activations = [item["example_activations"] for item in data]
      example_indirect_object = [item["indirect_object"] for item in data]
      return example_sentence, example_activations, example_indirect_object

def initialize_openrouter(model:str):
      """初始化OpenRouter API"""
      if model == "deepseek-v3":
            model = "deepseek-v3"
      api_key = "sk-t9ooUIrk73Zg4s72dFCf2QAYWrNsobTW1gT8P7AG7m1r4Wbd"
      # api_key = "sk-tWWLjYE4fua6zjAY024dE493C9874e2d8e1a877f65A51488"
      # api_key = "sk-MjSyxJuoVSlripXy2933FaBaEaBb4fC1A4B564DfB699B8C2"
      return OpenRouter(model=model, api_key=api_key)

def normalize_token(token):
    return token.strip().lower()


async def generate_hypothesis(open_router, layer, head, explanation, example_sentence, example_activations, example_indirect_object, output_dir):
      """生成假设"""
      user_content = "\n".join(
            f"{sentence}{activations}{io}" for sentence, activations, io in zip(example_sentence, example_activations, example_indirect_object)
      )
      print(f"Generating hypothesis for layer {layer}, head {head}...")
      messages = [
            {
                  "role": "system",
                  "content": (
                  "You are a meticulous AI researcher conducting an important investigation into patterns found in language. "
                  "The text is based on the Indirect Object Identification task, where the model is asked to identify the indirect object according to the first half sentence and deduce the next token of the uncompleted sentence to be the indirect object. "
                  "Your task is to analyze text and provide a rational hypothesis that thoroughly explains the function of the given attention head in this specific task.\n\n"
                  
                  "Additionally, you will receive a list of examples in which the indirect object has been hypothesized and inserted at the end of the sentence, marked using double curly braces (e.g., {{John}}). The same predicted indirect object will also be displayed separately after the sentence as a reference.\n\n"

                  "You are also given attention head's influence to the logit difference of the model output or its influence to other attention heads known to perform functions relevant to the task.\n\n"

                  "Guidelines:\n"
                  "- You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. These words have high attention score from the token in front of the indirect object quoted with {{ }} of the sentence.\n"
                  "- If a sequence of consecutive tokens is important, it is fully enclosed within the delimiters <<like this>>.\n"
                  "- Each example is followed by a list of important tokens and their scores (between 0 and 1) after 'Activations:', where higher values indicate stronger influence. The total sum of scores will not exceed 1.\n"
                  "- Your job is not to judge or validate the inserted indirect object, but to hypothesize what this attention head is doing based on the pattern of important tokens and insertions.\n"
                  "- Do not focus on listing examples or token scores. Instead, synthesize a clear, coherent hypothesis that explains the attention head's behavior.\n"
                  "Your hypothesis must also focus on the relation between the real important tokens, its score and the indirect object you suppose to deduce in IOI task. Since your goal is to analyze the function of specific head which may attend to specific tokens and affect the performance of the model in IOI task. \n\n"
                  "- Do not mention the symbols << >> or {{ }} in your final hypothesis.\n"
                  "- The final paragraph must be your hypothesis, beginning with [HYPOTHESIS]:\n"
                  "- The [HYPOTHESIS] should be one single paragraph, clearly and thoroughly articulating the head’s behavior.\n\n"
                  "- The hypothesis should describe the dominant functional behavior using precise linguistic, semantic, or structural terminology. Avoid overly abstract or generic phrasing like “tracks entities” or “maintains context”."
                  "- Analyze the examples by grouping them into input categories (e.g., sentence structures, entity numbers and roles, or context types) and explain how the head behaves differently across these types. Base your final hypothesis on this classification-aware reasoning."
                  "- Do not include tokens, examples, scores, or error context in the [HYPOTHESIS] paragraph.\n"
                  "- The hypothesis should sound like a standalone description of the head's role in the model with no concessions or negations.\n"
                  "When analyzing attention heads, please consider their contribution to the model's final prediction. While attention patterns show what a head focuses on, "
                  "logit contributions reveal how much this focus actually impacts the final prediction.\n\n"
                
                  "When forming your hypothesis, consider:\n"
                  "- How the head's attention pattern affects the model's ability to predict the indirect object\n"
                  "- In which types of sentence structures this head contributes most\n"
                  "- What relationship exists between attention patterns and logit contributions\n"
                  "- Whether the head is more important in specific contexts\n\n"
                  
                  "Example:\n"
                  "Input: Attention head 9 in layer 9 has a high influence to the logit difference of the model output.\n"
                  "Example 1: When <<Victoria>> and Jane got a snack at the store , Jane decided to give it to {{Victoria}}\n"
                  "Activations: (\"Victoria\", 0.72)\n"
                  "Indirect object: Victoria\n"
                  "Example 2: Then , Tom and <<James>> had a lot of fun at the park . Tom gave a present to {{James}}\n"
                  "Activations: (\"James\", 0.95)\n"
                  "Indirect object: James\n"
                  "Example 3: While Annie and <<Tony>> were working at the bank , Annie gave a hug to {{Tony}}\n"
                  "Activations: (\"Tony\", 0.83)\n"
                  "Indirect object: Tony\n"
                  "Example 4: <<Then>> , <<Felix>> and Sam had a long argument . Afterwards Sam said to {{Felix}}\n"
                  "Activations: (\"Then\", 0.19), (\"Felix\", 0.76)\n"
                  "Indirect object: Felix\n\n"
                  "Answer:\n"
                  "Begin by carefully analyzing the examples provided. Look for consistent patterns in what tokens the attention head attends to, and how these relate to the inserted indirect object in each case. Reflect on what types of entities, positions in the sentence, or semantic roles are being emphasized. Consider whether the attention is targeting recipients, agents, or contextual markers, and whether it consistently selects entities based on discourse order, syntactic position, or event structure. Use this reasoning to infer the general behavior of the attention head. \n\n"
                  "Then, write a single, concise paragraph starting with [HYPOTHESIS]: that clearly states what this attention head appears to do in the IOI task. Do not include token scores, example references, or symbolic markers in your hypothesis.\n"
                  "[HYPOTHESIS]: Your final hypothesis should now be written as a single paragraph here, summarizing all your insights and stating clearly what function the attention head likely performs in this task.\n"
                  "Your [HYPOTHESIS] paragraph must avoid vague or overly abstract language. Do not use phrases such as 'contextual cues', 'semantic relevance', 'discerning nuanced roles', or similar unspecific statements. Instead, use precise and interpretable descriptions of what the head attends to — such as the role and position of specific token and how it connects to other token."
                  "Your hypothesis should be directly testable or falsifiable through model behavior, not theoretical or ambiguous. Make it as clear and concise as possible so that another researcher could validate or reject it based on measurable attention behavior."
                  )
            },
            {
                  "role": "user",
                  "content": (
                  f"\n{explanation}"
                  f"\n{user_content}"
                  "Please write a full response including a thorough reasoning(do not copy the instruction) and a final [HYPOTHESIS] paragraph."
                  ),
            },
      ]
      
      hypothesis = await open_router.generate(messages=messages, output_dir=output_dir)
      print("hypothesis:", hypothesis.text,"\n")
      return hypothesis.text

def get_ground_truth_from_precomputed(precomputed_data, sender_head, top_k=3):
    """
    从预计算数据中，为给定的sender_head计算到所有下游receiver_head的平均因果效应，
    并提取Top-K注意力变化。
    """
    sender_key = f"{sender_head[0]}.{sender_head[1]}"
    
    try:
        # 获取该sender影响的所有receiver
        receivers_data = precomputed_data[sender_key]
        if not receivers_data:
            print(f"错误：Sender {sender_key} 在预计算文件中没有对应的receiver数据。")
            return {"increase": [], "decrease": []}
    except KeyError:
        print(f"错误：在预计算文件中未找到Sender {sender_key}。")
        return {"increase": [], "decrease": []}

    all_diff_matrices = []
    str_tokens = None

    # 遍历所有receiver，收集它们的diff_matrix
    for receiver_key, path_data in receivers_data.items():
        if "diff_matrix" in path_data and "tokens" in path_data:
            all_diff_matrices.append(np.array(path_data["diff_matrix"]))
            if str_tokens is None: # 所有句子的tokens都一样，取第一个即可
                str_tokens = path_data["tokens"]
    
    if not all_diff_matrices or str_tokens is None:
        print(f"错误：未能从Sender {sender_key}的下游头中收集到有效的diff_matrix。")
        return {"increase": [], "decrease": []}

    # --- 核心修改：对所有收集到的diff_matrix进行平均 ---
    print(f"为 Sender {sender_key} 平均 {len(all_diff_matrices)} 个下游头的差值矩阵...")
    avg_diff_matrix = np.mean(all_diff_matrices, axis=0)

    # 我们只关心最后一个token（查询位置）对所有其他token（键位置）的注意力变化
    last_token_diff_vector = avg_diff_matrix[-1]
    
    # 将token与它们对应的注意力变化分数配对
    token_changes = list(zip(str_tokens, last_token_diff_vector))
    
    # 按分数降序排序，以找到变化最大的token
    token_changes.sort(key=lambda x: x[1], reverse=True)
    
    # 提取增加最多的Top-K
    top_increase = [{"token": token, "score": float(score)} for token, score in token_changes[:top_k]]
    
    # 提取减少最多的Top-K (列表末尾)
    top_decrease = [{"token": token, "score": float(score)} for token, score in token_changes[-top_k:]]
    
    return {"increase": top_increase, "decrease": top_decrease}

async def predict_top_k_attention_changes(open_router, hypothesis_text, sentence_info, sender_head, receiver_description, top_k, output_dir):
    """
    让AI根据假设，预测Top-K的注意力变化（用<< >>和[[ ]]标记）。
    """
    print(f"Predicting Top-K attention changes...")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a meticulous AI researcher analyzing causal effects in transformer circuits.\n"
                f"You are given a hypothesis about how a 'sender' head ({sender_head}) influences a set of '{receiver_description}'.\n"
                "Your task is to predict, for each sentence, which tokens the receiver head will pay MORE or LESS attention to when the sender's influence is disrupted.\n\n"
                f"For each sentence, mark the Top {top_k} tokens that will receive MORE attention with << >>, and the Top {top_k} tokens that will receive LESS attention with [[ ]].\n"
                "Example:\n"
                "Sentence: Tom gave a book to Mary.\n"
                "Marked: Tom gave a book to <<Mary>>.\n"
                "If a token should be marked as LESS attention, use [[token]].\n"
                "If both, you can nest or list both.\n"
                "Output one marked sentence per line, prefixed by the sentence key (e.g., 1_test: ...)."
            )
        },
        {
            "role": "user",
            "content": (
                f"Hypothesis: {hypothesis_text}\n\n"
                f"Sentences:\n{sentence_info}\n\n"
                f"Please output the marked sentences as described."
            )
        }
    ]
    response = await open_router.generate(messages=messages, output_dir=output_dir)
    ai_attention_pred_path = os.path.join(output_dir, "ai_attention_pred_raw.txt")
    with open(ai_attention_pred_path, "a") as f:
        f.write(response.text + "\n")

    # 用正则提取
    result = {}
    for line in response.text.strip().split("\n"):
        if ":" not in line:
            continue
        key, marked_sentence = line.split(":", 1)
        # 提取 << >> 和 [[ ]] 标记的token
        increase = re.findall(r"<<(.*?)>>", marked_sentence)
        decrease = re.findall(r"\[\[(.*?)\]\]", marked_sentence)
        result[key.strip()] = {
            "increase": [tok.strip() for tok in increase][:top_k],
            "decrease": [tok.strip() for tok in decrease][:top_k]
        }
    return result
  
def extract_hypothesis_text(hypothesis_text):
      """从响应中提取假设文本，并返回匹配部分之前的字符串"""
      match = re.search(r"(.*)\[HYPOTHESIS\]:\s*(.*)", hypothesis_text, re.DOTALL)
      if match:
            before_hypothesis = match.group(1).strip()  # 匹配到 [HYPOTHESIS]: 之前的内容
            hypothesis = match.group(2).strip()        # 匹配到 [HYPOTHESIS]: 之后的内容
            return before_hypothesis, hypothesis
      print("No hypothesis found in the response.")
      return None, None

def extract_example_sentences(example_message_text):
      """从响应中提取示例句子"""
      match = re.search(r"{(.*?)}", example_message_text, re.DOTALL)
      if match:
            return json.loads(match.group(0).strip())
      print("No example sentences found in the response.")
      return {}


def calculate_sample_distribution(token_num_freq, batch_size):
      """
      根据 token_num 的频率计算采样分布，确保总和等于 batch_size。
      """
      # 1. 归一化频率
      total_freq = sum(token_num_freq.values())
      normalized_freq = {k: v / total_freq for k, v in token_num_freq.items()}

      # 2. 按比例分配采样数量
      initial_distribution = {k: v * batch_size for k, v in normalized_freq.items()}

      # 3. 四舍五入为整数
      rounded_distribution = {k: round(v) for k, v in initial_distribution.items()}

      # 4. 计算调整后的总和
      total_samples = sum(rounded_distribution.values())

      # 5. 调整分布以确保总和等于 batch_size
      if total_samples != batch_size:
            # 按误差排序（从大到小）
            sorted_token_nums = sorted(normalized_freq.keys(), key=lambda k: initial_distribution[k] - rounded_distribution[k], reverse=True)
            diff = batch_size - total_samples
            
            for token_num in sorted_token_nums:
                  if diff == 0:
                        break
                  # 调整采样数量
                  rounded_distribution[token_num] += 1 if diff > 0 else -1
                  diff += -1 if diff > 0 else 1

      return rounded_distribution

def random_sample_sentences(dataset_sentence_path, output_sentence_path, batch_size, iteration):
      
      with open(dataset_sentence_path, "r") as f:
            try:
                  data = json.load(f)  # 读取整个文件作为 JSON 数组
            except json.JSONDecodeError as e:
                  print(f"Error decoding JSONL file {dataset_sentence_path}: {e}")
                  return
      ## 计算出每条中"number_of_important_tokens": x中x出现的频次与频率
      token_num_counter = defaultdict(int)
      for item in data:
            token_num = item.get("number_of_important_tokens")
            if token_num is not None:
                  token_num_counter[token_num] += 1
      total_count = sum(token_num_counter.values())
      all_all_token_num_freq = {k: v / total_count for k, v in token_num_counter.items()}
      ## 如果freq小于等于0.05的就不考虑，其余则按频率采样batch_size句
      token_num_freq = {k: v for k, v in all_all_token_num_freq.items() if v > 0.05}
      ## 计算出每个token_num需要采样的数量且要保证相加等于batch_size
      sample_distribution = calculate_sample_distribution(token_num_freq, batch_size)

      # 按分布随机采样句子
      sampled_data = []
      for token_num, sample_count in sample_distribution.items():
            filtered_data = [item for item in data if item.get("number_of_important_tokens") == token_num]
            if len(filtered_data) >= sample_count:
                  sampled_data.extend(random.sample(filtered_data, sample_count))  # 随机采样
            else:
                  sampled_data.extend(filtered_data)  # 如果数量不足，直接全部添加

      # 按格式写入采样结果，同时构建返回的字典
      # 修改 random_sample_sentences 函数的返回部分
      result_dict = {}
      for i, item in enumerate(sampled_data, start=1):
            key = f"{i}_test"
            original_sentence = item.get("original_sentence", "")
            indirect_object = item.get("indirect_object", "")  # 添加间接对象
            result_dict[key] = {"sentence": original_sentence, "io": indirect_object}  # 确保格式正确
      write_dict = {
            f"itertaion_{iteration}": result_dict
      }
      with open(output_sentence_path, "a") as f:
            f.write(json.dumps(write_dict, ensure_ascii=False) + "\n")
      print(f"Filtered sentences saved to {output_sentence_path}")
      return result_dict

def calculate_logit_contribution(layer, head, example_sentences, 
                               precomputed_file="/home/wangziran/eap_auto/results/ioi/path_patching/heads_direct_effect_on_logit_difference.json"):
    """从预计算文件中获取注意力头对logit差异的贡献（保留原始符号）"""
    
    try:
        with open(precomputed_file, 'r') as f:
            head_contributions = json.load(f)
    except FileNotFoundError:
        print(f"预计算文件 {precomputed_file} 不存在，使用默认值")
        return {key: 0.0 for key in example_sentences.keys()}
    
    head_key = f"{layer}.{head}"
    # 直接使用原始的、带有符号的贡献度值
    base_contribution = head_contributions.get(head_key, 0.0)
    
    contributions = {key: base_contribution for key in example_sentences.keys()}
    
    print(f"使用预计算贡献度 - 头 {layer}.{head}: 原始贡献度 = {base_contribution:.4f}")
    return contributions

async def compare_and_refine_hypothesis(open_router, explanation, old_hypothesis, 
                                      attention_feedback_text, logit_comparison_text,
                                      sender_head, receiver_description, output_dir):
      """Compare and refine hypothesis based on attention patterns and logit contributions"""
      print(f"Comparing and refining hypothesis based on attention patterns and logit contributions...")
      
      receiver_function_description = "The downstream heads are known **Name Mover Heads**. Their primary function is to attend to the correct indirect object name at the end of the sentence to copy it to the output."
      
      messages = [
            {
                  "role": "system",
                  "content": (
                  "You are a meticulous AI researcher tasked with deducing the fundamental mechanism of an attention head in GPT2-small.\n\n"
                  f"**Experimental Setup:**\n"
                  f"- We are studying a **Sender Head ({sender_head})**.\n"
                  f"- To probe its function, we are observing its causal effect on a known set of downstream heads: **{receiver_description}**.\n"
                  f"- **Downstream Head Context:** {receiver_function_description}\n\n"
                  "**Your Goal:** Formulate a precise hypothesis about the **Sender Head's intrinsic function**.\n\n"
                  "You will receive feedback on how a previous hypothesis failed to predict the sender's effects on the downstream heads. This feedback is your primary evidence. Use it to refine your understanding of what the sender head *itself* is doing. A good hypothesis should explain the sender's core behavior (e.g., what kind of information it processes, what features it detects) in a way that naturally accounts for the observed downstream effects.\n\n"
                  "**Think Step-by-Step:**\n"
                  "1. **Analyze the Evidence:** The feedback reveals the sender's true causal impact. What does this tell you about the sender's role? Did it inhibit something unexpected? Did it promote something surprising?\n"
                  "2. **Deduce the Mechanism:** Based on this evidence, what is the most likely *internal mechanism* of the sender head? Is it an 'early stopping' head? A 'subject/entity detector'? A 'syntactic relation' head? Be specific.\n"
                  "3. **Formulate a New Hypothesis:** Create a new, more precise hypothesis for the sender head that describes this core mechanism. The hypothesis should be about the sender head itself, but be consistent with the evidence you've seen.\n\n"
                  "**Response Structure:**\n"
                  "1. **Reasoning:** Briefly explain your analysis of the evidence and how it led you to a new understanding of the sender head's mechanism.\n"
                  "2. **[HYPOTHESIS]:** Provide a single, clean paragraph starting with `[HYPOTHESIS]:` that describes your new, refined hypothesis for the sender head's fundamental function."
                  "Please output your reasoning as a paragraph, and then a single paragraph starting with [HYPOTHESIS]: ... Do NOT use markdown code blocks or JSON format."
                  )
            },
            {
                "role": "user",
                "content": (
                    f"Causal Path: Sender {sender_head} -> Receivers ({receiver_description})\n"
                    f"Explanation of Sender's Assumed Role: {explanation}\n\n"
                    "Old Hypothesis:\n"
                    f"{old_hypothesis}\n\n"
                    "--- FEEDBACK ON PREDICTION FAILURES ---\n\n"
                    "Feedback on Attention Prediction Failures:\n"
                    f"{attention_feedback_text}\n\n"
                    "Feedback on Logit Contribution Prediction Failures:\n"
                    f"{logit_comparison_text}\n\n"
                    "--- END OF FEEDBACK ---\n\n"
                    "Based on the old hypothesis and the feedback on its failures, propose a revised, more accurate hypothesis for the sender head's function."
                    "Please output your reasoning as a paragraph, and then a single paragraph starting with [HYPOTHESIS]: ... Do NOT use markdown code blocks or JSON format."
                )
            }
      ]
      
      new_hypothesis = await open_router.generate(messages=messages, output_dir=output_dir)
      print("new_hypothesis:", new_hypothesis.text,"\n")
      return new_hypothesis.text

async def predict_logit_contribution(open_router, layer, head, hypothesis_text, example_sentences, output_dir):
    """Predict the attention head's contribution to logit differences"""
    print(f"Predicting logit contribution for layer {layer}, head {head}...")
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise AI researcher analyzing the behavior of a specific attention head in the GPT2-small model during the Indirect Object Identification (IOI) task.\n\n"
                "Your task is to:\n"
                "1. Based on the given hypothesis, predict whether the contribution of this head to the final correct answer is 'Positive' or 'Negative' for each sentence.\n"
                "2. A 'Positive' contribution means the head's function is **contributory** and AIDS the model in correctly identifying the indirect object.\n"
                "3. A 'Negative' contribution means the head's function is **inhibitory** and HINDERS the model from identifying the correct indirect object.\n\n"
                
                "Output format:\n"
                "One line per sentence, formatted as: key: Positive/Negative\n"
                "For example:\n"
                "1_test: Positive\n"
                "2_test: Negative\n"
                
                "Important notes:\n"
                "- Only output sentence keys and the word 'Positive' or 'Negative'.\n"
                "- Do not provide explanations or scores."
            )
        },
        {
            "role": "user", 
            "content": (
                f"Based on the following hypothesis for layer {layer}, head {head}, predict the head's contribution sign for each sentence:\n\n"
                f"Hypothesis:\n{hypothesis_text}\n\n"
                f"Sentences:\n{example_sentences}\n\n"
                f"For each sentence, predict 'Positive' or 'Negative' indicating the head's contribution."
            )
        }
    ]
    
    predicted = await open_router.generate(messages=messages, output_dir=output_dir)
    return predicted.text.strip()


def compare_logit_contributions(predicted_contributions_text, real_contributions):
    """比较预测的（正/负）和真实的logit贡献度，计算准确率"""
    predicted_signs = {}
    real_signs = {}
    
    # 解析预测的文本
    for line in predicted_contributions_text.strip().split("\n"):
        if ":" in line:
            key, value_str = line.split(":", 1)
            # 统一转换为小写以进行比较
            predicted_signs[key.strip()] = value_str.strip().lower()
            
    # 计算评估指标
    correct_count = 0
    total_count = 0
    
    for key, real_value in real_contributions.items():
        if key in predicted_signs:
            # Path Patching中，负值表示正面贡献（帮助模型），正值表示负面贡献（伤害模型）
            # 这是该领域的一个惯例
            actual_sign = "positive" if real_value <= 0 else "negative"
            real_signs[key] = actual_sign
            
            if predicted_signs[key] == actual_sign:
                correct_count += 1
            total_count += 1
            
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    return {
        "accuracy": accuracy,
        "predicted": predicted_signs,
        "real": real_signs
    }
    
    
async def main():
    # --- 1. 初始化和参数解析 ---
    batch_size = 10
    iter_count = 10
    model_name = "gpt-4o"
    
    args = await parse_arguments()
    layer = args.layer
    head = args.head
    rounds = args.rounds
    output_dir = args.output_dir
    
    # --- 2. 定义因果路径和加载预计算数据 ---
    # sender_head 由命令行参数决定
    sender_head = (layer, head)
    top_k = 1

    # 加载预计算的因果效应数据
    causal_effects_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ioi", "path_patching", "causal_attention_effects.json")
    print(f"正在从 {causal_effects_path} 加载预计算的因果效应...")
    try:
        with open(causal_effects_path, "r") as f:
            precomputed_causal_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法加载或解析预计算的因果效应文件: {e}")
        print("请确保您已成功运行 precompute_causal_effects.py。")
        return

    # 加载sender head对logit的直接贡献数据 (这个逻辑保持不变)
    direct_logit_effect_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ioi", "path_patching", "heads_direct_effect_on_logit_difference.json")

    print("预计算数据加载成功。")

    # --- 3. 初始化AI和生成初始假设 ---
    # 这部分逻辑与之前类似，但请注意，现在explanation应该更侧重于描述sender head的因果角色
    open_router = initialize_openrouter(model=model_name)
    print(f"使用OpenRouter模型: {model_name}")

    dataset_sentence_path = os.path.join(output_dir, "raw_model_prompt_attention_scores.jsonl")
    # 这个路径用于记录每一轮迭代中采样出的句子，用于调试和追溯。
    output_sentence_path = os.path.join(output_dir, "training_sentences.jsonl")
    # 假设我们有一个解释文本，描述了我们正在研究的因果路径
    # 您可以从 head_explanations.jsonl 中加载或动态生成
    explanation = f"We are analyzing the causal effect of sender head {sender_head} on a set of downstream Name Mover Heads. The sender head is believed to be an S-Inhibition head."
    
    # 初始假设生成可以保持，但现在它应该基于因果解释
    # (为了简化，我们跳过初始生成，直接使用一个示例假设)
    hypothesis_text = f"The attention head {layer}.{head} acts as an S-inhibition head. It identifies the subject of the sentence and suppresses the attention of downstream Name Mover heads towards this subject, thereby promoting attention to the correct indirect object."
    hypothesis_analysis, extracted_hypothesis = extract_hypothesis_text(f"[HYPOTHESIS]: {hypothesis_text}")
    
    results = []
    iteration = 1
    while True:
        print(f"\n--- Iteration: {iteration}/{iter_count} ---")

        # --- 4. 获取“标准答案” (现在是快速的查表操作) ---
        real_attention_changes = get_ground_truth_from_precomputed(
            precomputed_causal_data, sender_head, top_k=top_k
        )
        if not real_attention_changes.get("increase") and not real_attention_changes.get("decrease"):
            print("无法获取真实的注意力变化，跳过本轮迭代。")
            break
        print(f"本轮迭代的真实Top-K变化: {real_attention_changes}")

        # --- 5. 采样句子并让AI预测 ---
        example_sentences = random_sample_sentences(dataset_sentence_path, output_sentence_path, batch_size, iteration)
        sentence_info_for_ai = "\n".join([f"{key}: {value['sentence']}" for key, value in example_sentences.items()])
        
        predicted_changes_json = await predict_top_k_attention_changes(
            open_router, extracted_hypothesis, sentence_info_for_ai, sender_head, "downstream Name Mover Heads", top_k, output_dir
        )
        print(f"AI预测的Top-K变化: {predicted_changes_json}")

        # --- 6. 评估因果预测的准确性 ---
        total_f1_increase, total_f1_decrease = 0, 0
        feedback_details = []
        
        for key, value in example_sentences.items():
            if key in predicted_changes_json:
                pred_inc_set = set(predicted_changes_json[key].get('increase', []))
                real_inc_set = {item['token'] for item in real_attention_changes['increase']}
                
                pred_dec_set = set(predicted_changes_json[key].get('decrease', []))
                real_dec_set = {item['token'] for item in real_attention_changes['decrease']}

                # 在 F1 计算前
                pred_inc_set = set(normalize_token(t) for t in predicted_changes_json[key].get('increase', []))
                real_inc_set = set(normalize_token(item['token']) for item in real_attention_changes['increase'])
                
                # 在 F1 计算前
                pred_dec_set = set(normalize_token(t) for t in predicted_changes_json[key].get('decrease', []))
                real_dec_set = set(normalize_token(item['token']) for item in real_attention_changes['decrease'])

                # 计算F1分数
                f1_increase = (2 * len(pred_inc_set.intersection(real_inc_set))) / (len(pred_inc_set) + len(real_inc_set)) if (len(pred_inc_set) + len(real_inc_set)) > 0 else 0
                f1_decrease = (2 * len(pred_dec_set.intersection(real_dec_set))) / (len(pred_dec_set) + len(real_dec_set)) if (len(pred_dec_set) + len(real_dec_set)) > 0 else 0
                
                total_f1_increase += f1_increase
                total_f1_decrease += f1_decrease

                feedback_details.append(
                    f"For sentence '{value['sentence']}', you predicted increase in {pred_inc_set} (real: {real_inc_set}) and decrease in {pred_dec_set} (real: {real_dec_set})."
                )
        
        avg_f1_score = ((total_f1_increase / batch_size) + (total_f1_decrease / batch_size)) / 2 if batch_size > 0 else 0

        # --- 7. 评估Logit贡献预测 (逻辑保持不变) ---
        real_logit_contributions = calculate_logit_contribution(layer, head, example_sentences, precomputed_file=direct_logit_effect_path)
        formatted_sentences = "\n".join([f"{key}: {value['sentence']} {{{{{value['io']}}}}} " for key, value in example_sentences.items()])
        predicted_logit_text = await predict_logit_contribution(open_router, layer, head, extracted_hypothesis, formatted_sentences, output_dir)
        logit_comparison = compare_logit_contributions(predicted_logit_text, real_logit_contributions)

        print(f"Iteration {iteration} Scores: Causal F1={avg_f1_score:.2f}, Logit Accuracy={logit_comparison['accuracy']:.2%}")

        # --- 8. 保存本轮结果 ---
        results.append({
            "iteration": iteration,
            "hypothesis": extracted_hypothesis,
            "scores": {
                "causal_f1": avg_f1_score,
                "logit_accuracy": logit_comparison["accuracy"]
            },
            "predicted_attention_changes": predicted_changes_json,
            "real_attention_changes": real_attention_changes,
            "logit_comparison": logit_comparison,
            "hypothesis_analysis": hypothesis_analysis,
        })

        # --- 9. 检查终止条件 ---
        if avg_f1_score >= 0.8 and logit_comparison["accuracy"] >= 0.95:
            print("Hypothesis is valid and meets all criteria.")
            break

        if iteration >= iter_count:
            print(f"Max iterations ({iter_count}) reached. Stopping.")
            break

        # --- 10. 精炼假设 ---
        # 准备反馈文本
        attention_feedback_text = "\n".join(feedback_details)
        logit_feedback_text = f"Logit Prediction Accuracy: {logit_comparison['accuracy']:.2%}. Details: {logit_comparison['predicted']} vs {logit_comparison['real']}"
        
        # 调用精炼函数 (您需要确保 compare_and_refine_hypothesis 的Prompt能处理这种新反馈)
        hypothesis_text = await compare_and_refine_hypothesis(
            open_router, explanation, extracted_hypothesis, 
            attention_feedback_text, logit_feedback_text,
            sender_head, "downstream Name Mover Heads", output_dir
        )
        hypothesis_analysis, extracted_hypothesis = extract_hypothesis_text(hypothesis_text)
        ai_raw_response_path = os.path.join(output_dir, f"ai_raw_responses_{rounds}.jsonl")
        raw_record = {
            "iteration": iteration,
            "ai_response": hypothesis_text,  # AI原始返回内容
            "parsed_analysis": hypothesis_analysis,
            "parsed_hypothesis": extracted_hypothesis,
            "attention_feedback": attention_feedback_text,
            "logit_feedback": logit_feedback_text
        }
        with open(ai_raw_response_path, "a") as f:
            f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
        if not extracted_hypothesis:
            print("Refinement failed to produce a new hypothesis. Exiting.")
            break
            
        iteration += 1

    # --- 11. 最终保存 ---
    final_results_path = os.path.join(output_dir, f"causal_results_{rounds}.json")
    with open(final_results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Final results saved to {final_results_path}")

if __name__ == "__main__":
    asyncio.run(main())