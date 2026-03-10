import asyncio
import sys
import json
import os
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
sys.path.append("/data31/private/wangziran/eap_auto/")
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
            model = "deepseek-v3-250324"
      api_key = "sk-MjSyxJuoVSlripXy2933FaBaEaBb4fC1A4B564DfB699B8C2"
      return OpenRouter(model=model, api_key=api_key)

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


def run_attention_analysis(layer, head, output_dir, example_sentences, outputfile="model_prompt_attention_scores.jsonl"):
      """运行注意力分数分析"""
      print(f"Running attention score analysis for layer {layer}, head {head} with new examples...")
      run(
            layer=layer,
            head=head,
            output_dir=output_dir,
            sequence=example_sentences,
            picture_mode=False,
            outputfile=outputfile,
      )
      print(f"Attention score analysis complete for layer {layer}, head {head} with new examples! Results saved to {output_dir}/{outputfile}","\n")

def load_real_attention_pattern(raw_model_attention_score_path, example_attention_score_path, example_sentences):
      os.makedirs(os.path.dirname(example_attention_score_path), exist_ok=True)
      with open(raw_model_attention_score_path, "r") as f:
            try:
                  raw_data = json.load(f)  # 读取整个文件作为 JSON 数组
            except json.JSONDecodeError as e:
                  print(f"Error decoding JSONL file {raw_model_attention_score_path}: {e}")
                  return
      updated_data = []
      # 遍历 example_sentences 保持顺序
      # example_sentences 的格式例如：
      # {'1_test': {'sentence': 'While spending time together, Steven and Andre were at the office. Andre gave a report to', 'io': 'Steven'}, ...}
      for ex_key, ex_value in example_sentences.items():
            ex_sentence = ex_value["sentence"]
            matched_item = next((item for item in raw_data if item["original_sentence"] == ex_sentence), None)
            if matched_item:
                  matched_item["key"] = ex_key
                  updated_data.append(matched_item)
            else:
                  print(f"Warning: No match found for example sentence '{ex_sentence}' in raw data.")
                  raise ValueError(f"Example sentence '{ex_sentence}' not found in raw data.")
      with open(example_attention_score_path, "w") as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=4)
      print(f"Updated attention scores saved to {example_attention_score_path}")

def format_examples_and_tokens(example_attention_data):
      """
      提取出 example_sentences , important_token_nums 和 indirect object 转换为目标格式字符串。
      """
      formatted_output = ""
      for example in example_attention_data:
            formatted_output += "Example: " + example["key"] + ": " + example["original_sentence"] + " {{" + example["indirect_object"] + "}}\n"
            formatted_output += f"Number of important tokens: {example['number_of_important_tokens']}"
            formatted_output += f"indirect object: {example['indirect_object']}"
            formatted_output += "\n"
      return formatted_output

async def underscore_important_tokens(open_router, layer, head, hypothesis_text, examples, output_dir):
      """画出important tokens"""
      print(f"Predicting attention scores for layer {layer}, head {head}...")

      def validate_highlighted_output(text):
            """检查每条句子中 << >> 的数量是否与 'Number of important tokens' 匹配"""
            pattern = r"Number of important tokens: (\d+)\s+(.+?: .+?)\s+indirect object:"
            blocks = re.findall(pattern, text)
            print(f"Blocks found: {blocks}")
            for expected_str, sentence in blocks:
                  expected = int(expected_str)
                  actual = len(re.findall(r"<<[^<>]+?>>", sentence))
                  if expected != actual:
                        print(f"Mismatch found: expected {expected}, got {actual} in: {sentence}")
                        return False
            return True
      def extract_token_sentences(text):
            """提取所有句子行（key: sentence）"""
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
                        "1. Predict exactly N important tokens in each sentence, based on the hypothesis and your reasoning. This importance is considered in the perspective of token right in front of the indirect object that is quoted with {{}}. \n"
                        "2. Highlight the predicted tokens by enclosing them in double angle brackets like <<this>>.\n"
                        "3. Ensure the number of highlighted tokens (<< >>) matches **exactly** the number given for that sentence.\n"
                        "4. Treat each occurrence of a word as a unique token depending on its position in the sentence (e.g., 'David' at the beginning and 'David' at the end are different).\n\n"
                        "5. The token you highlight with <<>> must be **in front of the indirect object quoted with {{}}**.\n"

                        "MUST FOLLOW:\n"
                        "- Your response should follow this structure strictly for each example:\n"
                        "  Line 1: Number of important tokens: N (unchanged)\n"
                        "  Line 2: sentence_key: the sentence with exactly N <<highlighted>> tokens\n"
                        "  Line 3: indirect object: actual indirect object (unchanged)\n"
                        "- Do not insert any extra lines between examples.\n"
                        "- Double check that every sentence has **exactly N** << >> highlighted tokens.\n"
                        "- Do not highlight the indirect object that quoted with {{}} in <<>>. You should only consider important tokens in the sentence before the {{indirect object}}. \n"
                        "- Do not omit or add to the sentence.\n"
                        "- Do not guess or hallucinate additional metadata—only format what's required.\n\n"
                        "Example:\n"
                        "Input:\n"
                        "Hypothesis: [REDACTED]\n"
                        "1_test: At the meeting, Sarah and Mike discussed the project, and Sarah promised to send the updates to {{Mike}}\n"
                        "Number of important tokens: 1\n"
                        "indirect object: Mike\n"
                        "2_test: When Liam and Ava got a certificate at the workshop, Liam decided to give it to {{Ava}}\n"
                        "Number of important tokens: 2\n"
                        "indirect object: Liam, Ava\n"
                        "3_test: During the game, Alex and Rachel cheered loudly, and Rachel handed the trophy to {{Alex}}\n"
                        "Number of important tokens: 1\n"
                        "indirect object: Alex\n"
                        "4_test: In the library, Rachel and Peter were studying, and Rachel handed the book to {{Peter}}\n"
                        "Number of important tokens: 2\n"
                        "indirect object: Peter\n"
                        "5_test: While cooking dinner, Emily and John prepared the meal, and Emily served the dish to {{John}}\n"
                        "Number of important tokens: 1\n"
                        "indirect object: John\n"
                        "6_test: After the concert, David and Lisa took pictures, and David showed the album to {{David}}\n"
                        "Number of important tokens: 2\n"
                        "indirect object: David\n"
                        "7_test: Then, Dylan and Audrey had a profound exchange. Afterwards Dylan said to {{Audrey}}\n"
                        "Number of important tokens: 3\n"
                        "indirect object: Audrey\n"
                        "Answer:\n"
                        "Number of important tokens: 1\n"
                        "1_test: At the meeting, Sarah and <<Mike>> discussed the project, and Sarah promised to send the updates to {{Mike}}\n"
                        "indirect object: Mike\n"
                        "Number of important tokens: 2\n"
                        "2_test: When <<Liam>> and <<Ava>> got a certificate at the workshop, Liam decided to give it to {{Ava}}\n"
                        "indirect object: Ava\n"
                        "Number of important tokens: 1\n"
                        "3_test: During the game, Alex and <<Rachel>> cheered loudly, and Rachel handed the trophy to {{Alex}}\n"
                        "indirect object: Alex\n"
                        "Number of important tokens: 2\n"
                        "4_test: In the library, Rachel and <<Peter>> were studying, and <<Rachel>> handed the book to {{Peter}}\n"
                        "indirect object: Peter\n"
                        "Number of important tokens: 1\n"
                        "5_test: While cooking dinner, Emily and <<John>> prepared the meal, and Emily served the dish to {{John}}\n"
                        "indirect object: John\n"
                        "Number of important tokens: 2\n"
                        "6_test: After the concert, <<David>> and <<Lisa>> took pictures, and Lisa showed the album to {{David}}\n"
                        "indirect object: David\n"
                        "Number of important tokens: 3\n"
                        "7_test: Then, <<Dylan>> and <<Audrey>> had a profound exchange. Afterwards <<Dylan>> said to {{Audrey}}\n"
                        "indirect object: Audrey\n"
                  )
            },
            {
                  "role": "user",
                  "content": (
                        f"You are given a hypothesis about attention head {head} in layer {layer} and several examples from the IOI task. "
                        "For each example, please identify exactly the number of important tokens stated under 'Number of important tokens', based on the hypothesis. "
                        "Highlight those tokens with << >> and strictly follow the output format.\n"
                        f"\n{hypothesis_text}"
                        f"\n{examples}"
                  )
            }

      ]
      
      for attempt in range(30):
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

async def predict_attention_scores(open_router, layer, head, hypothesis_text, highlighted_sentences, output_dir):
      """预测注意力分数"""
      print(f"Predicting attention scores for layer {layer}, head {head}...")

      def validate_format(output_text, original_text):
            """确保每个句子中 <<token score>> 与原始 <<token>> 数量一致，且格式规范"""
            original_lines = re.findall(r'^.+?: .+?{{.+?}}$', original_text.strip(), re.MULTILINE)
            predicted_lines = re.findall(r'^.+?: .+?{{.+?}}$', output_text.strip(), re.MULTILINE)

            if len(original_lines) != len(predicted_lines):
                  print("Mismatch in number of lines.")
                  print(f"Original lines: {original_lines}")
                  print(f"Predicted lines: {predicted_lines}")
                  return False

            for original, predicted in zip(original_lines, predicted_lines):
                  original_count = len(re.findall(r"<<[^<>]+?>>", original))
                  predicted_matches = re.findall(r"<<([^<>]+?) (\d\.\d{1,4})>>", predicted)
                  predicted_count = len(predicted_matches)
                  if original_count != predicted_count:
                        print(f"Mismatch in << >> count:\nOriginal: {original_count} vs Predicted: {predicted_count}\nSentence: {predicted}")
                        return False
                  if "<< " in predicted or " >>" in predicted:
                        print("Improper spacing inside << >>")
                        return False
            return True

      messages = [
            {
                  "role": "system",
                  "content": (
                        "You are a meticulous AI researcher conducting an important investigation into attention behavior in GPT2-small.\n"
                        "You are given:\n"
                        "- A hypothesis about the function of a specific attention head in the Indirect Object Identification (IOI) task.\n"
                        "- Several example sentences in the form: key: sentence with <<highlighted tokens>> and ending in {{indirect object}}.\n\n"
                        "Your job is to:\n"
                        "1. For each token enclosed in << >>, predict an attention score from the sentence's final token(the token right in front of the indirect object that quoted with {{}}) to that token (between 0 and 1).\n"
                        "2. Replace each <<token>> with <<token score>>, where 'score' is a float like 0.65.\n"
                        "3. Do NOT modify any other parts of the sentence. Only insert scores after highlighted tokens inside << >>.\n"
                        "4. The number of << >> regions and their content must be exactly the same as in the input.\n"
                        "5. Keep the sentence key and trailing {{...}} intact.\n\n"
                        "Rules:\n"
                        "- Do not add or remove any << >>.\n"
                        "- Each << >> must contain exactly one token followed by a single space and its attention score.\n"
                        "- Do not change punctuation or spacing outside << >>.\n"
                        "- The sum of attention scores across all highlighted tokens should be less than or equal to 1 (but not necessarily normalized). Remind that the sum of attention score from that token to all tokens in front of it should be 1, so you should spare some score to those non-important tokens. \n\n"
                        "- Do not put an empty line in your response.\n"
                        "Example:\n"
                        "Input:\n"
                        "Hypothesis: [REDACTED]\n"
                        "1_test: At the meeting, Sarah and <<Mike>> discussed the project, and Sarah promised to send the updates to {{Mike}}\n"
                        "2_test: When <<Liam>> and <<Ava>> got a certificate at the workshop, Liam decided to give it to {{Ava}}\n"
                        "3_test: During the game, Alex and <<Rachel>> cheered loudly, and Rachel handed the trophy to {{Alex}}\n"
                        "4_test: In the library, Rachel and <<Peter>> were studying, and <<Rachel>> handed the book to {{Peter}}\n"
                        "5_test: While cooking dinner, Emily and <<John>> prepared the meal, and Emily served the dish to {{John}}\n"
                        "6_test: After the concert, <<David>> and <<Lisa>> took pictures, and Lisa showed the album to {{David}}\n"
                        "7_test: Then, <<Dylan>> and <<Audrey>> had a profound exchange. Afterwards <<Dylan>> said to {{Audrey}}\n"
                        "Answer:\n"
                        "1_test: At the meeting, Sarah and <<Mike 0.72>> discussed the project, and Sarah promised to send the updates to {{Mike}}\n"
                        "2_test: When <<Liam 0.13>> and <<Ava 0.71>> got a certificate at the workshop, Liam decided to give it to {{Ava}}\n"
                        "3_test: During the game, Alex and <<Rachel 0.95>> cheered loudly, and Rachel handed the trophy to {{Alex}}\n"
                        "4_test: In the library, Rachel and <<Peter 0.82>> were studying, and <<Rachel 0.11>> handed the book to {{Peter}}\n"
                        "5_test: While cooking dinner, Emily and <<John 0.83>> prepared the meal, and Emily served the dish to {{John}}\n"
                        "6_test: After the concert, <<David 0.77>> and <<Lisa 0.16>> took pictures, and Lisa showed the album to {{David}}\n"
                        "7_test: Then, <<Dylan 0.19>> and <<Audrey 0.66>> had a profound exchange. Afterwards <<Dylan 0.10>> said to {{Dylan}}\n"
                  )
            },
            {
                  "role": "user",
                  "content": (
                  f"Give sequences of the hypothesis and sentences below to predict the function of Attention head {head} in layer {layer} in Indirect Object Identification task. You should give the attention score of the token you consider important in this IOI task and given hypothesis.\n"
                  f"\n{hypothesis_text}"
                  f"\n{highlighted_sentences}"
                  ),
            },
      ]
      
      for attempt in range(30):
            predicted = await open_router.generate(messages=messages, output_dir=output_dir)
            predicted_output = predicted.text.strip()
            # 过滤掉不符合格式要求的行
            predicted_lines = re.findall(r'^.+?: .+?{{.+?}}$', predicted_output, re.MULTILINE)
            predicted_output_filtered = "\n".join(predicted_lines)
            print(f"Attempt {attempt+1}: Validating format...")
            # # 临时跳过验证            
            # if len(predicted_lines) > 0:  # 简单检查是否有输出
            #       print("Basic validation passed.")
            #       print("Predicted attention scores:", predicted_output_filtered)
            #       return predicted_output_filtered
            if validate_format(predicted_output_filtered, highlighted_sentences):
                  print("All sentences valid.")
                  print("Predicted attention scores:", predicted_output_filtered)
                  return predicted_output_filtered
            else:
                  print("Format validation failed. Retrying...\n")

      raise ValueError("Failed to generate valid attention scores after 30 attempts.")

def convert_attention_to_json(example_attention_data):
      """提取出 example_sentences , important_tokens 和 indirect object 。"""
      results = []
      for example in example_attention_data:
            # 提取句子和重要 token
            sentence = example['example_sentence']
            important_tokens = example['important_tokens']
            indirect_object = example['indirect_object']
            number_of_important_tokens = example['number_of_important_tokens']
            
            # 构建结果字典
            result = {
                  "highlighted_sentence": sentence,
                  "important_tokens": important_tokens,
                  "indirect_object": indirect_object,
            }
            results.append(result)
      return results

def convert_predict_attention_to_json(predict_attention_text, real_attention_json):
      # 遍历每个句子
      results = []
      lines = [line.strip() for line in predict_attention_text.split("\n") if line.strip()]
      ## 把每个句子从第一个出现的{分离且取出前面的部分
      lines = [line.split("{", 1)[0] for line in lines]
      for index, line in enumerate(lines):
            # 分离句子编号和内容
            key, sentence = line.split(":", 1)
            highlighted_sentence = sentence
            important_tokens = []
            io = real_attention_json[index]["indirect_object"]
            matches = re.findall(r"<<(.*?)>>", sentence)
            scores = []
            for match in matches:
                  # 分离 token 和 score
                  parts = match.rsplit(" ", 1)
                  token = parts[0]
                  score = float(parts[1])
                  scores.append(score)
                  highlighted_sentence = highlighted_sentence.replace(f"{match}", token)
            token_counter_list = defaultdict(int)
            highlighted_sentence_tokens = word_tokenize(highlighted_sentence)
            for token in highlighted_sentence_tokens:
                  token_counter_list[token] += 1
            token_counter = defaultdict(int)
            highlight_mode = False ## 是否处于<<>>内部
            last_token = None ## 上一个 token
            highlight_index = 0 ## 当前处理到的important_tokens的编号
            # 遍历每个 token
            for token in highlighted_sentence_tokens:
                  if token == "{" or token == "}":
                        break
                  token_counter[token] += 1
                  if highlight_mode and token != "<" and token != ">":
                        ultimate_token = f"{token}_{token_counter[token]}" if token_counter_list[token] > 1 else token
                        important_tokens.append({
                              "token": ultimate_token,
                              "score": scores[highlight_index],
                        })
                        highlight_index += 1 
                  if token == "<" and last_token == "<":
                        highlight_mode = True
                  elif token == ">" and last_token == ">":
                        highlight_mode = False
                  last_token = token

            results.append({
                  "highlighted_sentence": f"{highlighted_sentence} {{{{{io}}}}}",
                  "important_tokens": important_tokens,
                  "indirect_object": io,
            })

      return results

def filter_attention_json(real_attention_json, predicted_attention_json, whole_attention_score):
      """
      比较 real_attention_json 和 predicted_attention_json 中相同位置的 important_tokens 数目，
      如果不同，则从两个列表中删除对应的条目, 同时把whole_attention_score中的相同位置的条目也删除。
      """
      filtered_real_attention = []
      filtered_predicted_attention = []
      filtered_whole_attention_score = []

      for real, pred, whole in zip(real_attention_json, predicted_attention_json, whole_attention_score):
            if len(real["important_tokens"]) == len(pred["important_tokens"]):
                  filtered_real_attention.append(real)
                  filtered_predicted_attention.append(pred)
                  filtered_whole_attention_score.append(whole)
            else:
                  print(f"Mismatch found and removed: real={real}, predicted={pred}")

            return filtered_real_attention, filtered_predicted_attention

def calculate_custom_kendall_tau(pred_tokens, real_tokens):
      """
      Calculate Kendall Tau considering token positions and values in the sequence.
      """
      n = len(pred_tokens)
      
      # 初始化 C 和 D
      C = 0  # 和谐对
      D = 0  # 不和谐对

      # 遍历所有二元对
      for i in range(n):
            for j in range(i + 1, n):
                  pred_i = pred_tokens[i]
                  pred_j = pred_tokens[j]
                  real_i = real_tokens[i]
                  real_j = real_tokens[j]
                  
                  # 检查 (i, j) 对的顺序是否一致
                  # 比较顺序一致的情况
                  if (pred_i == real_i and pred_j == real_j):
                        C += 1
                  else:
                        D += 1

      # 计算 Kendall Tau
      if C + D != 0:
            tau = (C - D) / (C + D)
      else:
            if pred_tokens == real_tokens:
                  tau = 1.0
            else:
                  tau = -1.0
      return tau

def compare_attention_score(predicted_attention, real_attention, mode, whole_attention_score=None):
      """比较预测的注意力分数和真实的注意力分数"""
      if mode == "accuracy":
            correct_count = 0
            total_tokens = 0
            for pred, real in zip(predicted_attention, real_attention):
                  pred_tokens = {token["token"] for token in pred["important_tokens"]}
                  real_tokens = {token["token"] for token in real["important_tokens"]}
                  matched_tokens = pred_tokens.intersection(real_tokens)
                  correct_count += len(matched_tokens)  # 每个匹配的 token 单独计数
                  total_tokens += len(real_tokens)  # 记录真实 token 的总数

            accuracy = correct_count / total_tokens if total_tokens > 0 else 0
            score_text = f"Score under {mode} mode: {accuracy}"
            print(score_text,"\n")
            return accuracy, score_text
      elif mode == "f1":
            correct_count = 0
            total_predicted = 0
            total_real = 0
            for pred, real in zip(predicted_attention, real_attention):
                  pred_tokens = {token["token"] for token in pred["important_tokens"]}
                  real_tokens = {token["token"] for token in real["important_tokens"]}
                  correct_count += len(pred_tokens.intersection(real_tokens))
                  total_predicted += len(pred_tokens)
                  total_real += len(real_tokens)
            precision = correct_count / total_predicted if total_predicted > 0 else 0
            recall = correct_count / total_real if total_real > 0 else 0
            print(f"correct_count: {correct_count}, total_predicted: {total_predicted}, total_real: {total_real}")
            print(f"precision: {precision}, recall: {recall}")
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            score_text = f"Score under {mode} mode: {f1_score}"
            print(score_text,"\n")
            return f1_score, score_text
      elif mode == "ndcg":
            dcg = 0
            idcg = 0
            ndcg = []
            for index,(pred, real) in enumerate(zip(predicted_attention, real_attention)):
                  # 创建 real_scores 和 pred_scores 和 attention_score 的副本
                  # 以避免修改原始数据
                  pred_scores = pred["important_tokens"][:]
                  real_scores = real["important_tokens"][:]
                  attention_score= whole_attention_score[index][:]

                  pred_scores_dict = {item['token']: item['score'] for item in pred_scores}

                  # 遍历 attention_score，确保所有 token 都在 pred_scores 中
                  expanded_pred_scores = []
                  for item in attention_score:
                        token = item['token']
                        # 如果 token 在 pred_scores 中，保留原有的 score；否则设置为 0
                        score = pred_scores_dict.get(token, 0)
                        expanded_pred_scores.append({'token': token, 'score': score})
                  pred_scores = expanded_pred_scores
                  # sort scores in descending order
                  real_scores.sort(key=lambda x: x["score"], reverse=True)
                  print(f"real_scores: {real_scores}")
                  pred_scores.sort(key=lambda x: x["score"], reverse=True)
                  print(f"pred_scores: {pred_scores}")
                  attention_score.sort(key=lambda x: x["score"], reverse=True)
                  print(f"attention_score: {attention_score}")
                  # rel_real为真实的attention score
                  rel = []
                  for real_token in real_scores:
                        rel.append({
                              real_token["token"]: real_token["score"]
                        })
                  # calculate DCG
                  for i, score in enumerate(pred_scores):
                        if score["token"] in rel[0]:
                              rel_score = rel[0][score["token"]]
                        else:
                              ## 从attention_score这个json数组里获取token=score["token"]的值
                              rel_score = next((item["score"] for item in real_scores if item["token"] == score['token']), 0)
                        dcg += rel_score / math.log2(i + 2)
                  # calculate IDCG
                  for i, score in enumerate(real_scores):
                        rel_score = score["score"]
                        idcg += rel_score / math.log2(i + 2)
                  print(f"dcg:{dcg}, idcg:{idcg}")
                  print(f"single ndcg:{dcg / idcg if idcg > 0 else 0}")
                  ndcg.append(dcg / idcg if idcg > 0 else 0)
                  dcg = 0
                  idcg = 0
            # calculate NDCG
            ndcg = sum(ndcg) / len(ndcg) if len(ndcg) > 0 else 0
            score_text = f"Score under {mode} mode: {ndcg}"
            print(score_text, "\n")
            return ndcg, score_text
      elif mode == "ndcg pre":
            dcg = 0
            idcg = 0
            ndcg = []
            for index,(pred, real) in enumerate(zip(predicted_attention, real_attention)):
                  # 创建 real_scores 和 pred_scores 的副本
                  # 以避免修改原始数据
                  pred_scores = pred["important_tokens"][:]
                  real_scores = real["important_tokens"][:]
                  attention_score= whole_attention_score[index][:]
                  # sort scores in descending order
                  real_scores.sort(key=lambda x: x["score"], reverse=True)
                  print(f"real_scores: {real_scores}")
                  pred_scores.sort(key=lambda x: x["score"], reverse=True)
                  print(f"pred_scores: {pred_scores}")
                  # rel_real为真实的attention score
                  rel = []
                  for real_token in real_scores:
                        rel.append({
                              real_token["token"]: real_token["score"]
                        })
                  # calculate DCG
                  for i, score in enumerate(pred_scores):
                        if score["token"] in rel[0]:
                              rel_score = rel[0][score["token"]]
                        else:
                              rel_score = next((item["score"] for item in real_scores if item["token"] == score['token']), 0)
                        dcg += rel_score / math.log2(i + 2)
                  # calculate IDCG
                  for i, score in enumerate(real_scores):
                        rel_score = score["score"]
                        idcg += rel_score / math.log2(i + 2)
                  print(f"dcg:{dcg}, idcg:{idcg}")
                  print(f"single ndcg:{dcg / idcg if idcg > 0 else 0}")
                  ndcg.append(dcg / idcg if idcg > 0 else 0)
                  dcg = 0
                  idcg = 0
            # calculate NDCG
            ndcg = sum(ndcg) / len(ndcg) if len(ndcg) > 0 else 0
            score_text = f"Score under {mode} mode: {ndcg}"
            print(score_text, "\n")
            return ndcg, score_text
      elif mode == "Kendall Tau":
            # 计算每对预测和真实注意力分数之间的Kendall Tau系数
            tau_values = []
            for index,(pred, real) in enumerate(zip(predicted_attention, real_attention)):
                  pred_s = pred["important_tokens"][:]
                  attention_score= whole_attention_score[index][:]
                  pred_scores_dict = {item['token']: item['score'] for item in pred_s}
                  
                  attention_score.sort(key=lambda x: x["score"], reverse=True)
                  print(f"attention_score: {attention_score}")
                  # 遍历 attention_score，确保所有 token 都在 pred_scores 中
                  expanded_pred_scores = []
                  for item in attention_score:
                        token = item['token']
                        # 如果 token 在 pred_scores 中，保留原有的 score；否则设置为 0
                        score = pred_scores_dict.get(token, 0)
                        expanded_pred_scores.append({'token': token, 'score': score})
                  pred_s = expanded_pred_scores
                  # sort scores in descending order
                  pred_s.sort(key=lambda x: x["score"], reverse=True)
                  print(f"pred_s: {pred_s}")
                  # sort scores in descending order

                  pred_tokens = [token["token"] for token in pred_s]
                  real_tokens = [token["token"] for token in attention_score]
                  print(f"pred_tokens: {pred_tokens}")
                  print(f"real_tokens: {real_tokens}")
                  # 计算Kendall Tau系数
                  tau = kendalltau(pred_tokens, real_tokens, variant='b')[0]
                  tau_values.append(tau)
                  print(f"Kendall Tau for this pair: {tau}")
            # 计算平均Kendall Tau系数
            avg_tau = sum(tau_values) / len(tau_values) if tau_values else 0
            score_text = f"Score under {mode} mode: {avg_tau}"
            print(score_text, "\n")
            return avg_tau, score_text
      elif mode == "Kendall Tau pre":
            # 计算每对预测和真实注意力分数之间的Kendall Tau系数
            tau_values = []
            for pred, real in zip(predicted_attention, real_attention):
                  pred_s = pred["important_tokens"][:]
                  real_s = real["important_tokens"][:]
                  
                  # sort scores in descending order
                  real_s.sort(key=lambda x: x["score"], reverse=True)
                  print(f"real_s: {real_s}")
                  pred_s.sort(key=lambda x: x["score"], reverse=True)
                  print(f"pred_s: {pred_s}")
                  pred_tokens = [token["token"] for token in pred_s]
                  real_tokens = [token["token"] for token in real_s]
                  print(f"pred_tokens: {pred_tokens}")
                  print(f"real_tokens: {real_tokens}")
                  # 自定义计算，考虑
                  tau = calculate_custom_kendall_tau(pred_tokens, real_tokens)
                  tau_values.append(tau)
                  print(f"Kendall Tau for this pair: {tau}")
            # 计算平均Kendall Tau系数
            avg_tau = sum(tau_values) / len(tau_values) if tau_values else 0
            score_text = f"Score under {mode} mode: {avg_tau}"
            print(score_text, "\n")
            return avg_tau, score_text
      else:
            raise ValueError("Invalid mode. Choose from 'accuracy', 'f1', 'ndcg' or 'Kendall Tau'.")

def extract_predicted_attention(predicted_attention, real_attention):
      """
      比较 predicted_attention 和 real_attention 中相同位置的句子，
      判断 important_tokens 在按 score 排序后顺序是否一致。
      如果一致，则在该句子后附加一行 "Predictions: Right"，否则附加 "Predictions: Wrong"。
      最后将每个句子的预测、真实以及判断结果拼接为字符串返回。
      """
      result = []
      
      for idx, (pred, real) in enumerate(zip(predicted_attention, real_attention), 1):
            # 获取并按照分数降序排序 important_tokens
            pred_tokens_sorted = sorted(pred["important_tokens"], key=lambda x: x["score"], reverse=True)
            real_tokens_sorted = sorted(real["important_tokens"], key=lambda x: x["score"], reverse=True)
            
            # 按 score 分组（同一分数的 token 组成一个组，组内按 token 名字排序）
            pred_grouped = [
                  sorted([token["token"] for token in group])
                  for _, group in groupby(pred_tokens_sorted, key=lambda x: x["score"])
            ]
            real_grouped = [
                  sorted([token["token"] for token in group])
                  for _, group in groupby(real_tokens_sorted, key=lambda x: x["score"])
            ]
            
            # 判断排序后分组是否一致
            correct = (pred_grouped == real_grouped)
            
            # 构建预测的句子行
            pred_sentence = pred.get("highlighted_sentence", "")
            pred_activations = pred.get("important_tokens", [])
            pred_line = f'{idx}_test (Predicted): {pred_sentence}\n'
            if pred_activations:
                  pred_acts = ", ".join(f'("{tok["token"]}", {round(tok["score"], 2)})' for tok in pred_activations)
                  pred_line += f'Predicted Activations: {pred_acts}\n'
                  
            # 构建真实的句子行
            real_sentence = real.get("highlighted_sentence", "")
            real_activations = real.get("important_tokens", [])
            real_line = f'{idx}_test (Real): {real_sentence}\n'
            if real_activations:
                  real_acts = ", ".join(f'("{tok["token"]}", {round(tok["score"], 2)})' for tok in real_activations)
                  real_line += f'Real Activations: {real_acts}\n'
            
            # 根据判断结果添加额外一行
            prediction_result = "Predictions: Right" if correct else "Predictions: Wrong"
            
            # 将结果行拼接到结果列表中
            result.append(pred_line)
            result.append(real_line)
            result.append(prediction_result + "\n")
      
      # 将所有结果转换为字符串返回
      predictions_text = "".join(result)
      return predictions_text


def calculate_logit_contribution(layer, head, example_sentences, model_name="gpt2"):
    """计算注意力头对每个句子的logit差异贡献"""
    from transformer_lens import HookedTransformer
    import torch
    
    model = HookedTransformer.from_pretrained(model_name)
    contributions = {}
    
    for key, sentence_data in example_sentences.items():
        sentence = sentence_data["sentence"]
        io = sentence_data["io"]
        
        # 计算基线logit差异（无干预）
        tokens = model.to_tokens(sentence)
        logits = model(tokens)
        io_token_id = model.to_tokens(io)[0, 1]
        
        # 计算所有token的平均logit作为对比
        baseline_diff = logits[0, -1, io_token_id] - logits[0, -1].mean()
        
        # 通过干预计算贡献（将该注意力头的输出置零）
        def intervention_hook(attn_out, hook):
            # 检查是否是目标层和头
            if hook.layer() == layer:
                # attn_out shape: [batch, seq_len, d_model]
                # 需要将特定头的输出置零
                head_dim = attn_out.shape[-1] // model.cfg.n_heads
                start_idx = head * head_dim
                end_idx = (head + 1) * head_dim
                attn_out[:, :, start_idx:end_idx] = 0
            return attn_out
            
        # 使用正确的hook名称：注意力输出
        hook_name = f'blocks.{layer}.attn.hook_result'
        hooks = [(hook_name, intervention_hook)]
        
        try:
            patched_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        except KeyError as e:
            print(f"Hook名称错误: {e}")
            print("可用的hook名称:")
            for name, _ in model.named_modules():
                if f'blocks.{layer}.attn' in name:
                    print(f"  {name}")
            raise
        
        # 计算干预后的logit差异
        patched_diff = patched_logits[0, -1, io_token_id] - patched_logits[0, -1].mean()
        
        # 贡献度 = (基线 - 干预后) / 基线，限制在0-1范围内
        epsilon = 1e-10
        if abs(baseline_diff) < epsilon:
            contribution = 0.0
        else:
            contribution = max(0, min(1, float((baseline_diff - patched_diff) / (baseline_diff + epsilon))))
        
        contributions[key] = contribution
        
    return contributions

async def compare_and_refine_hypothesis(open_router, explanation, old_hypothesis, 
                                      predicted_attention_text, logit_comparison_text,
                                      layer, head, output_dir):
      """Compare and refine hypothesis based on attention patterns and logit contributions"""
      print(f"Comparing and refining hypothesis based on attention patterns and logit contributions...")
      messages = [
            {
                  "role": "system",
                  "content": (
                  "You are a meticulous AI researcher analyzing the behavior of a specific attention head in a transformer model GPT2-small during the Indirect Object Identification (IOI) task. "
                  "IOI task involves letting a transformer model GPT2-small predict the correct indirect object (recipient) of an action from partial sentences .\n\n"
                  "Your task is to refine the hypothesis of this attention head based on the prior hypothesis and rightly or wrongly predicted attention pattern from that prior hypothesis generated by an anylysis model. \n\n"

                  "You are given:\n"
                  "- The attention head's influence to the logit difference of the model output or its influence to other attention heads known to perform functions relevant to the task.\n\n"
                  "- A prior hypothesis about what function this head may serve in IOI task.\n"
                  "- Examples where the model made correct or incorrect predictions while relying on this prior hypothesis.\n"
                  "- Beneath each predicted example, you will be given the real attention pattern of the GPT2-small's head. \n"
                  "- After the predicted and real attention patterns in each example sentences, you will be given whether the predictions is correct or incorrect in the format of 'Predictions: Right' or 'Predictions: Wrong'.\n\n"
                  "- Highlighted tokens marked with <<>> with high attention from the prediction point.\n"
                  "- The correct indirect object (marked as {{...}}).\n"
                  "- Attention activations attend from token immediately before the indirect object to highlighted tokens in the format of (important_tokens, activation score). The score is between 0 to 1 and the greater, the more the token immediately before the indirect object attend to that token.\n\n"

                  "Important:\n"
                  "In rightly predicted examples, after sort the score predicted and real activations of the important tokens respectively in descending order, the set of important tokens and the order important tokens is also same between predicted activations and real activations. That means that the prior hypothesis could explain the head's function in this example sentences. \n"
                  "The wrongly examples you receive are failure cases. That is when given an sentence related to IOI task, the analysis model, gpt-4o, predicts wrong (predicted important token, predicted attention score) based on earlier given hypothesis compared to the real attention pattern (real important token, real attention score) of the transformer model GPT2-small. Therefore, the predicted attention patterns given in this prompt reflect **misguided or misaligned behavior interpretation of the head's real function**. That means that the prior hypothesis could not explain the head's function in this example sentences and that's where you should modify the prior hypothesis to ensure that the new hypothesis is as close to real activations as possible. \n\n "
                  "Sometimes the prediction of (predicted important token, predicted attention score) may not be align with the prior hypothesis, thus you may need to change the hypothesis to be more concise when dealing with different sentence patterns.\n\n" 
                  "You must not treat these predicted patterns as evidence of what the head should be doing, but instead use them and the actual pattern below to understand **how the prior hypothesis mistakenly interpret the real head function. **.\n\n"
                  "Analyze the examples by grouping them into input categories (e.g., sentence structures, entity numbers and roles, or context types) and explain how the head behaves differently across these types. Base your final hypothesis on this classification-aware reasoning."
                  "Your hypothesis must also focus on the relation between the real important tokens, its score and the indirect object you suppose to deduce in IOI task. Since your goal is to analyze the function of specific head which may attend to specific tokens and affect the performance of the model in IOI task. \n\n"
                  "After you have analyzed the examples, you should be give a new hypothesis and then after that give specific pattern you hypothesize respectfully,(in different angles or focus on different things) each in a line and must be direct and clear. \n\n"

                  "Your goal:\n"
                  "- Consider the explanation of the head as ground truth and base your new hypothesis on this ground truth. \n"
                  "- Maintain the part of your hypothesis that is correct and can explain the rightly predicted activations of the head accurately.\n"
                  "- For thoses wrongly predicted sentences, carefully analyze the common patterns in these failures and compare them to the real and actual pattern.\n"
                  "- Identify why the prior hypothesis lead to these failures.\n"
                  "- Use this to infer what the head is actually doing and revise the hypothesis.\n\n"
                  "- Your revised hypothesis should reduce such mistakes to the greatest extent possible\n"
                  "- The hypothesis should describe the dominant functional behavior using precise linguistic, semantic, or structural terminology. Avoid overly abstract or generic phrasing like “tracks entities” or “maintains context”."
                  "- Analyze the examples by grouping them into input categories (e.g., sentence structures, entity numbers and roles, or context types) and explain how the head behaves differently across these types. Base your final hypothesis on this classification-aware reasoning."
                  "You will also see:\n"
                  "- Predictions of the attention head's contribution to logit differences and the actual contributions\n"
                  "- 'Logit Prediction: Right/Wrong' indicates whether the prediction was close to the real value\n"
                  "- Logit contributions represent the importance of this attention head for correctly predicting the indirect object\n"
                  "- High contribution values (close to 1) indicate the head is crucial for identifying the indirect object\n"
                  "- Low contribution values (close to 0) indicate the head contributes little to identifying the indirect object\n\n"
                
                  "In your analysis, consider both aspects:\n"
                  "1. Attention patterns - which tokens the head attends to and with what strength\n"
                  "2. Logit contributions - how much the head affects the final prediction\n"
                  "Your hypothesis should explain both behaviors, especially the relationship between them.\n\n"
                  
                  "**Response structure:**\n"
                  "1. Begin with a reasoning paragraph that analyzes the failed predictions and reflects on the discrepency between the predicted and real attention pattern.\n"
                  "2. Then figure out the misinterpretation of the prior hypotheisis\n"
                  "3. Then write a clean summary paragraph starting with [HYPOTHESIS]:\n"
                  "- This paragraph must clearly and abstractly describe what this head is doing (functionally).\n"
                  "- Do not include tokens, examples, scores, or error context in the [HYPOTHESIS] paragraph.\n"
                  "- The hypothesis should sound like a standalone description of the head's role in the model with no concessions or negations.\n"

                  "Example:\n"
                  "Input:\n"
                  "Layer .., Head ..\n"
                  "Explanation: ..."
                  "Old hypothesis: [REDACTED]\n"
                  "predicted examples:\n"
                  "1_test (Predicted): At ..., BBB and <<AAA>> did ..., and BBB promised to give ...  to {{AAA}}\n"
                  "Predicted Activations: (\"AAA\", 0.57)\n"
                  "1_test (Real): At ..., BBB and <<AAA>> did ..., and BBB promised to give ...  to {{AAA}}\n"
                  "Real Activations: (\"AAA\", 0.72)\n"
                  "Predictions: Right\n"
                  "2_test (Predicted): When <<CCC>> and <<DDD>> got a ..., CCC decided to give it to {{DDD}}\n"
                  "Predicted Activations: (\"CCC_1\", 0.73), (\"DDD\", 0.11)\n"
                  "2_test (Real): When <<CCC>> and <<DDD>> got a ..., CCC decided to give it to {{DDD}}\n"
                  "Real Activations: (\"CCC_1\", 0.13), (\"DDD\", 0.71)\n"
                  "Predictions: Wrong\n"
                  "Output:\n"
                  """
                  You are tasked with analyzing attention head behavior in a transformer model. Carefully examine the following examples and perform the following steps:

                  1. Analyze the discrepancies between predicted and real attention activations.
                  2. Reflect on the mistakes in the old hypothesis and identify where they fail to account for actual model behavior.
                  3. Group the examples by input categories, such as:
                  - Sentence structures
                  - Number and type of entities
                  - Context type
                  - Difference in roles of entities

                  4. For each category, describe in one line a possible refined and specific hypothesis about the attention head’s behavior. Each line should:
                  - Focus on a narrow, testable behavior
                  - Be interpretable and concrete
                  - Avoid vague terms like "contextual role" or "semantic importance"

                  [HYPOTHESIS]: Attention head X in layer Y consistently attends to [summarized pattern based on categories above], such as [brief examples]. This behavior is most prominent in [specific sentence types] and fails in [specific failure cases], suggesting that the head is encoding [interpretable function] rather than [previous mistaken assumption].

                  Ensure your [HYPOTHESIS] paragraph avoids vague or abstract phrasing. Use direct, falsifiable language that clearly describes which tokens are attended to and under what structural conditions.
                  """

                  "Your hypothesis should be directly testable or falsifiable through model behavior, not theoretical or ambiguous. Make it as clear and concise as possible so that another researcher could validate or reject it based on measurable attention behavior."

                  )
            },
            {
                "role": "user",
                "content": (
                    f"Layer {layer}, Head {head}\n"
                    f"Explanation: {explanation}\n"
                    "Old hypothesis:\n"
                    f"{old_hypothesis}\n\n"
                    "Wrongly predicted attention patterns:\n"
                    f"{predicted_attention_text}\n\n"
                    "Logit contribution comparison:\n"
                    f"{logit_comparison_text}\n\n"
                    "Based on old hypothesis and the incorrect predictions (both attention patterns and logit contributions), "
                    "please propose a revised hypothesis that better explains the attention head's behavior."
                    "Please write a full response including a thorough reasoning (do not copy the instruction), independent lines of clear and detailed hypothesis of a specific view of the head's function and a final [HYPOTHESIS] paragraph."
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
                "1. Based on the given hypothesis, predict the contribution of this specific attention head to the logit difference for each sentence\n"
                "2. Output a value between 0 and 1, representing the head's contribution to the logit difference\n"
                "3. Higher values indicate greater contribution to correctly predicting the indirect object\n\n"
                
                "Output format:\n"
                "One line per sentence, formatted as: key: score\n"
                "For example:\n"
                "1_test: 0.75\n"
                "2_test: 0.62\n"
                
                "Important notes:\n"
                "- Only output sentence IDs and scores, no explanations\n"
                "- Scores should be between 0 and 1\n"
                "- Base scores on how well the head's function described in the hypothesis applies to each sentence"
            )
        },
        {
            "role": "user", 
            "content": (
                f"Based on the following hypothesis for layer {layer}, head {head}, predict the head's contribution to logit differences for each sentence:\n\n"
                f"Hypothesis:\n{hypothesis_text}\n\n"
                f"Sentences:\n{example_sentences}\n\n"
                f"For each sentence, predict a score between 0-1 indicating this attention head's contribution to correctly identifying the indirect object."
            )
        }
    ]
    
    predicted = await open_router.generate(messages=messages, output_dir=output_dir)
    return predicted.text.strip()


def compare_logit_contributions(predicted_contributions_text, real_contributions):
    """比较预测的和真实的logit贡献度"""
    predicted_contributions = {}
    
    # 解析预测文本
    for line in predicted_contributions_text.strip().split("\n"):
        if ":" in line:
            key, value_str = line.split(":", 1)
            try:
                value = float(value_str.strip())
                predicted_contributions[key.strip()] = value
            except ValueError:
                print(f"无法解析logit贡献分数: {line}")
    
    # 计算评估指标
    mse = 0.0  # 均方误差
    mae = 0.0  # 平均绝对误差
    count = 0
    
    for key, real_value in real_contributions.items():
        if key in predicted_contributions:
            pred_value = predicted_contributions[key]
            mse += (pred_value - real_value) ** 2
            mae += abs(pred_value - real_value)
            count += 1
    
    if count > 0:
        mse /= count
        mae /= count
    
    return {
        "mse": mse,
        "mae": mae,
        "predicted": predicted_contributions,
        "real": real_contributions
    }
    
    
async def main():
      batch_size = 10  ## 每次处理的句子数量
      iter = 10        ## 迭代次数
      model = "gpt-4o"  ## 使用的模型
      # 解析命令行参数
      args = await parse_arguments()
      layer = args.layer
      head = args.head
      rounds = args.rounds
      typename = args.typename
      output_dir = args.output_dir
      print(f"Using layer: {layer}, head: {head}")
      print(f"Using rounds: {rounds}")
      print(f"Head typename: {typename}")
      path_patching_explanations = """
            Path patching selectively copies internal activations (like hidden states) from a "clean" run (with the correct output) to a "corrupted" run (with the wrong or altered input) along specific paths in the computational graph. By measuring how this affects the output, it reveals which components (like attention heads or MLP layers) are causally responsible for correct behavior.
      """
      explanations = {
            "name_mover_head": f"Attention head {head} in layer {layer} has a high influence to the logit difference of the model output. By path-patching this head to the final logit difference of the model output with irrelavant inputs, the logit difference of indirect subject and subject decreased by 17%, causing the model to perform worse on IOI task.",
            "s_inhibition_head":f"Attention head {head} in layer {layer} has a high influence to the logit difference of the model output. By path-patching this head to name mover heads like 9.6, 9.9 and 10.0, which are active at END, attend to previous names in the sentence, copy the names they attend to and output the remaining name. By changing the activations from this head to name mover heads to activations on the same model but irrelavant tasks. Model's logit difference of indirect subject and subject decreased by 20%, causing the model to perform worse on IOI task.",
            "induction_head":f"Attention head {head} in layer {layer} has a high influence to the logit difference of the model output. Firstly, to be clear, name mover heads like 9.6, 9.9 and 10.0 are active at END, attend to previous names in the sentence, copy the names they attend to and output the remaining name. Then, By path-patching head {head} in layer {layer} to s_inhibition_heads like 7.3, 7.9, 8.6 and 8.10, which are are active at the END token, attend to the S2(The second time that the subject of the sentence shows up) token, and write in the query of the Name Mover Heads, inhibiting their attention to S1(The first time that the subject of the sentence shows up) and S2 tokens, and overally remove duplicate tokes from name mover heads' attention(Path patching is to change the activations from this head to s_inhibition heads to activations on the same model but irrelavant tasks). Model's logit difference of indirect subject and subject decreased by 50%, causing the model to perform worse on IOI task."
      }
      # 从当前路径/data63/private/chensiyuan/EAP-IG/tests/experiments/auto.py索引到/data63/private/chensiyuan/EAP-IG/results/ioi/hypothesis
      root_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ioi", "hypothesis")
      # 初始化路径和OpenRouter

      json_path = os.path.join(output_dir, "attention_scores.jsonl")  ## 初始轮gpt2-small的attention score
      dataset_sentence_path = os.path.join(output_dir, "raw_model_prompt_attention_scores.jsonl") ## 基于数据库所有句子的gpt2-small的attention score
      output_sentence_path = os.path.join(output_dir, "training_sentences.jsonl") ## 迭代过程中每轮选出的句子
      raw_attention_score_path = os.path.join(output_dir, "raw_model_prompt_attention_scores.jsonl") ## 整个数据集的gpt2-small的attention score
      open_router = initialize_openrouter(model=model)
      open_router_highlight = initialize_openrouter(model="deepseek-v3")
      print("Using OpenRouter model:", model)
      head_explanation_path= os.path.join(root_dir, "head_explanations.jsonl")
      with open(head_explanation_path, "r") as f:
            head_explanations_list = json.load(f)
            specific_head= f"{layer}.{head}"
            head_explanation= None
            for item in head_explanations_list:
                  if specific_head in item:
                        head_explanation = item[specific_head]
                        break
            if head_explanation is None:
                  raise ValueError(f"Head explanation for {specific_head} not found in {head_explanation_path}.")
      explanation = path_patching_explanations + head_explanation
      # 第一步：生成假设
      example_sentence, example_activations, example_indirect_object = load_examples(json_path)
      hypothesis_text = await generate_hypothesis(open_router, layer, head, explanation, example_sentence, example_activations, example_indirect_object, output_dir)
      hypothesis_analysis, extracted_hypothesis = extract_hypothesis_text(hypothesis_text)
      print("hypothesis_analysis:", hypothesis_analysis)
      print("extracted_hypothesis:", extracted_hypothesis)
      if not extracted_hypothesis:
            return
      
      results=[]
      iteration = 1
      while True:
            print(f"Iteration: {iteration}")
            # 第二步：根据比例随机挑选例子输入给gpt2-small计算真实attention score
            example_sentences = random_sample_sentences(dataset_sentence_path, output_sentence_path, batch_size, iteration)
            print("example_sentences:", example_sentences,"\n")

            # 计算gpt2-small的真实attention score
            example_attention_score_filename = f"model_prompt_attention_scores_{rounds}_{iteration}.jsonl"  ## 保存当前iteration下gpt2-small的attention score
            example_attention_score_path = os.path.join(output_dir, example_attention_score_filename)
            # run_attention_analysis(layer, head, output_dir, example_sentences, example_attention_score_filename)
            load_real_attention_pattern(raw_attention_score_path, example_attention_score_path, example_sentences)
            
            # 第三步：预测注意力分数
            example_attention_score_file = os.path.join(output_dir, example_attention_score_filename)
            with open(example_attention_score_file, "r") as f:
                  example_attention_data = json.load(f)
            formatted_output = format_examples_and_tokens(example_attention_data)
            ## 让模型根据给定的假设和例子和重要token数目预测重要token并用<<>>标记生成高亮句子
            highlighted_sentences_text = await underscore_important_tokens(open_router_highlight, layer, head, extracted_hypothesis, formatted_output, output_dir)
            ## 让模型根据给定的假设和高亮句子预测重要token注意力分数
            predicted_attention_text = await predict_attention_scores(open_router, layer, head, extracted_hypothesis, highlighted_sentences_text, output_dir)
            

            # 第四步：比较和精炼假设
            ## 加载gpt2-small的真实的重要token注意力分数
            real_attention_json = convert_attention_to_json(example_attention_data)
            print("real_attention_json:", real_attention_json,"\n")
            predict_attention_json = convert_predict_attention_to_json(predicted_attention_text, real_attention_json)
            print("predicted_attention_json:", predict_attention_json,"\n")
            ## 过滤掉预测和真实注意力分数中重要token数量不一致的例子
            whole_attention_score= [item["attention_scores"] for item in example_attention_data]
            print("whole_attention_score:", whole_attention_score,"\n")
  
            print("\n🔍 验证预测和真实数据的对应关系:")
            mismatch_found = False
            for idx, (pred, real) in enumerate(zip(predict_attention_json, real_attention_json)):
                print(f"索引 {idx}:")
                print(f"  预测: key={pred.get('key')}, IO={pred.get('indirect_object')}, tokens={len(pred.get('important_tokens', []))}")
                print(f"  真实: key={real.get('key')}, IO={real.get('indirect_object')}, tokens={len(real.get('important_tokens', []))}")
                  
                if pred.get('key') != real.get('key'):
                    print(f"❌ 索引错位！")
                    mismatch_found = True  # 记录问题，但不break

            # 继续执行，这时会遇到IndexError
            if not mismatch_found:
                print("✅ 验证通过，继续计算Kendall Tau")
            
            accu, _ = compare_attention_score(predict_attention_json, real_attention_json, mode="accuracy")
            f1, _ = compare_attention_score(predict_attention_json, real_attention_json, mode="f1")
            ndcg, _ = compare_attention_score(predict_attention_json, real_attention_json, mode="ndcg", whole_attention_score=whole_attention_score)
            ndcg_pre, _ = compare_attention_score(predict_attention_json, real_attention_json, mode="ndcg pre", whole_attention_score=whole_attention_score)
            kendall_tau, _ = compare_attention_score(predict_attention_json, real_attention_json, mode="Kendall Tau", whole_attention_score=whole_attention_score)
            kendall_tau_pre, _ = compare_attention_score(predict_attention_json, real_attention_json, mode="Kendall Tau pre")
            results.append({
                  "iteration": iteration,
                  "hypothesis": extracted_hypothesis,
                  "scores": {
                        "accuracy": accu,
                        "f1": f1,
                        "ndcg": ndcg,
                        "ndcg pre": ndcg_pre,
                        "Kendall Tau": kendall_tau,
                        "Kellall Tau pre": kendall_tau_pre
                  },
                  "predicted_attention": predict_attention_json,
                  "real_attention": real_attention_json,
                  "hypothesis_analysis": hypothesis_analysis,
            })
            
            # 第五步 - 计算和预测logit差异贡献
            print("\n🔍 Evaluating logit difference contribution:")

            # 计算真实logit贡献
            real_logit_contributions = calculate_logit_contribution(layer, head, example_sentences)
            print(f"Real logit contributions: {real_logit_contributions}")

            # 构建格式化的句子文本，用于AI预测
            formatted_sentences = "\n".join([f"{key}: {value['sentence']} {{{{{value['io']}}}}} " 
                                        for key, value in example_sentences.items()])

            # 让AI预测logit贡献
            predicted_logit_text = await predict_logit_contribution(
                open_router, 
                layer, head, 
                extracted_hypothesis,
                formatted_sentences, 
                output_dir
            )

            # 比较预测和真实贡献
            logit_comparison = compare_logit_contributions(predicted_logit_text, real_logit_contributions)

            print(f"Logit contribution evaluation:")
            print(f"  Mean Squared Error (MSE): {logit_comparison['mse']:.4f}")
            print(f"  Mean Absolute Error (MAE): {logit_comparison['mae']:.4f}")

            # 将logit评估结果添加到results中
            results[-1]["scores"]["logit_mse"] = logit_comparison["mse"] 
            results[-1]["scores"]["logit_mae"] = logit_comparison["mae"]
            print(f"Iteration {iteration} results: {results[-1]}","\n")

            if (accu >= 1.0 and f1 >= 1.0 and ndcg >= 0.95 and kendall_tau >= 0.99 
                and ndcg_pre >= 0.8 and kendall_tau_pre >= 0.8 and logit_comparison["mse"] <= 0.1):
                print("The predicted attention scores and logit contributions match the real values.")
                print("The hypothesis is valid.")
                print("The hypothesis is:", extracted_hypothesis)
                with open(f"{output_dir}/hypothesis.txt", "w") as f:
                    f.write(extracted_hypothesis)
                break
            else: 
                predicted_attention_text = extract_predicted_attention(predict_attention_json, real_attention_json)
                
                # 构建logit比较文本
                logit_comparison_text = []
                for key, pred_value in logit_comparison["predicted"].items():
                    real_value = logit_comparison["real"].get(key, 0.0)
                    diff = abs(pred_value - real_value)
                    status = "Right" if diff < 0.2 else "Wrong"
                    logit_comparison_text.append(f"{key} (Predicted logit contribution): {pred_value:.2f}")
                    logit_comparison_text.append(f"{key} (Real logit contribution): {real_value:.2f}")
                    logit_comparison_text.append(f"Logit Prediction: {status}\n")
                
                logit_comparison_str = "\n".join(logit_comparison_text)
                
                # 修改函数调用，添加logit比较参数
                hypothesis_text = await compare_and_refine_hypothesis(
                    open_router, 
                    explanation, 
                    extracted_hypothesis, 
                    predicted_attention_text,
                    logit_comparison_str,  # 新增参数
                    layer, head, 
                    output_dir
                )
                
                hypothesis_analysis, extracted_hypothesis = extract_hypothesis_text(hypothesis_text)
            if iteration >= iter:
                  print(f"iteration is over {iteration}, stop the iteration.")
                  break
            iteration += 1

      with open(f"{output_dir}/results_{rounds}.json", "w") as f:
            json.dump(results, f, indent=4)
      print(f"Results saved to {output_dir}/results_{rounds}.json")
if __name__ == "__main__":
      asyncio.run(main())