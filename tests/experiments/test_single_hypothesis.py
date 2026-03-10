import asyncio
import sys
import json
import os
import argparse
import random
import re
from datetime import datetime

# 确保可以导入 auto_logit_s 中的函数
sys.path.append(os.path.dirname(__file__))
from auto_logit_s import (
    initialize_openrouter,
    sample_sentences_from_causal_dataset,
    split_dataset,
    normalize_token
)

async def parse_test_arguments():
    """解析用于测试单个假设的命令行参数"""
    parser = argparse.ArgumentParser(description="Test a single, manually provided hypothesis on a sample of sentences.")
    parser.add_argument("--layer", type=int, required=True, help="The sender head's layer.")
    parser.add_argument("--head", type=int, required=True, help="The sender head's number.")
    parser.add_argument("--hypothesis", type=str, required=True, help="The hypothesis text to test, enclosed in quotes.")
    parser.add_argument("--num_sentences", type=int, default=5, help="Number of random sentences to test on.")
    parser.add_argument("--dataset_split", type=str, default="validation", choices=["train", "validation"], help="Which part of the dataset to sample from.")
    return parser.parse_args()

def calculate_f1(pred_set, real_set):
    """计算单个集合的F1分数"""
    if not isinstance(pred_set, set): pred_set = set(pred_set)
    if not isinstance(real_set, set): real_set = set(real_set)
    
    common_elements = len(pred_set.intersection(real_set))
    if common_elements == 0:
        return 0.0
    
    precision = common_elements / len(pred_set)
    recall = common_elements / len(real_set)
    
    return (2 * precision * recall) / (precision + recall)

async def main():
    args = await parse_test_arguments()
    sender_head = (args.layer, args.head)
    
    # --- 文件输出设置 ---
    output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"test_run_L{args.layer}_H{args.head}.txt")
    
    print(f"--- 初始化和加载数据 ---")
    open_router = initialize_openrouter()
    
    full_causal_dataset_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ioi", "path_patching", "causal_dataset.json")
    try:
        with open(full_causal_dataset_path, "r") as f:
            full_causal_dataset = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法加载 {full_causal_dataset_path}: {e}")
        return

    train_dataset, validation_dataset = split_dataset(full_causal_dataset)
    target_dataset = validation_dataset if args.dataset_split == "validation" else train_dataset
    
    # 准备写入文件
    with open(output_filename, "w", encoding="utf-8") as f:
        header = (
            f"--- 手动假设测试报告 ---\n"
            f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"待测试头: {sender_head}\n"
            f"数据集: {args.dataset_split}\n"
            f"抽样数量: {args.num_sentences}\n"
            f"待测试假设: \"{args.hypothesis}\"\n"
            f"-----------------------------------\n\n"
        )
        print(header)
        f.write(header)

        test_sentences = sample_sentences_from_causal_dataset(target_dataset, batch_size=args.num_sentences)

        print("\n--- 开始并行预测 ---")
        
        tasks = []
        prompts = {}
        for sid, data in test_sentences.items():
            sentence_text = data['sentence_text']
            top_k = 1
            
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
                f"**Hypothesis:** {args.hypothesis}\n"
                "Now, using the **actual hypothesis for Head {sender_head}**, apply the same reasoning process to the following sentence.\n\n"
                f"**Sentence to Analyze:**\n`{sid}: {sentence_text}`"
            )
            prompts[sid] = f"---SYSTEM PROMPT---\n{system_prompt}\n\n---USER PROMPT---\n{user_prompt}"
            messages = [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_prompt}
            ]
            task = open_router.generate(messages=messages, output_dir=output_dir)
            tasks.append((sid, task))

        api_responses = await asyncio.gather(*[t for _, t in tasks])

        total_f1 = 0
        num_evaluated = 0

        for i, (sid, _) in enumerate(tasks):
            response = api_responses[i]
            response_text = response.text.strip()
            
            # 解析预测
            marked_sentence = ""
            prediction_match = re.search(r"\[PREDICTION\]\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
            if prediction_match:
                marked_sentence = prediction_match.group(1).strip()
            
            pred_inc = [normalize_token(t) for t in re.findall(r"<<(.*?)>>", marked_sentence)]
            pred_dec = [normalize_token(t) for t in re.findall(r"\[\[(.*?)\]\]", marked_sentence)]

            # 获取真实值
            sentence_info = target_dataset.get(sid, {})
            s_token = sentence_info.get("s_token")
            io_token = sentence_info.get("io_token")
            
            if s_token and io_token:
                real_inc = {normalize_token(s_token)}
                real_dec = {normalize_token(io_token)}
                
                # 计算F1分数
                f1_inc = calculate_f1(pred_inc, real_inc)
                f1_dec = calculate_f1(pred_dec, real_dec)
                avg_f1 = (f1_inc + f1_dec) / 2
                total_f1 += avg_f1
                num_evaluated += 1
            else:
                real_inc, real_dec, avg_f1 = {"N/A"}, {"N/A"}, "N/A"

            # --- 格式化输出 ---
            result_summary = (
                f"\n\n==================== 结果 {i+1}: 句子 {sid} ====================\n"
                f"\n--- 评估摘要 ---\n"
                f"Predicted:  increase={pred_inc}, decrease={pred_dec}\n"
                f"Ground Truth: increase={list(real_inc)}, decrease={list(real_dec)}\n"
                f"F1 Score:   {avg_f1:.4f}\n"
                f"\n--- Prompt (发送给AI的内容) ---\n\n{prompts[sid]}\n"
                f"\n--- Raw Response (AI的原始回复) ---\n\n{response_text}\n"
                f"\n==================== 结束 {sid} ====================\n"
            )
            print(result_summary)
            f.write(result_summary)
        
        # --- 最终总结 ---
        final_summary = "\n\n-------------------- 最终总结 --------------------\n"
        if num_evaluated > 0:
            overall_avg_f1 = total_f1 / num_evaluated
            final_summary += f"在 {num_evaluated} 个可评估句子上的平均F1分数为: {overall_avg_f1:.4f}\n"
        else:
            final_summary += "没有可评估的句子。\n"
        final_summary += f"详细报告已保存至: {output_filename}\n"
        
        print(final_summary)
        f.write(final_summary)

if __name__ == "__main__":
    asyncio.run(main())
