import asyncio
import sys
import json
import os
import re
import math
import random
from scipy.stats import kendalltau
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt_tab')
from collections import defaultdict
import argparse
sys.path.append("/home/wangziran/eap_auto/")
# sys.path.append("/data63/private/chensiyuan/EAP-IG/")
from api import OpenRouter
from attention_score_by_head import run
from auto import initialize_openrouter, convert_attention_to_json, convert_predict_attention_to_json, compare_attention_score, calculate_custom_kendall_tau, calculate_sample_distribution, random_sample_sentences, run_attention_analysis, load_examples, format_examples_and_tokens, underscore_important_tokens, predict_attention_scores, convert_attention_to_json, convert_predict_attention_to_json, load_real_attention_pattern

async def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Run OpenRouter with specified layer and head.")
    parser.add_argument("--head", type=str, required=True, help="The head number to analyze.")
    parser.add_argument("--typename", type=str, required=True, help="The type name for the analysis.")
    return parser.parse_args()

def load_hypotheses_from_jsonl(json_file):
    round_to_hypothesis = {}
    with open(json_file, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file {json_file}: {e}")
            return round_to_hypothesis

    for item in data:
        round_number = item.get("round")
        hypothesis = item.get("top_k_hypothesis", {}).get("hypothesis")
        if round_number is not None and hypothesis:
            round_to_hypothesis[round_number] = hypothesis
    return round_to_hypothesis


def collect_final_result(head_results, head_dir):
    """
    收集最终结果并保存到文件中。
    """
    final_results = []
    for head_result in head_results:
        head = head_result["head"]
        hypothesis = head_result["hypothesis"]
        results = head_result["results"]
        # 计算score里accuracy、f1、ndcg和Kendall Tau的平均值
        avg_accuracy = sum(result["scores"]["accuracy"] for result in results) / len(results)
        avg_f1 = sum(result["scores"]["f1"] for result in results) / len(results)
        avg_ndcg = sum(result["scores"]["ndcg"] for result in results) / len(results)
        avg_ndcg_pre = sum(result["scores"]["ndcg pre"] for result in results) / len(results)
        avg_kendall_tau = sum(result["scores"]["Kendall Tau"] for result in results) / len(results)
        avg_kendall_tau_pre = sum(result["scores"]["Kendall Tau pre"] for result in results) / len(results)
        #计算score里accuracy、f1、ndcg和Kendall Tau的最大值
        max_accuracy = max(result["scores"]["accuracy"] for result in results)
        max_f1 = max(result["scores"]["f1"] for result in results)
        max_ndcg = max(result["scores"]["ndcg"] for result in results)
        max_ndcg_pre = max(result["scores"]["ndcg pre"] for result in results)
        max_kendall_tau = max(result["scores"]["Kendall Tau"] for result in results)
        max_kendall_tau_pre = max(result["scores"]["Kendall Tau pre"] for result in results)
        final_results.append({
            "head": head,
            "hypothesis": hypothesis,
            "avg_accuracy": avg_accuracy,
            "max_accuracy": max_accuracy,
            "avg_f1": avg_f1,
            "max_f1": max_f1,
            "avg_ndcg": avg_ndcg,
            "max_ndcg": max_ndcg,
            "avg_ndcg_pre": avg_ndcg_pre,
            "max_ndcg_pre": max_ndcg_pre,
            "avg_kendall_tau": avg_kendall_tau,
            "max_kendall_tau": max_kendall_tau,
            "avg_kendall_tau_pre": avg_kendall_tau_pre,
            "max_kendall_tau_pre": max_kendall_tau_pre
        })
    # 先按照avg_score降序排列，若一致则按照avg_ndcg_pre，若一致则按照avg_kendall_tau_pre
    final_results.sort(key=lambda x: (-x["avg_accuracy"], -x["avg_ndcg_pre"], -x["avg_kendall_tau_pre"]))
    # 保存最终结果到文件
    final_result_path = os.path.join(head_dir, "final_results.json")
    with open(final_result_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print("Final results saved to final_results.json")


async def main():
    batch_size = 10
    iter = 5
    # 解析命令行参数
    args = await parse_arguments()
    specific_head = args.head
    typename = args.typename
    print(f"Using specific_head: {specific_head}")
    print(f"Analysis type: {typename}")
    layer, head= map(int, specific_head.split("."))
    path_patching_explanations = """
            Path patching selectively copies internal activations (like hidden states) from a "clean" run (with the correct output) to a "corrupted" run (with the wrong or altered input) along specific paths in the computational graph. By measuring how this affects the output, it reveals which components (like attention heads or MLP layers) are causally responsible for correct behavior.
    """
    # 从当前路径/data63/private/chensiyuan/EAP-IG/tests/experiments/auto.py索引到/data63/private/chensiyuan/EAP-IG/results/ioi/hypothesis
    root_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ioi", "hypothesis")
    # 初始化路径和OpenRouter
    output_dir = os.path.join(root_dir, typename)
    head_dir = os.path.join(root_dir, specific_head)  ## 单个head测试结果的路径
    output_head_dir = os.path.join(output_dir, specific_head) ## 这类head的测试结果输出路径
    validation_sentence_path = os.path.join(output_head_dir, "raw_model_prompt_attention_scores.jsonl") ## 需要分析的句子的数据集
    output_sentence_path = os.path.join(output_head_dir, "validation_sentences.jsonl") ## 迭代过程中每轮选出的句子
    raw_attention_score_path = os.path.join(output_head_dir, "raw_model_prompt_attention_scores.jsonl") ## 整个数据集的gpt2-small的attention score
    open_router = initialize_openrouter(model="gpt-4o")
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
    # 第一步：加载假设
    hypothesis_path = os.path.join(head_dir, "best_result.jsonl")
    hypotheses = load_hypotheses_from_jsonl(hypothesis_path)
    print(f"hypotheses for head {specific_head} :", hypotheses,"\n")
    
    head_results=[]  # 记录这个head的所有结果
    for index, hypothesis in hypotheses.items():
        results=[] # 记录每次具体假设的结果
        iteration = 1
        while True:
            print(f"head: {specific_head}, hypothesis: {index}: {hypothesis}","\n")
            print(f"Iteration: {iteration}")
            # 第二步：加载待分析句子
            example_sentences = random_sample_sentences(validation_sentence_path, output_sentence_path, batch_size, iteration)

            # 第三步：运行注意力分数分析
            example_attention_score_filename = f"model_prompt_attention_scores_{index}_{iteration}.jsonl"  ## 保存当前iteration下gpt2-small的attention score
            example_attention_score_path = os.path.join(output_head_dir, example_attention_score_filename)
            # run_attention_analysis(layer, head, output_dir, example_sentences, example_attention_score_filename)
            load_real_attention_pattern(raw_attention_score_path, example_attention_score_path, example_sentences)

            # 第四步：预测注意力分数
            example_attention_score_file = os.path.join(output_head_dir, example_attention_score_filename)
            with open(example_attention_score_file, "r") as f:
                example_attention_data = json.load(f)
            formatted_output = format_examples_and_tokens(example_attention_data)
            ## 让模型根据给定的假设和例子和重要token数目预测重要token并用<<>>标记生成高亮句子
            highlighted_sentences_text = await underscore_important_tokens(open_router, layer, head, hypothesis, formatted_output, output_head_dir)
            ## 让模型根据给定的假设和高亮句子预测重要token注意力分数
            predicted_attention_text = await predict_attention_scores(open_router, layer, head, hypothesis, highlighted_sentences_text, output_head_dir)
            print("predicted_attention_text:", predicted_attention_text,"\n")
            real_attention_json = convert_attention_to_json(example_attention_data)
            print("real_attention_json:", real_attention_json,"\n")
            predict_attention_json = convert_predict_attention_to_json(predicted_attention_text, real_attention_json)
            print("predicted_attention_json:", predict_attention_json,"\n")
            ## 过滤掉预测和真实注意力分数中重要token数量不一致的例子
            whole_attention_score= [item["attention_scores"] for item in example_attention_data]
            
            accu, _ = compare_attention_score(predict_attention_json, real_attention_json, mode="accuracy")
            f1, _ = compare_attention_score(predict_attention_json, real_attention_json, mode="f1")
            ndcg, _ = compare_attention_score(predict_attention_json, real_attention_json, mode="ndcg", whole_attention_score=whole_attention_score)
            ndcg_pre, _ = compare_attention_score(predict_attention_json, real_attention_json, mode="ndcg pre", whole_attention_score=whole_attention_score)
            kendall_tau, _ = compare_attention_score(predict_attention_json, real_attention_json, mode="Kendall Tau", whole_attention_score=whole_attention_score)
            kendall_tau_pre, _ = compare_attention_score(predict_attention_json, real_attention_json, mode="Kendall Tau pre")

            results.append({
                    "iteration": iteration,
                    "scores": {
                        "accuracy": accu,
                        "f1": f1,
                        "ndcg": ndcg,
                        "ndcg pre": ndcg_pre,
                        "Kendall Tau": kendall_tau,
                        "Kendall Tau pre": kendall_tau_pre
                    },
                    "predicted_attention": predict_attention_json,
                    "real_attention": real_attention_json,
            })
            print(f"Iteration {iteration} results: {results[-1]}","\n")
            if iteration >= iter:
                    print(f"iteration is over {iteration}, stop the iteration.")
                    break
            iteration += 1
        head_results.append({
                "head": specific_head,
                "hypothesis": hypothesis,
                "results": results
            })
    
    with open(f"{output_head_dir}/detailed_results.json", "w") as f:
        json.dump(head_results, f, indent=4)
    print(f"Detailed Results saved to {output_head_dir}/detailed_results.json")

    collect_final_result(head_results, output_head_dir)

if __name__ == "__main__":
    asyncio.run(main())