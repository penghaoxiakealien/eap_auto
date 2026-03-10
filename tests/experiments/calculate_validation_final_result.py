import os
import json
from auto_test import collect_final_result
typename = "name_mover_head"
specific_head = "9.6"
root_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ioi", "hypothesis")
# 初始化路径和OpenRouter
output_dir = os.path.join(root_dir, typename)
head_dir = os.path.join(root_dir, specific_head)  ## 单个head测试结果的路径
input_dir = os.path.join(output_dir, specific_head)  ## 单个head测试结果的路径
with open(os.path.join(input_dir, "detailed_results.json"), "r") as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file : {e}")
        exit(1)

    collect_final_result(data, input_dir)