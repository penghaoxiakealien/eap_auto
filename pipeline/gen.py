import openai
from openai import OpenAI
import time
import argparse
import csv
import io
import sys
import math

client = OpenAI(
    api_key="sk-tWWLjYE4fua6zjAY024dE493C9874e2d8e1a877f65A51488",
    base_url="https://yeysai.com/v1"
)

FEW_SHOT_EXAMPLES = [
    {
        "task_name": "Indirect Object Identification (IOI)",
        "description": """
        **Goal:** Test if a model can identify the correct indirect object from context.
        **Prediction Target:** The name of the indirect object.
        **Method:** A context sentence introduces two names. The second fragment must stop right after a preposition.
        **Columns:** `clean`, `corrupted`, `clean_word`, `corrupted_word`.
        **Separator:** Pipe '|'.
        """,
        "sample_output": """clean|corrupted|clean_word|corrupted_word
John and Mary were at the park. John gave a ball to|John and Mary were at the park. Steve gave a ball to|"Mary"|"Steve"
"""
    },
    {
        "task_name": "Subject-Verb Agreement",
        "description": """
        **Goal:** Test subject-verb agreement with a distractor phrase.
        **Prediction Target:** The correct verb form.
        **Method:** The sentence must stop right before the verb slot. Create balanced pairs.
        **Columns:** `clean`, `corrupted`, `clean_word`, `corrupted_word`.
        **Separator:** Pipe '|'.
        """,
        "sample_output": """clean|corrupted|clean_word|corrupted_word
The key to the cabinets|"The keys to the cabinet"|"is, was, has, 's"|"are, were, have, 've"
The keys to the cabinet|"The key to the cabinets"|"are, were, have, 've"|"is, was, has, 's"
"""
    },
    {
        "task_name": "Reflexive Pronoun Agreement",
        "description": """
        **Goal:** Test reflexive pronoun agreement.
        **Prediction Target:** The correct reflexive pronoun.
        **Method:** Use a verb that strongly implies a reflexive pronoun. Stop right after the verb. Create balanced pairs.
        **Columns:** `clean`, `corrupted`, `clean_word`, `corrupted_word`.
        **Separator:** Pipe '|'.
        """,
        "sample_output": """clean|corrupted|clean_word|corrupted_word
The defendant perjured|The defendants perjured|"himself, herself, itself"|"themselves"
The defendants perjured|The defendant perjured|"themselves"|"himself, herself, itself"
"""
    },
    {
        "task_name": "Gender Agreement",
        "description": """
        **Goal:** Test gender pronoun prediction based on a name.
        **Prediction Target:** The correct gendered pronoun.
        **Method:** Use a simple sentence structure. Stop right before the pronoun slot. Create balanced pairs.
        **Columns:** `clean`, `corrupted`, `clean_word`, `corrupted_word`.
        **Separator:** Pipe '|'.
        """,
        "sample_output": """clean|corrupted|clean_word|corrupted_word
Gary ran because|Laura ran because|"he, his"|"she, her"
Laura ran because|Gary ran because|"she, her"|"he, his"
"""
    }
]

def build_meta_prompt(few_shot_examples, new_task_desc, num_rows_to_generate):
    prompt_parts = [
        "You are a data generation bot. Your mission is to create datasets for causal analysis.",
        "",
        "**CRITICAL INSTRUCTION: Your output MUST be raw data only. It must start DIRECTLY with the header row (e.g., `clean|corrupted|...`) and contain ONLY the pipe-separated data. Do NOT include ANY introductory text, explanations, or markdown formatting like ```.**",
        "",
        "Each data row you generate represents a **minimal pair experiment**.",
        "",
        "**Here are your core principles:**",
        "1. **`clean` & `corrupted` Prompts:** Create a minimal pair of sentence fragments. The `corrupted` prompt is a single, precise intervention on the `clean` prompt.",
        "2. **`clean_word` & `corrupted_word` Columns:** These columns must contain a comma-separated list of all plausible words that could correctly complete the corresponding prompt. Enclose these lists in double quotes.",
        "3. **Prediction Probing:** Both prompts must stop *immediately* before the prediction target.",
        "4. **DIVERSITY:** You **must** use a wide and creative range of vocabulary across the generated rows.",
        "5. **Balanced Pairs:** For each generated pair, you must also generate its inverse, swapping the content of `clean`/`corrupted` and `clean_word`/`corrupted_word`.",
        "",
        "I will now show you several examples. Learn the experimental design from them.",
        "\n" + "="*20 + " EXAMPLES " + "="*20 + "\n"
    ]
    
    for example in few_shot_examples:
        prompt_parts.append("--- TASK DESCRIPTION ---")
        prompt_parts.append(example['description'].strip())
        prompt_parts.append("\n--- EXPECTED OUTPUT ---")
        prompt_parts.append(example['sample_output'].strip())
        prompt_parts.append("\n" + "="*50 + "\n")
        
    prompt_parts.append("Now, you understand your mission is to design causal analysis experiments with target word lists. Here is the new task. Apply all core principles with extreme precision.")
    prompt_parts.append("**Remember the CRITICAL INSTRUCTION: Your response must be raw data only, starting with the header.**")
    prompt_parts.append("\n" + "="*20 + " NEW TASK " + "="*20 + "\n")
    prompt_parts.append("--- TASK DESCRIPTION ---")
    prompt_parts.append(new_task_desc.strip())
    prompt_parts.append(f"\n--- EXPECTED OUTPUT ---")
    prompt_parts.append(f"(Generate {num_rows_to_generate} unique and balanced data rows for this new task, following all principles and the strict output format)")
    
    return "\n".join(prompt_parts)

def generate_data_for_new_task(task_description, num_rows=15):
    print("Constructing the meta-prompt with target word list generation...")
    meta_prompt = build_meta_prompt(FEW_SHOT_EXAMPLES, task_description, num_rows)
    
    print(f"  - 构建请求，请求生成 {num_rows} 行...")
    meta_prompt = build_meta_prompt(FEW_SHOT_EXAMPLES, task_description, num_rows)
    
    MAX_TOKENS_LIMIT = 16000
    estimated_tokens = int(num_rows * 80)
    tokens_to_request = min(estimated_tokens, MAX_TOKENS_LIMIT)
    
    print(f"  - 发送 API 请求 (tokens: {tokens_to_request})...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": meta_prompt}],
            max_tokens=tokens_to_request,
            temperature=0.8
        )
        
        generated_content = response.choices[0].message.content.strip()
        return generated_content
        
    except openai.APIError as e:
        print(f"  - API 错误: {e}")
        return None
    except Exception as e:
        print(f"  - 未知错误: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="通过批处理方式为语言模型因果分析生成大型数据集。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--task_file", type=str, required=True, help="包含新任务描述的文本文件路径。")
    parser.add_argument("--num_rows", type=int, default=20, help="希望生成的总行数。")
    parser.add_argument("--batch_size", type=int, default=100, help="每个 API 请求生成的行数。")
    parser.add_argument("--save", action="store_true", help="如果指定，则将生成的数据集保存到 .csv 文件。")
    parser.add_argument("--output_path", type=str, default="generated_dataset.csv", help="保存输出 .csv 文件的路径。仅在指定 --save 时使用。")
    parser.add_argument("--silent", action="store_true", help="如果指定，则不在终端打印生成的数据集。")
    
    args = parser.parse_args()

    try:
        with open(args.task_file, 'r', encoding='utf-8') as f:
            task_desc_to_run = f.read()
        print(f"成功加载任务描述: {args.task_file}")
    except FileNotFoundError:
        print(f"错误: 任务文件未找到 '{args.task_file}'", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    total_rows_to_generate = args.num_rows
    batch_size = args.batch_size
    num_batches = math.ceil(total_rows_to_generate / batch_size)

    print(f"计划生成 {total_rows_to_generate} 行数据，分为 {num_batches} 个批次 (每批 {batch_size} 行)。")

    for i in range(num_batches):
        print(f"\n--- 正在处理批次 {i + 1}/{num_batches} ---")
        rows_in_this_batch = min(batch_size, total_rows_to_generate - len(all_rows))
        
        if rows_in_this_batch <= 0:
            break

        generated_data = generate_data_for_new_task(task_desc_to_run, num_rows=rows_in_this_batch)
        
        if generated_data:
            if generated_data.startswith("```") and generated_data.endswith("```"):
                generated_data = generated_data.strip("```").strip()
                if generated_data.lower().startswith('csv'):
                    generated_data = generated_data[3:].strip()
            
            string_io = io.StringIO(generated_data)
            reader = csv.reader(string_io, delimiter='|')
            
            if i == 0:
                all_rows.extend(list(reader))
            else:
                next(reader)
                all_rows.extend(list(reader))
            
            print(f"  - 批次 {i + 1} 完成，已生成 {len(all_rows) - 1} 行有效数据。")
        else:
            print(f"  - 批次 {i + 1} 生成失败，跳过。")
        
        # 在两次请求之间稍作停顿，避免过于频繁地请求 API
        if i < num_batches - 1:
            time.sleep(2)

    if not all_rows:
        print("\n错误：未能生成任何数据。")
        sys.exit(1)

    print("\n所有批次处理完毕！")
    
    if not args.silent:
        print("\n--- 最终生成的数据集 (预览前5行) ---")
        # 将数据转换回纯文本进行打印
        output_io = io.StringIO()
        writer = csv.writer(output_io)
        writer.writerows(all_rows[:15]) # 打印表头+5行
        print(output_io.getvalue())
    
    if args.save:
        try:
            with open(args.output_path, "w", encoding="utf-8", newline='') as f:
                writer = csv.writer(f)
                writer.writerows(all_rows)
            print(f"\n数据集成功保存到 {args.output_path}，共 {len(all_rows) - 1} 行。")
        except Exception as e:
            print(f"\n错误: 无法将文件保存到 {args.output_path}。原因: {e}")