import asyncio
import json
import os
import re
import sys

# 确保可以导入项目中的其他模块
sys.path.append("/home/wangziran/eap_auto/") 
from api import OpenRouter

def initialize_openrouter(model: str = "gpt-4o"):
    """初始化OpenRouter API"""
    api_key = "sk-t9ooUIrk73Zg4s72dFCf2QAYWrNsobTW1gT8P7AG7m1r4Wbd" # 请确保这是你有效的Key
    return OpenRouter(model=model, api_key=api_key)

async def generate_ioi_sentences_with_metadata(open_router, num_sentences=100, output_dir=None):
    """调用LLM生成IOI句子，并直接返回包含IO和S token的结构化JSON。"""
    print(f"向LLM请求生成 {num_sentences} 条带有元数据的IOI句子...")
    
    # 【最终版Prompt】融合了你原始Prompt的所有细节（包括全部句式示例）和新的结构化JSON输出要求
    messages = [
        {
            "role": "system",
            "content": (
                "You are a meticulous AI researcher. Your task is to generate example sentences for the Indirect Object Identification (IOI) task and provide metadata for each.\n\n"
                "### PART 1: SENTENCE GENERATION GUIDELINES (Follow these strictly)\n\n"
                "- The sequence you formed should meet the need of Indirect Object Identification task, that is it is not complete sentence but only missing the indirect object at the end of the sentence.\n"
                "- The sequences you formed should be different from the sequences given in the examples.\n"
                "- Try to form sentences that match the requirements but with different sentence structure, content and especially names.\n"
                "- Form sentence that subjects and indirect objects are in different sequence, that is do not assume that subjects always appears at the first. You could form sentence like 'A and B do sth, B give sth to'\n"
                "- Beware that subjects could appear later than the indirect object in the first half sentence, so you should not assume that the subject always appears at the beginning of the sentence.\n"
                "- Do not assume that the first name in the sentence is the subject, it could be the indirect object.\n"
                "- Form around half of the sentences that first name is not the subject, but the indirect object.\n"
                "- Form sentences that the second name is the subject, which appears independently at the later half of the sentence and the first name is the indirect object, which only appears once in the sentence. \n"
                "- Make sure that the subject appears twice in the sentence and indirect object only once, so that the rational indirect object to complete the sentence is unique and clear.\n"
                "- You should not use personal pronoun but use specific names. \n"
                "- Do not use the same name as the subject as the indirect object.\n\n"
                "### PART 2: PROVIDED RESOURCES\n\n"
                "- **Name Options (use these and others):** 'Aaron', 'Adam', 'Alan', 'Alex', 'Alice', 'Amy', 'Anderson', 'Andre', 'Andrew', 'Andy', 'Anna', 'Anthony', 'Arthur', 'Austin', 'Blake', 'Brandon', 'Brian', 'Carter', 'Charles', 'Charlie', 'Christian', 'Christopher', 'Clark', 'Cole', 'Collins', 'Connor', 'Crew', 'Crystal', 'Daniel', 'David', 'Dean', 'Edward', 'Elizabeth', 'Emily', 'Eric', 'Eva', 'Ford', 'Frank', 'George', 'Georgia', 'Graham', 'Grant', 'Henry', 'Ian', 'Jack', 'Jacob', 'Jake', 'James', 'Jamie', 'Jane', 'Jason', 'Jay', 'Jennifer', 'Jeremy', 'Jessica', 'John', 'Jonathan', 'Jordan', 'Joseph', 'Joshua', 'Justin', 'Kate', 'Kelly', 'Kevin', 'Kyle', 'Laura', 'Leon', 'Lewis', 'Lisa', 'Louis', 'Luke', 'Madison', 'Marco', 'Marcus', 'Maria', 'Mark', 'Martin', 'Mary', 'Matthew', 'Max', 'Michael', 'Michelle', 'Morgan', 'Patrick', 'Paul', 'Peter', 'Prince', 'Rachel', 'Richard', 'River', 'Robert', 'Roman', 'Rose', 'Ruby', 'Russell', 'Ryan', 'Sarah', 'Scott', 'Sean', 'Simon', 'Stephen', 'Steven', 'Sullivan', 'Taylor', 'Thomas', 'Tyler', 'Victoria', 'Warren', 'William'.\n"
                "- **Structure Examples (use these as inspiration for diverse structures):**\n"
                "  \"Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to\"\n"
                "  \"Then, [B] and [A] had a lot of fun at the [PLACE]. [A] gave a [OBJECT] to\"\n"
                "  \"Then, [B] and [A] were working at the [PLACE]. [A] decided to give a [OBJECT] to\"\n"
                "  \"Then, [A] and [B] were thinking about going to the [PLACE]. [A] wanted to give a [OBJECT] to\"\n"
                "  \"Then, [A] and [B] had a long argument, and afterwards [A] said to\"\n"
                "  \"After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to\"\n"
                "  \"When [A] and [B] got a [OBJECT] at the [PLACE], [A] decided to give it to\"\n"
                "  \"When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to\"\n"
                "  \"While [A] and [B] were working at the [PLACE], [A] gave a [OBJECT] to\"\n"
                "  \"While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to\"\n"
                "  \"After the lunch, [B] and [A] went to the [PLACE]. [A] gave a [OBJECT] to\"\n"
                "  \"Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to\"\n"
                "  \"Then, [B] and [A] had a long argument. Afterwards [A] said to\"\n"
                "  \"The [PLACE] [B] and [A] went to had a [OBJECT]. [A] gave it to\"\n"
                "  \"Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to\"\n"
                "  \"Then in the morning, [A] and [B] went to the [PLACE]. [A] gave a [OBJECT] to\"\n"
                "  \"Then in the morning, [A] and [B] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to\"\n"
                "  \"Then in the morning, [A] and [B] were working at the [PLACE]. [A] decided to give a [OBJECT] to\"\n"
                "  \"Then in the morning, [B] and [A] were thinking about going to the [PLACE]. [A] wanted to give a [OBJECT] to\"\n"
                "  \"Then in the morning, [A] and [B] had a long argument, and afterwards [B] said to\"\n"
                "  \"After taking a long break [A] and [B] went to the [PLACE], [A] gave a [OBJECT] to\"\n"
                "  \"When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [A] decided to give it to\"\n"
                "  \"When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [A] decided to give the [OBJECT] to\"\n"
                "  \"While spending time together [A] and [B] were working at the [PLACE], [A] gave a [OBJECT] to\"\n"
                "  \"While spending time together [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to\"\n"
                "  \"After the lunch in the afternoon, [B] and [A] went to the [PLACE]. [A] gave a [OBJECT] to\"\n"
                "  \"Afterwards, while spending time together [A] and [B] went to the [PLACE]. [B] gave a [OBJECT] to\"\n"
                "  \"Then in the morning afterwards, [B] and [A] had a long argument. Afterwards [A] said to\"\n"
                "  \"The local big [PLACE] [A] and [B] went to had a [OBJECT]. [B] gave it to\"\n"
                "  \"Friends separated at birth [A] and [B] found a [OBJECT] at the [PLACE]. [A] gave it to\"\n"
                "  \"Then, [A] and [B] went to the [PLACE]. [A] gave a [OBJECT] to\"\n"
                "  \"Then, [B] and [A] had a lot of fun at the [PLACE]. [A] gave a [OBJECT] to\"\n"
                "  \"Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to\"\n"
                "  \"Then, [A] and [B] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to\"\n"
                "  \"Then, [B] and [A] had a long argument and after that [A] said to\"\n"
                "  \"After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to\"\n"
                "  \"Afterwards, [B] and [A] went to the [PLACE]. [A] gave a [OBJECT] to\"\n"
                "  \"Then, [B] and [A] had a long argument. Afterwards [A] said to\"\n"
                "  \"Then [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to\"\n"
                "  \"Then [B] and [A] had a lot of fun at the [PLACE], and [A] gave a [OBJECT] to\"\n"
                "  \"Then [B] and [A] were working at the [PLACE], and [B] decided to give a [OBJECT] to\"\n"
                "  \"Then [B] and [A] were thinking about going to the [PLACE], and [A] wanted to give a [OBJECT] to\"\n"
                "  \"Then [B] and [A] had a long argument, and after that [B] said to\"\n"
                "  \"After the lunch [B] and [A] went to the [PLACE], and [A] gave a [OBJECT] to\"\n"
                "  \"Afterwards [A] and [B] went to the [PLACE], and [A] gave a [OBJECT] to\"\n\n"
                "### PART 3: CRITICAL OUTPUT FORMAT\n\n"
                "1.  **Analyze each sentence you create.** Identify the `io_token` and `s_token`.\n"
                "2.  **Format your entire response as a single, raw JSON array of objects.** Do NOT add any introductory text like 'Here is the JSON...'.\n"
                "3.  Each object in the array must contain three keys: `sentence_text`, `io_token`, and `s_token`.\n"
                "4.  The `io_token` and `s_token` values MUST match the names in the sentence exactly, **including any preceding spaces** (e.g., ' Mary', not 'Mary').\n\n"
                "### PART 4: EXAMPLE OF THE FINAL REQUIRED OUTPUT\n\n"
                "```json\n"
                "[\n"
                "  {\n"
                "    \"sentence_text\": \"Then, Mary and John went to the store. John gave a book to\",\n"
                "    \"io_token\": \" Mary\",\n"
                "    \"s_token\": \" John\"\n"
                "  },\n"
                "  {\n"
                "    \"sentence_text\": \"After Kate and David had lunch, David gave a flower to\",\n"
                "    \"io_token\": \" Kate\",\n"
                "    \"s_token\": \" David\"\n"
                "  }\n"
                "]\n"
                "```"
            )
        },
        { "role": "user", "content": f"Please generate {num_sentences} unique IOI sentences that strictly follow all the rules, in the specified JSON format." },
    ]
    
    response = await open_router.generate(messages=messages, output_dir=output_dir)
    
    try:
        structured_data = json.loads(response.text)
        print(f"成功从LLM响应中解析出 {len(structured_data)} 条句子。")
        return structured_data
    except json.JSONDecodeError:
        print("错误：LLM未能返回有效的JSON格式。")
        match = re.search(r'\[.*\]', response.text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                print("后备方案：通过正则表达式找到JSON。")
                return data
            except json.JSONDecodeError:
                print("后备方案失败：正则表达式提取的内容也不是有效的JSON。")
                return []
        return []

async def main():
    """主函数：只负责生成句子并保存。"""
    NUM_SENTENCES_TO_GENERATE = 100
    BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ioi", "path_patching")
    OUTPUT_SENTENCE_FILE = os.path.join(BASE_OUTPUT_DIR, "structured_sentences.jsonl")
    LOG_DIR = os.path.join(BASE_OUTPUT_DIR, "generation_logs")
    
    os.makedirs(LOG_DIR, exist_ok=True)

    open_router = initialize_openrouter()
    sentences_with_metadata = await generate_ioi_sentences_with_metadata(
        open_router, 
        NUM_SENTENCES_TO_GENERATE,
        output_dir=LOG_DIR
    )
    
    if not sentences_with_metadata:
        print("未能生成句子，程序终止。")
        return

    valid_sentences_count = 0
    with open(OUTPUT_SENTENCE_FILE, "w") as f:
        for i, sentence_info in enumerate(sentences_with_metadata):
            if not all(k in sentence_info for k in ['sentence_text', 'io_token', 's_token']):
                print(f"警告：跳过一条缺少键的记录: {sentence_info}")
                continue

            if sentence_info['io_token'] in sentence_info['sentence_text'] and sentence_info['s_token'] in sentence_info['sentence_text']:
                record = {
                    "sentence_id": f"ioi_{i+1:04d}",
                    "sentence_text": sentence_info["sentence_text"],
                    "io_token": sentence_info["io_token"],
                    "s_token": sentence_info["s_token"]
                }
                f.write(json.dumps(record) + "\n")
                valid_sentences_count += 1
            else:
                print(f"警告：跳过一条不符合规范的句子 (IO或S不在文本中): {sentence_info}")
            
    print(f"\n✅ 步骤一完成！共保存了 {valid_sentences_count} 条有效句子至: {OUTPUT_SENTENCE_FILE}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    asyncio.run(main())