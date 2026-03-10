import asyncio
import sys
import json
import os
import re
import math
import argparse
sys.path.append("/home/chensiyuan/EAP-IG/")
from api import OpenRouter
def initialize_openrouter(model:str = "gpt-4o"):
      """初始化OpenRouter API"""
      api_key = "sk-MjSyxJuoVSlripXy2933FaBaEaBb4fC1A4B564DfB699B8C2"
      return OpenRouter(model=model, api_key=api_key)
def extract_example_sentences(example_message_text):
      """从响应中提取示例句子"""
      match = re.search(r"{(.*?)}", example_message_text, re.DOTALL)
      if match:
            return json.loads(match.group(0).strip())
      print("No example sentences found in the response.")
      return {}

async def generate_sentences(open_router, output_dir):
      print(f"Generating example sentences...")
      """生成测试样例"""
      messages = [
            {
                  "role": "system",
                  "content": (
                  "You are a meticulous AI researcher conducting an important investigation into patterns found in language. "
                  "The text is based on the Indirect Object Identification task, where the model is asked to identify the indirect object in a sentence. "
                  "Your task is to generate example sentences to meet the need of IOI dataset.\n"
                  "Guidelines:\n\n"
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
                  "- Do not use the same name as the subject as the indirect object.\n"
                  "- The format of the formed sentence should be sequence number start at 1_test: the sentence you formed"
                  "- At the last line, you should arrange all the 'Example sequence number: formed sequence' into json format and output the result in one single line."
                  "- Here are some name options, but you could also come up with other names: 'Aaron', 'Adam', 'Alan', 'Alex', 'Alice', 'Amy', 'Anderson', 'Andre', 'Andrew', 'Andy', 'Anna', 'Anthony', 'Arthur', 'Austin', 'Blake', 'Brandon', 'Brian', 'Carter', 'Charles', 'Charlie', 'Christian', 'Christopher', 'Clark', 'Cole', 'Collins', 'Connor', 'Crew', 'Crystal', 'Daniel', 'David', 'Dean', 'Edward', 'Elizabeth', 'Emily', 'Eric', 'Eva', 'Ford', 'Frank', 'George', 'Georgia', 'Graham', 'Grant', 'Henry', 'Ian', 'Jack', 'Jacob', 'Jake', 'James', 'Jamie', 'Jane', 'Jason', 'Jay', 'Jennifer', 'Jeremy', 'Jessica', 'John', 'Jonathan', 'Jordan', 'Joseph', 'Joshua', 'Justin', 'Kate', 'Kelly', 'Kevin', 'Kyle', 'Laura', 'Leon', 'Lewis', 'Lisa', 'Louis', 'Luke', 'Madison', 'Marco', 'Marcus', 'Maria', 'Mark', 'Martin', 'Mary', 'Matthew', 'Max', 'Michael', 'Michelle', 'Morgan', 'Patrick', 'Paul', 'Peter', 'Prince', 'Rachel', 'Richard', 'River', 'Robert', 'Roman', 'Rose', 'Ruby', 'Russell', 'Ryan', 'Sarah', 'Scott', 'Sean', 'Simon', 'Stephen', 'Steven', 'Sullivan', 'Taylor', 'Thomas', 'Tyler', 'Victoria', 'Warren', 'William'"
                  "- Here are some optional sentence structure, but you must also generate sentences with other structures: "
                  "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to"
                  "Then, [B] and [A] had a lot of fun at the [PLACE]. [A] gave a [OBJECT] to"
                  "Then, [B] and [A] were working at the [PLACE]. [A] decided to give a [OBJECT] to"
                  "Then, [A] and [B] were thinking about going to the [PLACE]. [A] wanted to give a [OBJECT] to"
                  "Then, [A] and [B] had a long argument, and afterwards [A] said to"
                  "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to"
                  "When [A] and [B] got a [OBJECT] at the [PLACE], [A] decided to give it to"
                  "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to"
                  "While [A] and [B] were working at the [PLACE], [A] gave a [OBJECT] to"
                  "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to"
                  "After the lunch, [B] and [A] went to the [PLACE]. [A] gave a [OBJECT] to"
                  "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to"
                  "Then, [B] and [A] had a long argument. Afterwards [A] said to"
                  "The [PLACE] [B] and [A] went to had a [OBJECT]. [A] gave it to"
                  "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to"
                  "Then in the morning, [A] and [B] went to the [PLACE]. [A] gave a [OBJECT] to"
                  "Then in the morning, [A] and [B] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to"
                  "Then in the morning, [A] and [B] were working at the [PLACE]. [A] decided to give a [OBJECT] to"
                  "Then in the morning, [B] and [A] were thinking about going to the [PLACE]. [A] wanted to give a [OBJECT] to"
                  "Then in the morning, [A] and [B] had a long argument, and afterwards [B] said to"
                  "After taking a long break [A] and [B] went to the [PLACE], [A] gave a [OBJECT] to"
                  "When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [A] decided to give it to"
                  "When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [A] decided to give the [OBJECT] to"
                  "While spending time together [A] and [B] were working at the [PLACE], [A] gave a [OBJECT] to"
                  "While spending time together [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to"
                  "After the lunch in the afternoon, [B] and [A] went to the [PLACE]. [A] gave a [OBJECT] to"
                  "Afterwards, while spending time together [A] and [B] went to the [PLACE]. [B] gave a [OBJECT] to"
                  "Then in the morning afterwards, [B] and [A] had a long argument. Afterwards [A] said to"
                  "The local big [PLACE] [A] and [B] went to had a [OBJECT]. [B] gave it to"
                  "Friends separated at birth [A] and [B] found a [OBJECT] at the [PLACE]. [A] gave it to"
                  "Then, [A] and [B] went to the [PLACE]. [A] gave a [OBJECT] to"
                  "Then, [B] and [A] had a lot of fun at the [PLACE]. [A] gave a [OBJECT] to"
                  "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to"
                  "Then, [A] and [B] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to"
                  "Then, [B] and [A] had a long argument and after that [A] said to"
                  "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to"
                  "Afterwards, [B] and [A] went to the [PLACE]. [A] gave a [OBJECT] to"
                  "Then, [B] and [A] had a long argument. Afterwards [A] said to"
                  "Then [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to"
                  "Then [B] and [A] had a lot of fun at the [PLACE], and [A] gave a [OBJECT] to"
                  "Then [B] and [A] were working at the [PLACE], and [B] decided to give a [OBJECT] to"
                  "Then [B] and [A] were thinking about going to the [PLACE], and [A] wanted to give a [OBJECT] to"
                  "Then [B] and [A] had a long argument, and after that [B] said to"
                  "After the lunch [B] and [A] went to the [PLACE], and [A] gave a [OBJECT] to"
                  "Afterwards [A] and [B] went to the [PLACE], and [A] gave a [OBJECT] to"
                  "Then [B] and [A] had a long argument, and afterwards [A] said to"
                  "Example:\n"
                  "Answer:\n"
                  "1_test: When Victoria and Jane got a snack at the store , Jane decided to give it to\n"
                  "2_test: Then , Tom and James had a lot of fun at the park . Tom gave a present to\n"
                  "3_test: While Annie and Tony were working at the bank , Annie gave a hug to\n"
                  "4_test: Then , Felix and Sam had a long argument . Afterwards Sam said to\n"
                  "{\"1_test\":\"When Victoria and Jane got a snack at the store , Jane decided to give it to\", \"2_test\":\"Then , Tom and James had a lot of fun at the park . Tom gave a present to\", \"3_test\":\"While Annie and Tony were working at the bank , Annie gave a hug to\", \"4_test\":\"Then , Felix and Sam had a long argument . Afterwards Sam said to\"}\n"
                  )
            },
            {
                  "role": "user",
                  "content": (
                  f"Generate 100 sentences for Indirect Object Identification task. "
                  ),
            },
      ]
      
      example_message = await open_router.generate(messages=messages, output_dir=output_dir)
      print("example_message:", example_message.text)
      return example_message.text
async def main():
     
      output_dir = os.path.join("/home/chensiyuan/EAP-IG/results/ioi/hypothesis", "sentences")
      open_router = initialize_openrouter(model="gpt-4o")
      
      sentences_text = await generate_sentences(open_router, output_dir)
      sentences = extract_example_sentences(sentences_text)
      print("sentences:", sentences)
      # 将生成的句子保存到文件
      output_file = os.path.join(output_dir, "generated_sentences.json")
      with open(output_file, "w") as f:
            json.dump(sentences, f, indent=4)
      print(f"Generated sentences saved to {output_file}")

if __name__ == "__main__":
      asyncio.run(main())