import json
import httpx
import os
from asyncio import sleep

class Response:
    def __init__(self, response):
        self.text = response

class OpenRouter:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url="https://yeysai.com/v1/chat/completions",
    ):
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.url = base_url
        self.client = httpx.AsyncClient()

    def postprocess(self, response):
        response_json = response.json()
        msg = response_json["choices"][0]["message"]["content"]
        return Response(msg)

    async def generate(  # type: ignore
        self, messages: str, output_dir: str, raw: bool = False, max_retries: int = 20, **kwargs  # type: ignore
    ) -> Response:  # type: ignore
        kwargs.pop("schema", None)
        max_tokens = kwargs.pop("max_tokens", 500)
        temperature = kwargs.pop("temperature", 0.3)
        
        # 请求数据
        data = {
            "model": self.model, 
            "messages": messages,
            "temperature": temperature,
            #"max_tokens": max_tokens,
        }
        #print("Request data:", data)

        os.makedirs(output_dir, exist_ok=True)

        # 构造 prompt 和 answer 文件路径
        prompt_file = os.path.join(output_dir, "prompt.txt")
        answer_file = os.path.join(output_dir, "answer.txt")
        # 将请求的 prompt 写入文件
        with open(prompt_file, "a") as f:
            f.write(str(messages) + "\n")

        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    url=self.url, json=data, headers=self.headers, timeout=90
                )
                #print("Response status code:", response.status_code)
                #print("Response content:", response.content)
                if raw:
                    return response.json()
                result = self.postprocess(response)

                with open(answer_file, "a") as f:
                    f.write(str(result.text) + "\n")
            
                return result

            except json.JSONDecodeError:
                print(f"Attempt {attempt + 1}: Invalid JSON response, retrying...")
            except Exception as e:
                print(f"Attempt {attempt + 1}: {str(e)}, retrying...")

            await sleep(1)

        raise RuntimeError("Failed to generate text after multiple attempts.")
