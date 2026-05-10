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
        base_url = "https://api.key77qiqi.cn/v1/chat/completions"
        # base_url="https://yeysai.com/v1/chat/completions",
    ):
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.url = base_url
        self.client = httpx.AsyncClient()

    def postprocess(self, response_json):
        # 假设 response_json 已经是经过验证的成功响应
        msg = response_json["choices"][0]["message"]["content"]
        return Response(msg)

    async def generate(  
        self, messages: str, output_dir: str, raw: bool = False, max_retries: int = 8, **kwargs  # type: ignore
    ) -> Response:  
        kwargs.pop("schema", None)
        max_tokens = kwargs.pop("max_tokens", 500)
        temperature = kwargs.pop("temperature", 0.3)
        
        data = {
            "model": self.model, 
            "messages": messages,
            "temperature": temperature,
        }

        os.makedirs(output_dir, exist_ok=True)
        prompt_file = os.path.join(output_dir, "prompt.txt")
        answer_file = os.path.join(output_dir, "answer.txt")
        raw_response_log_file = os.path.join(output_dir, "raw_api_responses.jsonl")
        
        with open(prompt_file, "a") as f:
            # 只记录 user content，避免日志过长
            user_content = next((msg['content'] for msg in messages if msg['role'] == 'user'), '{}')
            f.write(f"--- PROMPT ---\n{user_content}\n\n")

        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    url=self.url, json=data, headers=self.headers, timeout=90
                )
                
                # 增加详细的错误检查
                response.raise_for_status()  # 如果状态码不是 2xx，则会抛出 HTTPStatusError

                response_json = response.json()

                with open(raw_response_log_file, "a") as log_f:
                    log_entry = {
                        "prompt_user_content": next((msg['content'] for msg in messages if msg['role'] == 'user'), '{}'),
                        "raw_response": response_json
                    }
                    log_f.write(json.dumps(log_entry) + "\n")
                if "error" in response_json:
                    error_message = response_json["error"].get("message", "Unknown error from API")

                if raw:
                    return response.json()
                
                result = self.postprocess(response_json)

                with open(answer_file, "a") as f:
                    f.write(f"--- RESPONSE ---\n{result.text}\n\n")
            
                return result

            except httpx.HTTPStatusError as e:
                # 捕获HTTP状态错误（如 401, 429, 500）
                print(f"Attempt {attempt + 1}: HTTP Error {e.response.status_code} - {e.response.text}. Retrying...")
            except json.JSONDecodeError:
                print(f"Attempt {attempt + 1}: Invalid JSON response from server. Retrying...")
            except httpx.RequestError as e:
                print(
                    f"Attempt {attempt + 1}: RequestError ({e.__class__.__name__}): {repr(e)}. Retrying..."
                )
            except Exception as e:
                # 捕获其他所有错误，比如之前的 KeyError
                print(
                    f"Attempt {attempt + 1}: An unexpected error occurred ({e.__class__.__name__}): {repr(e)}. Retrying..."
                )

            await sleep(attempt + 1) # 增加重试的等待时间

        raise RuntimeError("Failed to generate text after multiple attempts.")
