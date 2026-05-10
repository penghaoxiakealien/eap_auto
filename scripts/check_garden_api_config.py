import argparse
import asyncio
import json
import os
import sys

REPO_ROOT = "/home/wangziran/eap_auto"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from api import OpenRouter


DEFAULT_BASE_URL = "https://api.key77qiqi.cn/v1/chat/completions"
DEFAULT_API_KEY = "sk-cdENMjpwVIpdd1Iv0auFiHizYdgnWM0ZFKhHN3UBYqKIoqpA"
DEFAULT_MODEL = "gpt-5.2-2025-12-11"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Check the API config used by garden auto_terminal.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="/tmp/check_garden_api_config")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    client = OpenRouter(model=args.model, api_key=args.api_key, base_url=args.base_url)

    messages = [
        {"role": "system", "content": "Reply with exactly OK."},
        {"role": "user", "content": "health check"},
    ]

    print("CONFIG")
    print(json.dumps({
        "base_url": args.base_url,
        "api_key_prefix": args.api_key[:8],
        "model": args.model,
        "output_dir": args.output_dir,
    }, ensure_ascii=False, indent=2))

    try:
        result = await client.generate(
            messages=messages,
            output_dir=args.output_dir,
            max_retries=1,
            temperature=0.0,
        )
        print("\nRESULT")
        print(result.text)
    finally:
        await client.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
