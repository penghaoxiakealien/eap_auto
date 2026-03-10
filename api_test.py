import asyncio
from api import OpenRouter

async def main():
    api_key = "sk-MjSyxJuoVSlripXy2933FaBaEaBb4fC1A4B564DfB699B8C2"  # 替换为你的实际 API 密钥
    open_router = OpenRouter(model="gpt-4o", api_key=api_key)
    
    prompt = "Hi, How are you?"
    response = await open_router.generate(prompt)
    
    print(response.text)

# 运行异步主函数
asyncio.run(main())
