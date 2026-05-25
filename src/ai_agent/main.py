#!/usr/bin/env python3
"""CLI entry point for the AI API Agent v2."""

import asyncio
import sys

from .agent.graph import ApiAgent
from .settings import get_settings, setup_logging


async def main_async():
    setup_logging()
    settings = get_settings()

    print("=" * 60)
    print("AI API Agent v2 - LangGraph 驱动")
    print("=" * 60)

    agent = ApiAgent(settings)
    count = await agent.load_endpoints()
    agent.build_graph()
    print(f"\n模型: {settings.model.name} | 端点: {count} | 场景: {settings.api_scenarios.active}")
    print("输入 /help 查看命令，/quit 退出\n")

    while True:
        try:
            user_input = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("再见！")
            break
        if user_input.lower() == "/help":
            print("命令: /help /tools /clear /quit\n直接输入自然语言即可调用 API。\n")
            continue
        if user_input.lower() == "/clear":
            agent.clear_history()
            print("对话已清除。")
            continue

        print("\nAI > ", end="", flush=True)
        try:
            async for chunk in agent.chat_stream(user_input):
                print(chunk, end="", flush=True)
            print()
        except Exception as e:
            print(f"\n[错误: {e}]")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
