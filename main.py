#!/usr/bin/env python3
"""
CLI 入口
========
命令行交互模式的启动脚本。

用法：
  python main.py

交互命令：
  /help    - 显示帮助
  /tools   - 列出可用 API 工具
  /clear   - 清除对话历史
  /quit    - 退出

与 Web 模式的区别：
  - CLI 使用 chat()（非流式）获取完整回复
  - 不保存对话历史到磁盘（每次运行独立）
  - 不提供配置管理功能（直接编辑 config.yaml）
"""

import asyncio
import sys

from src.api_docs import load_all_api_docs, get_all_endpoints
from src.agent import ApiAgent
from src.config import get_config


async def main_async():
    """异步主函数"""
    config = get_config()

    print("=" * 60)
    print("AI API Agent - 对话式 API 调用")
    print("=" * 60)

    # 加载 API 文档
    print(f"\n正在从 {len(config.api_docs_urls)} 个文档源加载 API 文档...")
    docs = await load_all_api_docs(config.api_docs_urls, config.api_docs_timeout)
    if not docs:
        print("错误：无法加载任何 API 文档，请检查 config.yaml 配置")
        sys.exit(1)

    endpoints = get_all_endpoints(docs)
    print(f"共发现 {len(endpoints)} 个 API 端点")

    # 初始化 Agent
    agent = ApiAgent(config)
    agent.load_tools(endpoints)
    print(agent.get_tool_summary())

    print(f"\n模型：{config.model['name']} ({config.model['provider']})")
    print("输入 /help 查看命令，/quit 退出\n")

    # 对话循环
    while True:
        try:
            user_input = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 处理命令
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("再见！")
            break
        if user_input.lower() == "/help":
            print("""
命令：
  /help       - 显示此帮助
  /tools      - 列出可用 API 工具
  /clear      - 清除对话历史
  /quit       - 退出

直接用自然语言描述你的需求，AI 会自动调用对应的 API 接口。
""")
            continue
        if user_input.lower() == "/tools":
            print(agent.get_tool_summary())
            continue
        if user_input.lower() == "/clear":
            agent.clear_history()
            print("对话历史已清除。")
            continue

        # 正常对话：调用 Agent 的 chat 方法
        print("\nAI > ", end="", flush=True)
        try:
            full = ""
            # CLI 模式使用流式输出（逐 token 显示）
            async for chunk in agent.chat_stream(user_input):
                print(chunk, end="", flush=True)
                full += chunk
            print()
        except Exception as e:
            print(f"\n[错误：{e}]")


def main():
    """入口函数"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
