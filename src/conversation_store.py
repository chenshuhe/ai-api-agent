"""
对话存储模块
============
管理多会话的持久化：每个对话保存为一个 JSON 文件在 conversations/ 目录下。

数据模型：
  Conversation {
    id:      唯一的 12 位 hex 标识
    name:    对话名称（用户可自定义）
    created_at: ISO 时间戳
    messages: 对话历史 [{"role": "...", "content": "...", "tool_call_id": "..."}]
    global_params: 此对话的全局参数（token 等）
  }

文件结构：
  conversations/
    ├── a1b2c3d4e5f6.json
    ├── 7890abcdef12.json
    └── ...

设计要点：
  - 每个对话独立存储，互不干扰
  - 全局参数跟随对话（对话 A 的 token 不会泄露到对话 B）
  - 服务重启后对话历史完整保留
  - 默认对话自动创建（首次使用时）
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

CONV_DIR = Path("conversations")


def _ensure_dir():
    """确保对话存储目录存在"""
    CONV_DIR.mkdir(parents=True, exist_ok=True)


def _conv_path(conv_id: str) -> Path:
    """对话 ID → 文件路径"""
    return CONV_DIR / f"{conv_id}.json"


class Conversation:
    """
    一个对话会话。

    属性：
      id            - 唯一标识（12位 hex）
      name          - 用户定义的名称
      created_at    - 创建时间（ISO 格式）
      messages      - 消息列表（不含系统提示词）
      global_params - 此对话专属的全局参数
    """

    def __init__(self, conv_id: str, name: str, created_at: str = "",
                 messages: list[dict] | None = None, global_params: list[dict] | None = None):
        self.id = conv_id
        self.name = name
        self.created_at = created_at or datetime.now().isoformat()
        self.messages = messages or []
        self.global_params = global_params or []

    def to_dict(self) -> dict:
        """序列化为字典（用于 JSON 存储和 API 传输）"""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "messages": self.messages,
            "global_params": self.global_params,
        }

    @staticmethod
    def from_dict(d: dict) -> "Conversation":
        """从字典反序列化"""
        return Conversation(
            conv_id=d["id"],
            name=d["name"],
            created_at=d.get("created_at", ""),
            messages=d.get("messages", []),
            global_params=d.get("global_params", []),
        )


class ConversationStore:
    """
    对话存储管理器。

    负责：
      - 列出所有对话
      - 创建 / 切换 / 删除 / 重命名对话
      - 保存对话状态（agent 每次回复后调用）
    """

    def __init__(self):
        _ensure_dir()

    def list_all(self) -> list[dict]:
        """
        列出所有对话（摘要信息，不含完整消息）。

        按修改时间倒序排列（最近使用的在前）。
        """
        result = []
        for f in sorted(CONV_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                result.append({
                    "id": d["id"],
                    "name": d["name"],
                    "created_at": d.get("created_at", ""),
                    "message_count": len(d.get("messages", [])),
                    "has_params": len(d.get("global_params", [])) > 0,  # 是否配置了 token 等
                })
            except Exception:
                pass  # 损坏的文件跳过
        return result

    def get(self, conv_id: str) -> Conversation | None:
        """根据 ID 获取完整对话"""
        path = _conv_path(conv_id)
        if not path.exists():
            return None
        return Conversation.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, conv: Conversation):
        """
        保存对话到磁盘。

        每次 AI 回复后调用，确保对话历史不丢失。
        """
        _ensure_dir()
        data = conv.to_dict()
        _conv_path(conv.id).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def create(self, name: str) -> Conversation:
        """创建新对话（生成唯一 ID 并保存空文件）"""
        conv = Conversation(conv_id=uuid.uuid4().hex[:12], name=name)
        self.save(conv)
        return conv

    def delete(self, conv_id: str) -> bool:
        """删除对话文件"""
        path = _conv_path(conv_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def get_or_create_default(self) -> Conversation:
        """
        获取最近使用的对话；如果没有则创建默认对话。
        用于服务首次启动或所有对话被删除后的回退。
        """
        existing = self.list_all()
        if existing:
            return self.get(existing[0]["id"])
        return self.create("默认对话")
