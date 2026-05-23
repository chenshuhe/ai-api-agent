# AI API Agent

基于大语言模型的智能 API 代理，将 OpenAPI 文档转换为 LLM 可调用的函数工具，让你用自然语言与后端 API 交互。

An LLM-powered API agent that converts OpenAPI specs into function-calling tools — interact with backend APIs using natural language.

---

## 功能特性

- **自然语言调用 API** — 描述需求，AI 自动选择合适的 API 并传参调用
- **OpenAPI 文档加载** — 支持多后端服务的 Swagger/OpenAPI 3.x JSON 文档
- **多环境切换** — 支持 default / production 等多套环境，自动替换请求地址
- **自动登录** — AI 自动调用登录接口，提取 token 并注入后续请求
- **Web UI + CLI 双模式** — 浏览器聊天界面 & 命令行交互
- **多会话管理** — 创建/切换/删除对话，每会话独立上下文
- **流式输出** — SSE 实时推送 AI 回复
- **多模型支持** — OpenAI / Anthropic Claude / Ollama
- **代码分析** — 可选关联 Java 项目源码，API 报错时 AI 可检索相关代码

## 技术栈

| 层级 | 技术 |
|---|---|
| 语言 | Python 3.10+ |
| Web 框架 | FastAPI + Uvicorn |
| AI 提供商 | OpenAI 兼容接口 / Anthropic / Ollama |
| HTTP 客户端 | httpx (async) |
| 配置 | YAML + 环境变量插值 |
| 前端 | Vanilla JS + HTML + CSS |

## 快速开始

### 1. 克隆项目

```bash
git clone <repo-url>
cd agent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置

```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml`，设置：
- `model.api_key` — LLM API 密钥（支持 `${DEEPSEEK_API_KEY}` 环境变量）
- `api_docs.urls` — 后端 OpenAPI 文档地址
- `global_params` — 全局请求头（如认证 token）

也可通过环境变量注入敏感信息：

```bash
export DEEPSEEK_API_KEY=sk-your-key-here
export AUTH_TOKEN=your-jwt-token
```

### 4. 启动

**Web 模式**（推荐）：

```bash
python app.py
```

浏览器打开 `http://localhost:8000`，在聊天框中输入自然语言指令。

**CLI 模式**：

```bash
python main.py
```

### 5. 使用示例

```
你 > 帮我查一下用户手机号 138xxxx1234 的订单列表
AI > 正在调用 /api/shop/order/list ...（展示订单列表）

你 > 创建一个测试商品，名称叫"测试道具"，价格 100 金币
AI > 正在调用 /api/shop/product/create ... 创建成功，商品 ID：12345
```

## 项目结构

```
agent/
├── app.py                  # FastAPI Web 服务入口
├── main.py                 # CLI 入口
├── config.yaml             # 配置文件（不提交 git）
├── config.example.yaml     # 配置模板
├── requirements.txt        # Python 依赖
├── src/
│   ├── agent.py            # 核心 Agent：工具调用循环
│   ├── ai_client.py        # LLM 抽象层
│   ├── api_docs.py         # OpenAPI 文档解析
│   ├── api_executor.py     # HTTP 请求构建与执行
│   ├── config.py           # 配置加载（YAML + 环境变量）
│   ├── conversation_store.py # 对话持久化
│   └── tool_converter.py   # Endpoint → LLM tool 转换
└── web/
    ├── index.html          # 前端聊天界面
    └── app.js              # 前端逻辑
```

## 配置说明

| 配置项 | 说明 |
|---|---|
| `model.provider` | LLM 提供商：`openai` / `anthropic` / `ollama` |
| `model.api_key` | API 密钥，支持 `${ENV_VAR}` 格式 |
| `api_docs.urls` | 后端 OpenAPI 文档 URL 列表 |
| `auto_login` | 自动登录：header 名称、提示文字、开关 |
| `api_scenarios` | 多环境 URL 映射 |
| `global_params` | 附加到所有请求的全局 header/query |
| `project_dir` | （可选）Java 源码目录，用于错误分析 |

## License

MIT
