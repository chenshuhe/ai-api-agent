/**
 * AI API Agent - 前端聊天界面
 * ==============================
 * 负责 Web UI 交互：对话展示、配置管理、流式接收 AI 回复。
 *
 * 核心功能模块：
 *   1. 对话管理   — 新建/切换/删除对话，对话列表渲染
 *   2. 消息收发   — SSE 流式接收 AI 回复，实时渲染
 *   3. 配置面板   — API 文档 URL、AI 模型、全局参数、请求场景
 *   4. 工具列表   — 展示可用 API 接口
 *   5. 快捷测试   — 🧪 按钮一键启动接口测试
 *
 * 通信协议：
 *   /api/chat 返回 SSE (Server-Sent Events) 流
 *   每条消息格式：data: {"chunk": "文本"} 或 data: {"done": true}
 */

// ========== DOM 缓存 ==========
// 频繁访问的元素提前获取引用，避免重复 querySelector
const msgs = document.getElementById('messages');
const input = document.getElementById('userInput');
const btn = document.getElementById('sendBtn');
let sending = false;          // 防止重复发送
let _currentConvId = '';      // 当前对话 ID

// ========== 工具函数 ==========

/** 展开/折叠侧边栏面板 */
function toggleSection(h) {
    h.querySelector('.arrow').classList.toggle('open');
    h.nextElementSibling.classList.toggle('open');
}

/** HTML 转义（防 XSS） */
function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** 右上角 Toast 通知 */
function toast(msg, type) {
    const t = document.createElement('div');
    t.className = 'toast ' + (type || 'success');
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2500);
}

/** 向聊天区添加一条消息气泡 */
function addMessage(role, text) {
    const div = document.createElement('div');
    div.className = 'message ' + role;  // user / assistant / system
    div.textContent = text;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;  // 自动滚到底部
    return div;
}

// ============================================================
// 对话管理
// ============================================================

/** 加载对话列表并渲染 */
async function loadConversations() {
    try {
        const r = await fetch('/api/conversations');
        const d = await r.json();
        _currentConvId = d.current_id;
        const list = document.getElementById('convList');
        if (!d.conversations.length) {
            list.innerHTML = '<div style="color:#666;font-size:0.75em;">暂无对话</div>';
            return;
        }
        // 渲染每个对话条目
        list.innerHTML = d.conversations.map(c => {
            const active = c.id === _currentConvId ? ' active' : '';
            const count = c.message_count ? ` (${c.message_count}条)` : '';
            return `<div class="conv-item${active}" onclick="switchConversation('${c.id}')" title="${esc(c.name)}">
                <span class="conv-name">${esc(c.name)}</span>
                <span class="conv-info">${count}</span>
                <span class="conv-del" onclick="event.stopPropagation();deleteConversation('${c.id}')" title="删除">✕</span>
            </div>`;
        }).join('');
        // 更新聊天区标题
        document.getElementById('convTitle').textContent =
            d.conversations.find(c => c.id === _currentConvId)?.name || '对话';
    } catch (e) {
        document.getElementById('convList').textContent = '加载失败';
    }
}

/** 创建新对话 */
async function newConversation() {
    const name = prompt('对话名称：', '新对话');
    if (!name) return;
    const r = await fetch('/api/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
    });
    const d = await r.json();
    if (d.status === 'ok') {
        msgs.innerHTML = '<div class="message assistant">新对话已创建，有什么可以帮你的？</div>';
        await Promise.all([loadConversations(), loadStatus(), loadTools()]);
        toast('已创建: ' + name);
    }
}

/** 切换到指定对话 */
async function switchConversation(id) {
    const r = await fetch('/api/conversations/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id }),
    });
    const d = await r.json();
    if (d.status === 'ok') {
        await Promise.all([loadConversations(), loadStatus(), loadTools()]);
        // 恢复该对话的消息历史
        msgs.innerHTML = '';
        const conv = d.conversation;
        if (conv.messages && conv.messages.length) {
            conv.messages.forEach(m => {
                if (m.content) addMessage(m.role === 'tool' ? 'system' : m.role, m.content);
            });
        } else {
            addMessage('assistant', '已切换到「' + esc(conv.name) + '」，有什么可以帮你的？');
        }
        document.getElementById('convTitle').textContent = conv.name;
    }
}

/** 删除对话 */
async function deleteConversation(id) {
    if (!confirm('确定删除该对话？')) return;
    const r = await fetch('/api/conversations/' + id, { method: 'DELETE' });
    const d = await r.json();
    if (d.status === 'ok') {
        await loadConversations();
        msgs.innerHTML = '<div class="message assistant">对话已删除，已切换到默认对话。</div>';
        toast('已删除');
    }
}

// ============================================================
// 状态 & 工具列表
// ============================================================

/** 加载运行状态 */
async function loadStatus() {
    try {
        const r = await fetch('/api/status');
        const d = await r.json();
        let html = `<span class="label">模型：</span>${d.model || '-'} (${d.provider || '-'})<br>`;
        html += `<span class="label">接口数：</span>${d.endpoints}<br>`;
        const statusText = d.ready ? '🟢 就绪' : d.loading ? '⏳ 加载中' : '🔴 未就绪';
        html += `<span class="label">状态：</span>${statusText}`;
        if (d.current_conv) html += `<br><span class="label">对话：</span>${esc(d.current_conv)}`;
        document.getElementById('status').innerHTML = html;
    } catch (e) {
        document.getElementById('status').textContent = '连接失败';
    }
}

/** 加载可用 API 工具列表 */
async function loadTools() {
    try {
        const r = await fetch('/api/tools');
        const d = await r.json();
        document.getElementById('toolCount').textContent = d.count;
        const list = document.getElementById('toolsList');
        if (!d.tools || !d.tools.length) {
            list.innerHTML = '<div style="color:#888;padding:6px;">暂无</div>';
            return;
        }
        list.innerHTML = d.tools.slice(0, 200).map(t =>
            `<div style="padding:5px 6px;margin-bottom:3px;background:#1a1a2e;border-radius:3px;border-left:3px solid #e94560;">
                <span style="color:#e94560;font-weight:bold;">${esc(t.name)}</span>
                <div style="color:#aaa;margin-top:1px;font-size:0.85em;">${esc(t.description).substring(0, 80)}</div>
            </div>`
        ).join('');
        if (d.tools.length > 200)
            list.innerHTML += `<div style="color:#888;padding:4px;">...还有 ${d.tools.length - 200} 个</div>`;
    } catch (e) {
        document.getElementById('toolsList').innerHTML = '<div style="color:#888;">加载失败</div>';
    }
}

// ============================================================
// 配置管理
// ============================================================

let _urls = [], _params = [], _scenarios = [];

/** 加载所有配置到表单 */
async function loadConfig() {
    try {
        const r = await fetch('/api/config');
        const d = await r.json();
        // API 文档 URL
        _urls = d.api_docs?.urls || [];
        renderUrlList();
        // 请求场景
        _scenarios = d.api_scenarios?.list || [];
        renderScenarios();
        // AI 模型
        const m = d.model || {};
        document.getElementById('modelProvider').value = m.provider || 'openai';
        document.getElementById('modelName').value = m.name || '';
        document.getElementById('modelBaseUrl').value = m.base_url || '';
        document.getElementById('modelApiKey').value = m.api_key || '';
        document.getElementById('modelTemp').value = m.temperature ?? 0.1;
        document.getElementById('modelMaxTokens').value = m.max_tokens ?? 4096;
        // 全局参数
        _params = d.global_params || [];
        renderParamTable();
        // Project settings
        document.getElementById('projectDir').value = d.project_dir || '';
        const al = d.auto_login || {};
        document.getElementById('autoLoginHeader').value = al.header_name || 'X-Dts-Admin-Token';
        document.getElementById('autoLoginHint').value = al.login_hint || '手机号';
        document.getElementById('autoLoginEnabled').checked = al.enabled !== false;
    } catch (e) {
        console.error('Config load error', e);
    }
}

async function saveProjectSettings() {
    const project_dir = document.getElementById('projectDir').value.trim();
    const auto_login = {
        enabled: document.getElementById('autoLoginEnabled').checked,
        header_name: document.getElementById('autoLoginHeader').value.trim() || 'X-Dts-Admin-Token',
        login_hint: document.getElementById('autoLoginHint').value.trim() || '手机号',
    };
    await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_dir, auto_login }),
    });
    toast('项目设置已保存');
    await fetch('/api/reload', { method: 'POST' });
    await loadStatus();
}

// ---- API 文档 URL ----

function renderUrlList() {
    const list = document.getElementById('urlList');
    if (!_urls.length) {
        list.innerHTML = '<div style="color:#666;font-size:0.72em;">暂无</div>';
        return;
    }
    list.innerHTML = _urls.map((u, i) =>
        `<div class="url-item"><span title="${esc(u)}">${esc(u)}</span><button onclick="_urls.splice(${i},1);renderUrlList();saveApiDocs()">✕</button></div>`
    ).join('');
}

function addUrl() {
    const v = document.getElementById('newUrl').value.trim();
    if (!v || _urls.includes(v)) return;
    _urls.push(v);
    renderUrlList();
    document.getElementById('newUrl').value = '';
    saveApiDocs();
}

async function saveApiDocs() {
    await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_docs: { urls: _urls, timeout: 30 } }),
    });
    toast('已保存');
}

/** 保存 AI 模型配置 */
async function saveModelConfig() {
    const model = {
        provider: document.getElementById('modelProvider').value,
        name: document.getElementById('modelName').value,
        base_url: document.getElementById('modelBaseUrl').value,
        api_key: document.getElementById('modelApiKey').value,
        temperature: parseFloat(document.getElementById('modelTemp').value) || 0.1,
        max_tokens: parseInt(document.getElementById('modelMaxTokens').value) || 4096,
    };
    await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model }),
    });
    toast('模型配置已保存');
}

// ---- 全局参数 ----

function renderParamTable() {
    const tbody = document.getElementById('paramsTbody');
    if (!_params.length) {
        tbody.innerHTML = '<tr><td colspan="4" style="color:#666;">暂无</td></tr>';
        return;
    }
    tbody.innerHTML = _params.map((p, i) =>
        `<tr>
            <td><input value="${esc(p.name || '')}" onchange="_params[${i}].name=this.value"></td>
            <td><input value="${esc(p.value || '')}" onchange="_params[${i}].value=this.value"></td>
            <td><select onchange="_params[${i}].type=this.value">
                <option value="header" ${p.type === 'header' ? 'selected' : ''}>Header</option>
                <option value="query" ${p.type === 'query' ? 'selected' : ''}>Query</option>
            </select></td>
            <td><button class="btn-del" onclick="_params.splice(${i},1);renderParamTable()">✕</button></td>
        </tr>`
    ).join('');
}

function addParam() {
    _params.push({ name: '', value: '', type: 'header' });
    renderParamTable();
}

async function saveGlobalParams() {
    await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ global_params: _params }),
    });
    toast('全局参数已保存');
}

// ---- 请求场景（环境切换）----

function renderScenarios() {
    const sel = document.getElementById('scenarioSelect');
    if (!_scenarios.length) {
        sel.innerHTML = '<option>暂无场景</option>';
        return;
    }
    sel.innerHTML = _scenarios.map(s =>
        `<option value="${s.name}" ${s.name === (_scenarios._active || 'default') ? 'selected' : ''}>${esc(s.name)} - ${esc(s.description || '')}</option>`
    ).join('');
    // 显示当前激活场景的 URL 映射
    const activeScenario = _scenarios.find(s => s.name === (_scenarios._active || 'default')) || _scenarios[0];
    document.getElementById('scenarioMapping').value = JSON.stringify(activeScenario?.mapping || {}, null, 2);
}

/** 切换激活场景 */
async function switchScenario() {
    const name = document.getElementById('scenarioSelect').value;
    const scenario = _scenarios.find(s => s.name === name);
    if (!scenario) return;
    const updated = { active: name, list: _scenarios };
    await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_scenarios: updated }),
    });
    _scenarios._active = name;
    document.getElementById('scenarioMapping').value = JSON.stringify(scenario.mapping || {}, null, 2);
    toast('已切换到: ' + name);
    // 切换场景后重新加载 Agent（应用新的 URL 映射）
    await fetch('/api/reload', { method: 'POST' });
    await loadStatus();
}

/** 保存当前场景的 URL 映射 */
async function saveScenarioMapping() {
    const name = document.getElementById('scenarioSelect').value;
    let mapping = {};
    try {
        mapping = JSON.parse(document.getElementById('scenarioMapping').value);
    } catch (e) {
        toast('JSON 格式错误', 'error');
        return;
    }
    const scenario = _scenarios.find(s => s.name === name);
    if (!scenario) return;
    scenario.mapping = mapping;
    const updated = { active: _scenarios._active || 'default', list: _scenarios };
    await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_scenarios: updated }),
    });
    toast('URL 映射已保存');
    await fetch('/api/reload', { method: 'POST' });
    await loadStatus();
}

// ============================================================
// 重新加载
// ============================================================

async function reloadDocs() {
    const rb = document.getElementById('reloadBtn');
    rb.disabled = true;
    rb.textContent = '加载中...';
    document.getElementById('loadError').style.display = 'none';
    try {
        const r = await fetch('/api/reload', { method: 'POST' });
        const d = await r.json();
        if (d.load_error) {
            document.getElementById('loadError').style.display = 'block';
            document.getElementById('loadError').textContent = d.load_error;
            toast('加载失败', 'error');
        } else {
            toast(`加载成功，${d.endpoints} 个接口`);
        }
        await Promise.all([loadStatus(), loadTools(), loadConversations()]);
    } catch (e) {
        toast('加载失败', 'error');
    }
    rb.disabled = false;
    rb.textContent = '重新加载 API 文档';
}

// ============================================================
// 消息收发（核心交互）
// ============================================================

/**
 * 发送消息并流式接收 AI 回复。
 *
 * 流程：
 *   1. POST /api/chat 携带用户消息
 *   2. 读取 Response body 的 ReadableStream
 *   3. 逐行解析 SSE data: 字段
 *   4. 实时渲染到聊天区
 *   5. 收到 done 信号后刷新对话列表
 */
async function sendMessage() {
    if (sending) return;  // 防止重复发送
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    sending = true;
    btn.disabled = true;

    // 用户消息气泡
    addMessage('user', text);
    // AI 回复气泡（初始为空，带加载动画）
    const assistantDiv = addMessage('assistant', '');
    const loadingSpan = document.createElement('span');
    loadingSpan.className = 'loading';
    assistantDiv.appendChild(loadingSpan);

    let fullText = '';
    try {
        const r = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text }),
        });

        // 使用 ReadableStream 读取 SSE 流
        const reader = r.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            // SSE 可能在一个 chunk 中包含多行
            const lines = decoder.decode(value).split('\n');
            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const data = JSON.parse(line.slice(6));  // 去掉 "data: " 前缀
                    if (data.chunk) {
                        // 收到文本片段 → 移除加载动画，追加内容
                        if (assistantDiv.contains(loadingSpan)) assistantDiv.removeChild(loadingSpan);
                        fullText += data.chunk;
                        assistantDiv.textContent = fullText;
                        msgs.scrollTop = msgs.scrollHeight;
                    }
                    if (data.error) {
                        assistantDiv.textContent = '出错了：' + data.error;
                    }
                } catch (e) {
                    // 空行或非 JSON 行，忽略
                }
            }
        }
    } catch (e) {
        assistantDiv.textContent = '连接失败：' + e.message;
    }
    sending = false;
    btn.disabled = false;
    input.focus();
    // 刷新对话列表（更新消息计数）
    loadConversations();
}

/** 清除当前对话 */
async function clearChat() {
    try {
        await fetch('/api/clear', { method: 'POST' });
    } catch (e) {}
    msgs.innerHTML = '<div class="message assistant">对话已清除。</div>';
    loadConversations();
}

/** 🧪 快捷测试按钮 */
async function quickTest() {
    const feature = prompt('输入要测试的功能名称，例如：\n- 资讯评价\n- 商品管理\n- 用户地址\n- 优惠券', '资讯评价');
    if (!feature) return;
    input.value = '帮我测试' + feature + '功能';
    sendMessage();
}

// ============================================================
// 初始化
// ============================================================

// 页面加载时立即获取所有数据
loadStatus();
loadTools();
loadConfig();
loadConversations();
input.focus();  // 自动聚焦输入框
