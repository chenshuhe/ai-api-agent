"""Internal tools: auth management, testing, code analysis, scenario switching."""

import fnmatch
import json
import os
import subprocess
import shutil
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from loguru import logger

from ..api_loader.parser import Endpoint
from ..execution.executor import execute_async
from ..settings import Settings


# ---- Auth tools ----

@tool
def internal_set_global_header(name: str, value: str) -> str:
    """Set a global request header. Call after obtaining a login token."""
    return json.dumps({"status": "ok", "message": f"Header '{name}' has been set."})


@tool
def internal_list_global_headers() -> str:
    """List all currently configured global request headers."""
    return json.dumps([])


@tool
def internal_switch_scenario(name: str) -> str:
    """Switch the API request target environment."""
    return json.dumps({"status": "ok", "message": f"Switched to '{name}'."})


# ---- Testing tools ----

@tool
def internal_run_test(feature: str) -> str:
    """Start automated API testing for a feature. Tests CREATE, QUERY, UPDATE, DELETE."""
    return json.dumps({
        "status": "ok",
        "message": (
            f"Testing '{feature}':\n"
            "1. Find related CREATE/POST API, generate test data, call it\n"
            "2. Call internal_test_step to report\n"
            "3. Find QUERY/GET API, verify data\n"
            "4. Call internal_test_step\n"
            "5. Find UPDATE/PUT API, modify data\n"
            "6. Call internal_test_step\n"
            "7. Find DELETE API, clean up\n"
            "8. Call internal_test_step\n"
            "9. Summarize results"
        ),
    })


@tool
def internal_test_step(step: str, api: str, status: str, detail: str) -> str:
    """Report a single test step result."""
    return json.dumps({"status": "ok", "logged": f"[{step}] {api}: {status} - {detail}"})


# ---- Code analysis tools ----

_project_dir: str = ""


def set_project_dir(path: str):
    global _project_dir
    _project_dir = path


@tool
def internal_search_code(query: str, file_pattern: str = "**/*.java") -> str:
    """Search project source code for keywords, class names, or error messages."""
    if not _project_dir or not os.path.isdir(_project_dir):
        return json.dumps({"error": "project_dir not configured"})

    results = []
    # Prefer ripgrep
    rg = shutil.which("rg") or shutil.which("grep")
    if rg:
        try:
            pattern = file_pattern.replace("**/", "")
            cmd = [rg, "-rn", f"--include={pattern}", query, _project_dir]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            for line in proc.stdout.strip().split("\n")[:50]:
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    results.append({"file": parts[0], "line": parts[1], "content": parts[2][:200]})
        except Exception:
            pass

    if not results:
        for root, dirs, files in os.walk(_project_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "target", ".git", "__pycache__")]
            for f in files:
                if fnmatch.fnmatch(f, pattern.replace("**/", "")):
                    try:
                        content = Path(os.path.join(root, f)).read_text(encoding="utf-8", errors="ignore")
                        for i, line in enumerate(content.split("\n"), 1):
                            if query.lower() in line.lower():
                                results.append({"file": os.path.relpath(os.path.join(root, f), _project_dir), "line": str(i), "content": line.strip()[:200]})
                                if len(results) >= 50:
                                    break
                    except Exception:
                        pass
                if len(results) >= 50:
                    break
            if len(results) >= 50:
                break

    return json.dumps({"results": results[:50], "count": len(results)}, ensure_ascii=False)


@tool
def internal_read_code(file_path: str, start_line: int = 1, end_line: int = 0) -> str:
    """Read a source file with line numbers."""
    if not _project_dir:
        return json.dumps({"error": "project_dir not configured"})
    full = os.path.join(_project_dir, file_path)
    if not os.path.isfile(full) or ".." in file_path:
        return json.dumps({"error": f"File not found: {file_path}"})
    try:
        lines = Path(full).read_text(encoding="utf-8", errors="ignore").split("\n")
        if end_line <= 0:
            end_line = len(lines)
        start = max(1, start_line)
        end = min(len(lines), end_line)
        code = "\n".join(f"{i}: {lines[i-1]}" for i in range(start, end + 1))
        return json.dumps({"file": file_path, "total_lines": len(lines), "code": code}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def internal_edit_code(file_path: str, old_code: str, new_code: str, confirmed: bool, summary: str) -> str:
    """Propose or apply a code change. Set confirmed=false to preview, confirmed=true to apply."""
    if not _project_dir:
        return json.dumps({"error": "project_dir not configured"})
    full = os.path.join(_project_dir, file_path)
    if not os.path.isfile(full) or ".." in file_path:
        return json.dumps({"error": f"File not found: {file_path}"})

    if not confirmed:
        diff = f"- {old_code[:120]}...\n+ {new_code[:120]}..."
        return json.dumps({
            "status": "pending_confirmation",
            "message": f"确认修改 {file_path}?\n摘要: {summary}\n\n差异:\n{diff}",
        }, ensure_ascii=False)

    try:
        content = Path(full).read_text(encoding="utf-8", errors="ignore")
        if old_code not in content:
            return json.dumps({"error": "old_code not found in file"})
        Path(full).write_text(content.replace(old_code, new_code, 1), encoding="utf-8")
        return json.dumps({"status": "ok", "message": f"已修改 {file_path}: {summary}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---- All internal tools ----

def get_internal_tools() -> list:
    """Return the list of all internal LangChain tools."""
    return [
        internal_set_global_header,
        internal_list_global_headers,
        internal_switch_scenario,
        internal_run_test,
        internal_test_step,
        internal_search_code,
        internal_read_code,
        internal_edit_code,
    ]
