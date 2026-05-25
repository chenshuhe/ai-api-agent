"""
Application settings.

Uses pydantic-settings for type-safe configuration loaded from config.yaml.
Environment variables override YAML values (e.g. OPENAI_API_KEY).
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${VAR} and ${VAR:-default} in config values."""
    if isinstance(value, str):
        def replace(m: re.Match) -> str:
            expr = m.group(1)
            if ":-" in expr:
                var, default = expr.split(":-", 1)
                return os.environ.get(var.strip(), default.strip())
            return os.environ.get(expr.strip(), "")
        return re.sub(r"\$\{([^}]+)\}", replace, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


class ModelSettings(BaseModel):
    """AI model configuration."""
    provider: str = "openai"
    name: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 4096


class ApiDocsSettings(BaseModel):
    """API documentation sources."""
    urls: list[str] = Field(default_factory=list)
    timeout: int = 30


class ApiAuthSettings(BaseModel):
    """Backend API authentication."""
    type: str = "none"  # none, bearer, basic, api_key
    token: str = ""
    username: str = ""
    password: str = ""
    key_name: str = "X-API-Key"


class ScenarioConfig(BaseModel):
    """A named request scenario with URL mapping."""
    name: str = "default"
    description: str = ""
    mapping: dict[str, str] = Field(default_factory=dict)


class ScenarioSettings(BaseModel):
    """Request scenarios (environment switching)."""
    active: str = "default"
    scenarios: list[dict] = Field(default_factory=list)  # list of ScenarioConfig-compatible dicts

    def get_active_mapping(self) -> dict[str, str]:
        for s in self.scenarios:
            if isinstance(s, dict):
                if s.get("name") == self.active:
                    return s.get("mapping", {})
            elif hasattr(s, "name") and s.name == self.active:
                return s.mapping
        return {}


class AutoLoginSettings(BaseModel):
    """Auto-login configuration."""
    enabled: bool = True
    header_name: str = "X-Dts-Admin-Token"
    login_hint: str = "手机号"


class ServerSettings(BaseModel):
    """Web server settings."""
    host: str = "0.0.0.0"
    port: int = 8000


class GlobalParam(BaseModel):
    """A single global request parameter."""
    name: str
    value: str
    type: str = "header"  # header or query


class Settings(BaseSettings):
    """Root application settings loaded from config.yaml."""
    model_config = SettingsConfigDict(env_prefix="AI_AGENT_")

    model: ModelSettings = Field(default_factory=ModelSettings)
    api_docs: ApiDocsSettings = Field(default_factory=ApiDocsSettings)
    api_auth: ApiAuthSettings = Field(default_factory=ApiAuthSettings)
    api_scenarios: ScenarioSettings = Field(default_factory=ScenarioSettings)
    auto_login: AutoLoginSettings = Field(default_factory=AutoLoginSettings)
    project_dir: str = ""
    global_params: list[GlobalParam] = Field(default_factory=list)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Settings":
        """Load settings from YAML file with env var resolution."""
        yaml_path = Path(path)
        if not yaml_path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()

        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        # Rename 'list' to 'scenarios' for Python compatibility
        if "api_scenarios" in raw and "list" in raw["api_scenarios"]:
            raw["api_scenarios"]["scenarios"] = raw["api_scenarios"].pop("list")
        resolved = _resolve_env_vars(raw)
        return cls(**resolved)

    def save_to_yaml(self, path: str = "config.yaml"):
        """Persist current settings back to YAML."""
        # Build raw dict from model dump
        data = self.model_dump(exclude={"model_config"})
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ---- Logging setup ----

def setup_logging(level: str = "INFO"):
    """Configure loguru logging."""
    logger.remove()
    logger.add(
        "logs/agent_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="7 days",
        level=level,
        format="{time:HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )
    return logger


# ---- Global instance ----

_settings: Settings | None = None


def get_settings(config_path: str = "config.yaml") -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml(config_path)
    return _settings


def reload_settings(config_path: str = "config.yaml"):
    global _settings
    _settings = Settings.from_yaml(config_path)
    return _settings
