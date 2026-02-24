"""
OpenSage Configuration Manager.

Manages persistent configuration for the OpenSage CLI tool.

Configuration priority (highest to lowest):
  1. CLI flags (--model, --work-dir, etc.)
  2. Environment variables (OPENSAGE_MODEL, ANTHROPIC_API_KEY, etc.)
  3. Config file (~/.opensage/config.json)
  4. Built-in defaults

Config is stored at ~/.opensage/config.json
Session audit logs are stored at ~/.opensage/logs/YYYY-MM-DD.jsonl
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_DIR = Path.home() / ".opensage"
CONFIG_FILE = CONFIG_DIR / "config.json"
LOG_DIR = CONFIG_DIR / "logs"

DEFAULTS: Dict[str, Any] = {
    "model": "claude-opus-4-5",
    "topology": "auto",
    "max_iterations": 15,
    "log_sessions": True,
}

VALID_MODELS = [
    "claude-opus-4-5",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

VALID_TOPOLOGIES = ["auto", "vertical", "horizontal", "single"]

VALID_KEYS = {
    "api_key": "Anthropic API key",
    "model": f"Claude model ({', '.join(VALID_MODELS)})",
    "topology": f"Default topology ({', '.join(VALID_TOPOLOGIES)})",
    "max_iterations": "Max agent iterations per task (integer)",
    "log_sessions": "Write audit logs to ~/.opensage/logs/ (true/false)",
}


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class Config:
    """
    Persistent configuration for the OpenSage CLI.

    Loads from ~/.opensage/config.json on init, saves on set().
    Environment variables override stored values.
    """

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._ensure_dirs()
        self._load()

    def _ensure_dirs(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                # Corrupted config — start fresh, don't crash
                self._data = {}

    def _save(self) -> None:
        self._ensure_dirs()
        with open(CONFIG_FILE, "w") as f:
            json.dump(self._data, f, indent=2)
        # Restrict permissions so API key is not world-readable
        os.chmod(CONFIG_FILE, 0o600)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a config value.

        Checks environment variables first, then config file, then defaults.
        Special case: 'api_key' also checks ANTHROPIC_API_KEY env var.
        """
        # Check OPENSAGE_<KEY> environment variable
        env_key = f"OPENSAGE_{key.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            # Coerce types for known boolean/int keys
            if key == "log_sessions":
                return env_val.lower() not in ("false", "0", "no")
            if key == "max_iterations":
                try:
                    return int(env_val)
                except ValueError:
                    pass
            return env_val

        # Special handling: api_key also checks ANTHROPIC_API_KEY
        if key == "api_key":
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            if anthropic_key:
                return anthropic_key

        return self._data.get(key, DEFAULTS.get(key, default))

    def set(self, key: str, value: Any) -> None:
        """Store a configuration value persistently."""
        if key not in VALID_KEYS:
            raise ConfigError(
                f"Unknown configuration key: '{key}'\n"
                f"Valid keys: {', '.join(sorted(VALID_KEYS))}"
            )

        # Type coercions
        if key == "max_iterations":
            try:
                value = int(value)
                if value < 1 or value > 50:
                    raise ConfigError("max_iterations must be between 1 and 50")
            except (TypeError, ValueError):
                raise ConfigError(f"max_iterations must be an integer, got: {value!r}")

        if key == "log_sessions":
            if isinstance(value, str):
                value = value.lower() not in ("false", "0", "no")

        if key == "model" and value not in VALID_MODELS:
            # Allow unknown models but warn (don't block, API will reject if invalid)
            pass  # warning printed by caller

        if key == "topology" and value not in VALID_TOPOLOGIES:
            raise ConfigError(
                f"Invalid topology: '{value}'. Must be one of: {', '.join(VALID_TOPOLOGIES)}"
            )

        self._data[key] = value
        self._save()

    def unset(self, key: str) -> bool:
        """Remove a key from config. Returns True if it existed."""
        if key in self._data:
            del self._data[key]
            self._save()
            return True
        return False

    def all_settings(self) -> Dict[str, Any]:
        """
        Return all settings with merged defaults, masking the API key.
        Includes metadata about where each value comes from.
        """
        result: Dict[str, Any] = {}

        for key in VALID_KEYS:
            val = self.get(key)
            result[key] = val

        # Mask the API key for display
        api_key = result.get("api_key")
        if api_key:
            result["api_key"] = (
                f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
            )

        # Add source metadata
        if os.environ.get("ANTHROPIC_API_KEY"):
            result["api_key_source"] = "ANTHROPIC_API_KEY (env)"
        elif "api_key" in self._data:
            result["api_key_source"] = str(CONFIG_FILE)
        else:
            result["api_key_source"] = "not configured"

        return result

    def require_api_key(self) -> str:
        """
        Return the API key, raising ConfigError if not configured.
        Used by CLI commands before creating an engine.
        """
        key = self.get("api_key")
        if not key:
            raise ConfigError(
                "No Anthropic API key configured.\n\n"
                "Set it one of these ways:\n"
                "  1. opensage config set api_key sk-ant-...\n"
                "  2. export ANTHROPIC_API_KEY=sk-ant-...\n\n"
                "Get your key at https://console.anthropic.com/"
            )
        return key

    def __repr__(self) -> str:
        return f"Config(file={CONFIG_FILE}, keys={list(self._data.keys())})"
