from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import ValidationError

from configs.schema import AppConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def normalize_legacy_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "experiment" in raw:
        return raw
    if all(key in raw for key in ("corpus", "words", "years")):
        experiment = {
            "corpus": raw.get("corpus"),
            "words": raw.get("words"),
            "years": raw.get("years"),
        }
        if raw.get("env_path"):
            experiment["env_path"] = raw.get("env_path")
        paths = {}
        if raw.get("data_path"):
            paths["sentences_dir"] = raw.get("data_path")
        return {"experiment": experiment, "paths": paths, "runtime": {}}
    return raw


def resolve_path(path: str, base: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def load_config(
    config_path: str,
    env_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AppConfig:
    config_file = resolve_path(config_path, REPO_ROOT)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    raw = read_yaml(config_file)
    raw = normalize_legacy_config(raw)

    env_from_exp = None
    if isinstance(raw.get("experiment"), dict):
        env_from_exp = raw["experiment"].get("env_path")
    if env_path is None:
        env_path = env_from_exp or "configs/env/local.yaml"

    env_file = resolve_path(env_path, config_file.parent)
    if not env_file.exists():
        # Support two styles:
        # 1) env_path relative to the config file directory (recommended)
        # 2) env_path relative to the repo root (common in legacy configs)
        env_file = resolve_path(env_path, REPO_ROOT)
    if not env_file.exists():
        raise FileNotFoundError(
            f"Environment config not found: {env_file}. Copy configs/env/local.example.yaml first and fill it in."
        )

    env_data = read_yaml(env_file)
    merged = deep_merge(raw, env_data)
    if overrides:
        merged = deep_merge(merged, overrides)

    try:
        return AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ValueError(
            "Configuration validation failed. Please check configs/exp/*.yaml and configs/env/local.yaml.\n"
            + str(exc)
        ) from exc
