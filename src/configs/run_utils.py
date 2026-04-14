from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional

from configs.schema import AppConfig, ModelSpec


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", name.strip())
    return cleaned.strip("._") or "unnamed"


def derive_model_name(llm_path: Optional[str]) -> str:
    if not llm_path:
        return "model"
    path = Path(llm_path)
    return _sanitize_name(path.name or path.stem or "model")


def derive_sae_id(sae_path: Optional[str]) -> str:
    if not sae_path:
        return "unknown"
    path = Path(sae_path)
    stem = _sanitize_name(path.stem or path.name or "sae")
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    if len(stem) > 32:
        stem = stem[:24]
    return f"{stem}-{digest}"


def get_active_model(cfg: AppConfig) -> ModelSpec:
    models = cfg.experiment.models or []
    if models:
        if len(models) > 1:
            raise ValueError(
                "experiment.models contains multiple entries. Keep only one active model configuration before running."
            )
        return models[0]
    layer = cfg.experiment.layer if cfg.experiment.layer is not None else -1
    return ModelSpec(
        name=derive_model_name(cfg.paths.llama_path),
        llm_path=cfg.paths.llama_path or "",
        sae_path=cfg.paths.sae_ckpt or "",
        layer=layer,
        batch_size=cfg.runtime.batch_size,
    )


def resolve_run_root(cfg: AppConfig, model: ModelSpec) -> Path:
    exp_name = _sanitize_name(cfg.experiment.name or "default")
    corpus = _sanitize_name(cfg.experiment.corpus)
    model_name = _sanitize_name(model.name)
    sae_id = derive_sae_id(model.sae_path)
    layer = model.layer
    return (
        Path(cfg.paths.output_root)
        / corpus
        / model_name
        / f"layer_{layer}"
        / f"sae_{sae_id}"
        / exp_name
    )


def get_git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return None


def build_run_manifest(cfg: AppConfig, model: ModelSpec) -> Dict[str, object]:
    return {
        "model_name": model.name,
        "llm_path": model.llm_path,
        "sae_path": model.sae_path,
        "layer": model.layer,
        "git_commit": get_git_commit(),
        "config_snapshot": cfg.model_dump(),
        "model_spec": model.model_dump(),
    }
