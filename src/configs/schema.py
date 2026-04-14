from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    llm_path: str
    sae_path: str
    layer: int
    tokenizer_name_or_path: Optional[str] = None
    dtype: Optional[str] = None
    device_map: Optional[str] = None
    batch_size: Optional[int] = None


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    corpus: str
    words: List[str]
    years: List[int]
    env_path: Optional[str] = None
    target_word: Optional[str] = None
    compare_concepts: Dict[str, str] = Field(default_factory=dict)
    layer: Optional[int] = None
    models: Optional[List[ModelSpec]] = None

    @field_validator("years", mode="before")
    @classmethod
    def parse_years(cls, value: List[int | str]) -> List[int]:
        years: List[int] = []
        if value is None:
            return years
        for item in value:
            if isinstance(item, int):
                years.append(item)
                continue
            if not isinstance(item, str):
                raise ValueError(f"Invalid years config entry: {item}")
            match = None
            if "-" in item:
                match = item.split("-", 1)
            if match and len(match) == 2 and match[0].isdigit() and match[1].isdigit():
                start, end = int(match[0]), int(match[1])
                years.extend(list(range(start, end + 1)))
            elif item.isdigit():
                years.append(int(item))
            else:
                raise ValueError(f"Invalid years config entry: {item}")
        return sorted(set(years))


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_root: Optional[str] = None
    sentences_dir: Optional[str] = None
    metadata_dir: Optional[str] = None
    output_root: Optional[str] = None
    sae_ckpt: Optional[str] = None
    llama_path: Optional[str] = None
    font_path: Optional[str] = None
    models_dir: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def infer_defaults(cls, values: Dict[str, str]) -> Dict[str, str]:
        if not isinstance(values, dict):
            return values
        data_root = values.get("data_root")
        if data_root:
            root = Path(data_root)
            values = dict(values)
            values.setdefault("sentences_dir", str(root / "sentences"))
            values.setdefault("metadata_dir", str(root / "metadata"))
            values.setdefault("output_root", str(root / "output"))
        return values

    @model_validator(mode="after")
    def validate_required(self) -> "PathsConfig":
        missing = [
            name
            for name in [
                "sentences_dir",
                "metadata_dir",
                "output_root",
                "font_path",
            ]
            if getattr(self, name) in (None, "")
        ]
        if missing:
            raise ValueError(
                "paths is missing required fields: "
                + ", ".join(missing)
                + ". Check configs/env/local.yaml or provide command-line overrides."
            )
        return self


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str = "cpu"
    batch_size: Optional[int] = None
    seed: Optional[int] = None
    num_workers: Optional[int] = None


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentConfig
    paths: PathsConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def validate_model_paths(self) -> "AppConfig":
        models = self.experiment.models or []
        if models:
            for model in models:
                missing = [
                    field
                    for field in ["name", "llm_path", "sae_path"]
                    if getattr(model, field) in (None, "")
                ]
                if missing:
                    raise ValueError(
                        "experiment.models is missing required fields: "
                        + ", ".join(missing)
                        + "."
                    )
            return self

        if self.paths.sae_ckpt in (None, "") or self.paths.llama_path in (None, ""):
            raise ValueError(
                "paths.sae_ckpt and paths.llama_path are required unless model specs are provided in experiment.models."
            )
        return self
