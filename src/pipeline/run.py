from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_ROOT))

from configs.load import load_config
from configs.run_utils import build_run_manifest, get_active_model, resolve_run_root
from configs.schema import AppConfig


@dataclass
class RunnerContext:
    config_path: str
    resume: bool
    force: bool
    dry_run: bool
    max_tokens: int
    oom_recover: bool
    non_target_aliases: Optional[str] = None
    non_target_aliases_file: Optional[str] = None
    non_target_match_mode: str = "substring"
    non_target_target_regex: Optional[str] = None
    non_target_context_mode: str = "prepend"
    non_target_min_context_tokens: int = 20
    non_target_no_evidence_filter: bool = False
    non_target_no_evidence_dedupe: bool = False
    non_target_evidence_max_per_doc: int = 2
    non_target_high_activation_quantile: float = 0.95
    non_target_skip_extraction: bool = False
    non_target_recompute_contains_target: bool = False


@dataclass
class StageResult:
    name: str
    artifacts: List[Path]
    duration_s: float
    summary: str


RUN_CONTEXT: Optional[RunnerContext] = None


def set_context(context: RunnerContext) -> None:
    global RUN_CONTEXT
    RUN_CONTEXT = context


def get_context() -> RunnerContext:
    if RUN_CONTEXT is None:
        raise RuntimeError("Runner context has not been initialized.")
    return RUN_CONTEXT


def log_info(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [INFO] {message}")


def log_warning(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [WARN] {message}")


def resolve_corpus_path(root: Path, corpus: str) -> Path:
    candidate = root / corpus
    return candidate if candidate.exists() else root


def stage_done_path(output_dir: Path, stage: str) -> Path:
    return output_dir / f".{stage}.done"


def should_skip_stage(done_path: Path) -> bool:
    context = get_context()
    if context.force:
        return False
    if not context.resume:
        return False
    return done_path.exists()


def run_subprocess(args: List[str]) -> None:
    subprocess.run(args, check=True)


def prepare_word_output(cfg: AppConfig, word: str) -> Path:
    model = get_active_model(cfg)
    run_root = resolve_run_root(cfg, model)
    output_dir = run_root / word
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(build_run_manifest(cfg, model), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return output_dir


def stage_extract(cfg: AppConfig) -> StageResult:
    context = get_context()
    stage_name = "extract"
    start = time.perf_counter()
    artifacts: List[Path] = []
    model = get_active_model(cfg)

    for word in cfg.experiment.words:
        output_dir = prepare_word_output(cfg, word)
        done_path = stage_done_path(output_dir, stage_name)
        act_dir = output_dir / "activations"
        sent_dir = resolve_corpus_path(Path(cfg.paths.sentences_dir), cfg.experiment.corpus)
        meta_root = resolve_corpus_path(Path(cfg.paths.metadata_dir), cfg.experiment.corpus)
        meta_path = meta_root / "documents.jsonl"

        if should_skip_stage(done_path):
            log_info(f"[{stage_name}] Skipping {word} (already exists: {done_path})")
            artifacts.append(act_dir)
            continue

        artifacts.append(act_dir)

        if context.dry_run:
            continue

        act_dir.mkdir(parents=True, exist_ok=True)
        run_subprocess([
            "python",
            str(Path(__file__).parent / "extract_activations.py"),
            "--word",
            word,
            "--years",
            *map(str, cfg.experiment.years),
            "--sent_dir",
            str(sent_dir),
            "--meta_path",
            str(meta_path),
            "--out_dir",
            str(act_dir),
            "--sae_ckpt",
            model.sae_path,
            "--llama_path",
            model.llm_path,
            "--device",
            cfg.runtime.device,
            "--corpus",
            cfg.experiment.corpus,
        ])
        done_path.write_text(f"{stage_name} done at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    duration = time.perf_counter() - start
    return StageResult(stage_name, artifacts, duration, "Activation extraction completed")


def stage_centers(cfg: AppConfig) -> StageResult:
    context = get_context()
    stage_name = "centers"
    start = time.perf_counter()
    artifacts: List[Path] = []

    for word in cfg.experiment.words:
        output_dir = prepare_word_output(cfg, word)
        done_path = stage_done_path(output_dir, stage_name)
        act_dir = output_dir / "activations"
        center_f = output_dir / "yearly_centers.json"
        dist_f = output_dir / "yearly_distances.json"

        if should_skip_stage(done_path):
            log_info(f"[{stage_name}] Skipping {word} (already exists: {done_path})")
            artifacts.extend([center_f, dist_f])
            continue

        artifacts.extend([center_f, dist_f])

        if context.dry_run:
            continue

        run_subprocess([
            "python",
            str(Path(__file__).parent / "compute_centers.py"),
            "--word",
            word,
            "--act_dir",
            str(act_dir),
            "--out_center",
            str(center_f),
            "--out_dist",
            str(dist_f),
        ])
        done_path.write_text(f"{stage_name} done at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    duration = time.perf_counter() - start
    return StageResult(stage_name, artifacts, duration, "Yearly center computation completed")


def stage_drift(cfg: AppConfig) -> StageResult:
    context = get_context()
    stage_name = "drift"
    start = time.perf_counter()
    artifacts: List[Path] = []
    model = get_active_model(cfg)

    for word in cfg.experiment.words:
        output_dir = prepare_word_output(cfg, word)
        done_path = stage_done_path(output_dir, stage_name)
        act_dir = output_dir / "activations"
        center_f = output_dir / "yearly_centers.json"
        drift_f = output_dir / "top_drift_bases.json"

        if should_skip_stage(done_path):
            log_info(f"[{stage_name}] Skipping {word} (already exists: {done_path})")
            artifacts.append(drift_f)
            continue

        artifacts.append(drift_f)

        if context.dry_run:
            continue

        run_subprocess([
            "python",
            str(Path(__file__).parent / "identify_drift.py"),
            "--center_f",
            str(center_f),
            "--drift_f",
            str(drift_f),
            "--top_n",
            "15",
            "--layer",
            str(model.layer),
            "--source",
            cfg.experiment.corpus,
            "--concept",
            word,
            "--year_start",
            str(min(cfg.experiment.years)),
            "--year_end",
            str(max(cfg.experiment.years)),
        ])
        done_path.write_text(f"{stage_name} done at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    duration = time.perf_counter() - start
    return StageResult(stage_name, artifacts, duration, "Drift-base selection completed")


def stage_peak_sentences(cfg: AppConfig) -> StageResult:
    context = get_context()
    stage_name = "peak_sentences"
    start = time.perf_counter()
    artifacts: List[Path] = []

    for word in cfg.experiment.words:
        output_dir = prepare_word_output(cfg, word)
        done_path = stage_done_path(output_dir, stage_name)
        act_dir = output_dir / "activations"
        drift_f = output_dir / "top_drift_bases.json"
        out_json = output_dir / "key_bases_peak_change.json"
        txt_dir = output_dir / "sentences"

        if should_skip_stage(done_path):
            log_info(f"[{stage_name}] Skipping {word} (already exists: {done_path})")
            artifacts.append(out_json)
            continue

        artifacts.append(out_json)

        if context.dry_run:
            continue

        run_subprocess([
            "python",
            str(Path(__file__).parent / "extract_peak_sentences.py"),
            "--word",
            word,
            "--act_dir",
            str(act_dir),
            "--drift_f",
            str(drift_f),
            "--out_json",
            str(out_json),
            "--txt_dir",
            str(txt_dir),
            "--top_sent",
            "5",
        ])
        done_path.write_text(f"{stage_name} done at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    duration = time.perf_counter() - start
    return StageResult(stage_name, artifacts, duration, "Peak-year sentence extraction completed")


def stage_wordclouds(cfg: AppConfig) -> StageResult:
    context = get_context()
    stage_name = "wordclouds"
    start = time.perf_counter()
    artifacts: List[Path] = []

    for word in cfg.experiment.words:
        output_dir = prepare_word_output(cfg, word)
        done_path = stage_done_path(output_dir, stage_name)
        act_dir = output_dir / "activations"
        drift_f = output_dir / "top_drift_bases.json"
        txt_dir = output_dir / "top30_sentences"
        wc_dir = output_dir / "wordclouds"

        if should_skip_stage(done_path):
            log_info(f"[{stage_name}] Skipping {word} (already exists: {done_path})")
            artifacts.append(wc_dir)
            continue

        artifacts.append(wc_dir)

        if context.dry_run:
            continue

        run_subprocess([
            "python",
            str(Path(__file__).parent / "build_wordclouds.py"),
            "--word",
            word,
            "--act_dir",
            str(act_dir),
            "--drift_f",
            str(drift_f),
            "--txt_dir",
            str(txt_dir),
            "--wc_dir",
            str(wc_dir),
            "--font_path",
            cfg.paths.font_path,
        ])
        done_path.write_text(f"{stage_name} done at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    duration = time.perf_counter() - start
    return StageResult(stage_name, artifacts, duration, "Word clouds and top-sentence exports completed")


def stage_target_sentence_exports(cfg: AppConfig) -> StageResult:
    context = get_context()
    stage_name = "target_sentence_exports"
    start = time.perf_counter()
    artifacts: List[Path] = []

    for word in cfg.experiment.words:
        output_dir = prepare_word_output(cfg, word)
        done_path = stage_done_path(output_dir, stage_name)
        act_dir = output_dir / "activations"
        export_dir = output_dir / "target_sorted_sentences_by_year"

        if should_skip_stage(done_path):
            log_info(f"[{stage_name}] Skipping {word} (already exists: {done_path})")
            artifacts.append(export_dir)
            continue

        artifacts.append(export_dir)

        if context.dry_run:
            continue

        run_subprocess([
            "python",
            str(Path(__file__).parent / "export_target_ranked_sentences.py"),
            "--word",
            word,
            "--act_dir",
            str(act_dir),
            "--out_dir",
            str(export_dir),
        ])
        done_path.write_text(f"{stage_name} done at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    duration = time.perf_counter() - start
    return StageResult(stage_name, artifacts, duration, "Target yearly ranked sentence export completed")


def stage_visualize_simple(cfg: AppConfig) -> StageResult:
    return _stage_visualize(cfg, "visualize_simple", "simple")


def stage_visualize_full(cfg: AppConfig) -> StageResult:
    return _stage_visualize(cfg, "visualize_full", "full")


def _stage_visualize(cfg: AppConfig, stage_name: str, mode: str) -> StageResult:
    context = get_context()
    start = time.perf_counter()
    artifacts: List[Path] = []

    for word in cfg.experiment.words:
        output_dir = prepare_word_output(cfg, word)
        done_path = stage_done_path(output_dir, stage_name)
        viz_dir = output_dir / "visualizations"

        if should_skip_stage(done_path):
            log_info(f"[{stage_name}] Skipping {word} (already exists: {done_path})")
            artifacts.append(viz_dir)
            continue

        artifacts.append(viz_dir)

        if context.dry_run:
            continue

        run_subprocess([
            "python",
            str(Path(__file__).resolve().parents[1] / "analysis" / "visualize.py"),
            "--mode",
            mode,
            "--config",
            context.config_path,
            "--target-word",
            word,
        ])
        done_path.write_text(f"{stage_name} done at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    duration = time.perf_counter() - start
    return StageResult(stage_name, artifacts, duration, f"Visualization ({mode}) completed")


def run_non_target_visualizations(
    cfg: AppConfig,
    model,
    word: str,
    analysis_dir: Path,
) -> List[Path]:
    artifacts: List[Path] = []
    annual_summary = analysis_dir / f"{word}_annual_means_summary.json"
    manifest_path = (
        analysis_dir
        / "without_target_high_activation_sentences"
        / f"{word}_high_activation_manifest.json"
    )

    if annual_summary.exists():
        run_subprocess([
            "python",
            str(Path(__file__).resolve().parents[1] / "analysis" / "target_vs_non_target.py"),
            "--corpus",
            cfg.experiment.corpus,
            "--concept",
            word,
            "--run_root",
            str(resolve_run_root(cfg, model)),
            "--font_path",
            cfg.paths.font_path,
        ])
        artifacts.append(prepare_word_output(cfg, word) / "visualizations")
    else:
        log_warning(f"[non_target_visualize] Missing yearly summary file, skipping comparison plot: {annual_summary}")

    if manifest_path.exists():
        run_subprocess([
            "python",
            str(Path(__file__).resolve().parents[1] / "analysis" / "visualize_non_target.py"),
            "--corpus",
            cfg.experiment.corpus,
            "--concept",
            word,
            "--run_root",
            str(resolve_run_root(cfg, model)),
            "--font_path",
            cfg.paths.font_path,
        ])
        artifacts.append(analysis_dir / "visualizations")
    else:
        log_warning(
            f"[non_target_visualize] Missing high-activation manifest, skipping non-target plots: {manifest_path}"
        )

    return artifacts


def stage_non_target(cfg: AppConfig) -> StageResult:
    context = get_context()
    stage_name = "non_target"
    start = time.perf_counter()
    artifacts: List[Path] = []
    model = get_active_model(cfg)

    sent_dir = resolve_corpus_path(Path(cfg.paths.sentences_dir), cfg.experiment.corpus)
    meta_root = resolve_corpus_path(Path(cfg.paths.metadata_dir), cfg.experiment.corpus)
    meta_path = meta_root / "documents.jsonl"

    for word in cfg.experiment.words:
        output_dir = prepare_word_output(cfg, word)
        done_path = stage_done_path(output_dir, stage_name)
        analysis_dir = output_dir / "non_target_analysis"
        top_bases_file = output_dir / "top_drift_bases.json"
        annual_summary = analysis_dir / f"{word}_annual_means_summary.json"

        if should_skip_stage(done_path):
            log_info(f"[{stage_name}] Skipping {word} (already exists: {done_path})")
            artifacts.append(analysis_dir)
            continue

        artifacts.append(analysis_dir)

        if context.dry_run:
            continue

        if not top_bases_file.exists():
            log_warning(f"[{stage_name}] Missing top_drift_bases.json, skipping {word}")
            continue

        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / "all_activations").mkdir(parents=True, exist_ok=True)
        (analysis_dir / "logs").mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            str(Path(__file__).resolve().parents[1] / "non_target" / "analyze_non_target_activations.py"),
            "--word",
            word,
            "--years",
            *map(str, cfg.experiment.years),
            "--sent_dir",
            str(sent_dir),
            "--meta_path",
            str(meta_path),
            "--top_bases_file",
            str(top_bases_file),
            "--sae_ckpt",
            model.sae_path,
            "--llama_path",
            model.llm_path,
            "--device",
            cfg.runtime.device,
            "--output_dir",
            str(analysis_dir),
            "--top_n_sentences",
            "30",
            "--max_tokens",
            str(context.max_tokens),
        ]
        if context.non_target_skip_extraction:
            cmd.append("--skip_extraction")
        if context.non_target_recompute_contains_target:
            cmd.append("--recompute_contains_target")
        if context.non_target_aliases:
            cmd.extend(["--aliases", context.non_target_aliases])
        if context.non_target_aliases_file:
            cmd.extend(["--aliases_file", context.non_target_aliases_file])
        if context.non_target_match_mode:
            cmd.extend(["--match_mode", context.non_target_match_mode])
        if context.non_target_target_regex:
            cmd.extend(["--target_regex", context.non_target_target_regex])
        if context.non_target_context_mode:
            cmd.extend(["--context_mode", context.non_target_context_mode])
        cmd.extend(["--min_context_tokens", str(context.non_target_min_context_tokens)])
        if context.non_target_no_evidence_filter:
            cmd.append("--no_evidence_filter")
        if context.non_target_no_evidence_dedupe:
            cmd.append("--no_evidence_dedupe")
        cmd.extend(["--evidence_max_per_doc", str(context.non_target_evidence_max_per_doc)])
        cmd.extend(["--high_activation_quantile", str(context.non_target_high_activation_quantile)])
        if context.oom_recover:
            cmd.append("--oom_recover")

        run_subprocess(cmd)

        run_non_target_visualizations(cfg, model, word, analysis_dir)

        done_path.write_text(f"{stage_name} done at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    duration = time.perf_counter() - start
    return StageResult(stage_name, artifacts, duration, "Non-target analysis completed")


def stage_non_target_visualize(cfg: AppConfig) -> StageResult:
    context = get_context()
    stage_name = "non_target_visualize"
    start = time.perf_counter()
    artifacts: List[Path] = []
    model = get_active_model(cfg)

    for word in cfg.experiment.words:
        output_dir = prepare_word_output(cfg, word)
        done_path = stage_done_path(output_dir, stage_name)
        analysis_dir = output_dir / "non_target_analysis"
        concept_viz_dir = output_dir / "visualizations"
        non_target_viz_dir = analysis_dir / "visualizations"

        if should_skip_stage(done_path):
            log_info(f"[{stage_name}] Skipping {word} (already exists: {done_path})")
            artifacts.extend([concept_viz_dir, non_target_viz_dir])
            continue

        artifacts.extend([concept_viz_dir, non_target_viz_dir])

        if context.dry_run:
            continue

        if not analysis_dir.exists():
            log_warning(f"[{stage_name}] Missing non_target_analysis directory, skipping {word}: {analysis_dir}")
            continue

        run_non_target_visualizations(cfg, model, word, analysis_dir)
        done_path.write_text(f"{stage_name} done at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    duration = time.perf_counter() - start
    return StageResult(stage_name, artifacts, duration, "Non-target visualization completed")


STAGES: Dict[str, Callable[[AppConfig], StageResult]] = {
    "extract": stage_extract,
    "centers": stage_centers,
    "drift": stage_drift,
    "peak_sentences": stage_peak_sentences,
    "wordclouds": stage_wordclouds,
    "target_sentence_exports": stage_target_sentence_exports,
    "visualize": stage_visualize_simple,
    "visualize_simple": stage_visualize_simple,
    "visualize_full": stage_visualize_full,
    "non_target": stage_non_target,
    "non_target_visualize": stage_non_target_visualize,
}

STAGE_ORDER = [
    "extract",
    "centers",
    "drift",
    "peak_sentences",
    "wordclouds",
    "target_sentence_exports",
    "visualize",
]


def parse_stage_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return STAGE_ORDER
    stages = [stage.strip() for stage in raw.split(",") if stage.strip()]
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s): {', '.join(unknown)}")
    return stages


def main() -> int:
    parser = argparse.ArgumentParser(description="HistSAE pipeline runner")
    parser.add_argument("--config", required=True, help="Experiment config path")
    parser.add_argument("--stages", default=None, help="Comma-separated stage list")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a stage when its artifacts already exist (enabled by default)",
    )
    parser.add_argument("--force", action="store_true", help="Force reruns for all stages")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running computations")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum token count for non-target analysis")
    parser.add_argument(
        "--oom-recover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue after OOM errors during non-target analysis (enabled by default)",
    )
    parser.add_argument(
        "--non-target-aliases",
        default=None,
        help="Extra aliases for non-target analysis, comma-separated, used for contains_target checks",
    )
    parser.add_argument(
        "--non-target-aliases-file",
        default=None,
        help="Alias file for non-target analysis, one alias per line with optional # comments",
    )
    parser.add_argument(
        "--non-target-match-mode",
        default="substring",
        choices=["substring", "regex"],
        help="contains_target matching mode for non-target analysis",
    )
    parser.add_argument(
        "--non-target-target-regex",
        default=None,
        help="Regular expression used when --non-target-match-mode=regex",
    )
    parser.add_argument(
        "--non-target-context-mode",
        default="prepend",
        choices=["prepend", "none"],
        help="Whether to prepend the previous line as additional context in non-target analysis",
    )
    parser.add_argument(
        "--non-target-min-context-tokens",
        type=int,
        default=20,
        help="Minimum token target when --non-target-context-mode=prepend",
    )
    parser.add_argument(
        "--non-target-no-evidence-filter",
        action="store_true",
        help="Disable evidence-sentence filtering in non-target analysis",
    )
    parser.add_argument(
        "--non-target-no-evidence-dedupe",
        action="store_true",
        help="Disable evidence-sentence deduplication in non-target analysis",
    )
    parser.add_argument(
        "--non-target-evidence-max-per-doc",
        type=int,
        default=2,
        help="Maximum number of evidence sentences to keep per document when deduplication is enabled",
    )
    parser.add_argument(
        "--non-target-high-activation-quantile",
        type=float,
        default=0.95,
        help="Quantile threshold used to export high-activation non-target sentences, based on non-zero without_target activations",
    )
    parser.add_argument(
        "--non-target-skip-extraction",
        action="store_true",
        help="Reuse existing all_activations files for non-target analysis without rerunning SAE extraction",
    )
    parser.add_argument(
        "--non-target-recompute-contains-target",
        action="store_true",
        help="Recompute contains_target during non-target analysis using aliases, match mode, and target regex",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    context = RunnerContext(
        config_path=args.config,
        resume=args.resume,
        force=args.force,
        dry_run=args.dry_run,
        max_tokens=args.max_tokens,
        oom_recover=args.oom_recover,
        non_target_aliases=args.non_target_aliases,
        non_target_aliases_file=args.non_target_aliases_file,
        non_target_match_mode=args.non_target_match_mode,
        non_target_target_regex=args.non_target_target_regex,
        non_target_context_mode=args.non_target_context_mode,
        non_target_min_context_tokens=args.non_target_min_context_tokens,
        non_target_no_evidence_filter=args.non_target_no_evidence_filter,
        non_target_no_evidence_dedupe=args.non_target_no_evidence_dedupe,
        non_target_evidence_max_per_doc=args.non_target_evidence_max_per_doc,
        non_target_high_activation_quantile=args.non_target_high_activation_quantile,
        non_target_skip_extraction=args.non_target_skip_extraction,
        non_target_recompute_contains_target=args.non_target_recompute_contains_target,
    )
    set_context(context)

    stages = parse_stage_list(args.stages)
    log_info(f"Planned stages: {', '.join(stages)}")
    log_info(f"Resume: {context.resume}, Force: {context.force}, Dry-run: {context.dry_run}")

    try:
        for stage in stages:
            stage_func = STAGES[stage]
            log_info(f"Starting stage: {stage}")
            result = stage_func(cfg)
            log_info(
                f"Finished stage: {stage} (duration {result.duration_s:.2f}s, artifacts {len(result.artifacts)})"
            )
    except Exception as exc:
        log_warning(f"Stage execution failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
