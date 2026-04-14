#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate supplemental visualizations for non_target_analysis.

Inputs:
- {run_root}/{concept}/top_drift_bases.json
- {run_root}/{concept}/non_target_analysis/{concept}_annual_means_summary.json
- {run_root}/{concept}/non_target_analysis/without_target_high_activation_sentences/{concept}_high_activation_manifest.json

Outputs:
- {run_root}/{concept}/non_target_analysis/visualizations/*.png
- {run_root}/{concept}/non_target_analysis/visualizations/{concept}_non_target_visualization_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

FONT_PROP = None


def setup_chinese_font(font_path: str | None = None):
    global FONT_PROP
    if font_path and os.path.exists(font_path):
        try:
            font_manager.fontManager.addfont(font_path)
            FONT_PROP = font_manager.FontProperties(fname=font_path)
            font_name = FONT_PROP.get_name()
            plt.rcParams["font.family"] = [font_name]
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return FONT_PROP
        except Exception:
            FONT_PROP = None
    plt.rcParams["axes.unicode_minus"] = False
    return None


def get_font_prop():
    return FONT_PROP


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def resolve_concept_dir(output_dir: str, corpus: str, concept: str) -> Path:
    base_dir = Path(output_dir)
    candidate = base_dir / corpus / concept
    if candidate.exists():
        return candidate
    candidate = base_dir / concept
    if candidate.exists():
        return candidate
    return base_dir / corpus / concept


def quantile_tag(value: float) -> str:
    return f"q{int(round(float(value) * 1000)):03d}"


def normalize_top_bases(obj) -> List[dict]:
    if isinstance(obj, dict) and "top_bases" in obj:
        return obj["top_bases"]
    if isinstance(obj, dict):
        normalized = []
        for base_id, info in obj.items():
            if str(base_id).isdigit():
                normalized.append({"base_id": int(base_id), **info})
        normalized.sort(key=lambda item: float(item.get("cum_drift", 0.0)), reverse=True)
        return normalized
    if isinstance(obj, list):
        return obj
    return []


def unique_sentence_key(record: dict) -> Tuple[str, str]:
    return (str(record.get("doc_id", "")), str(record.get("sentence", "")).strip())


def collect_years(
    annual_summary: Dict[str, Dict],
    manifest: Dict[str, object],
    base_entries: Dict[str, dict],
) -> List[int]:
    years = {
        int(year)
        for year in annual_summary.keys()
        if str(year).isdigit()
    }
    years.update(
        int(year)
        for year in manifest.get("high_activation_by_year_and_base_files", {}).keys()
        if str(year).isdigit()
    )
    for entry in base_entries.values():
        years.update(
            int(year)
            for year in entry.get("high_activation_sentence_count_by_year", {}).keys()
            if str(year).isdigit()
        )
    return sorted(years)


def build_mean_matrix(
    annual_summary: Dict[str, Dict],
    base_ids: Sequence[str],
    years: Sequence[int],
) -> np.ndarray:
    matrix = np.zeros((len(base_ids), len(years)), dtype=float)
    for row_idx, base_id in enumerate(base_ids):
        for col_idx, year in enumerate(years):
            entry = annual_summary.get(str(year), {}).get(str(base_id), {})
            matrix[row_idx, col_idx] = float(
                entry.get("without_target_mean_including_zeros", 0.0)
            )
    return matrix


def build_count_matrix(
    base_entries: Dict[str, dict],
    base_ids: Sequence[str],
    years: Sequence[int],
) -> np.ndarray:
    matrix = np.zeros((len(base_ids), len(years)), dtype=float)
    for row_idx, base_id in enumerate(base_ids):
        year_counts = base_entries.get(str(base_id), {}).get(
            "high_activation_sentence_count_by_year", {}
        )
        for col_idx, year in enumerate(years):
            matrix[row_idx, col_idx] = float(year_counts.get(str(year), 0))
    return matrix


def render_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[int],
    *,
    title: str,
    subtitle: str,
    output_path: Path,
    cmap: str,
    value_fmt: str,
) -> None:
    font_prop = get_font_prop()
    fig_w = max(8.0, 0.72 * len(col_labels) + 3.5)
    fig_h = max(5.0, 0.42 * len(row_labels) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([str(year) for year in col_labels], rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    if font_prop:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_prop)
    if font_prop:
        ax.set_xlabel("Year", fontproperties=font_prop)
        ax.set_ylabel("Base", fontproperties=font_prop)
        ax.set_title(title, fontproperties=font_prop, pad=18)
    else:
        ax.set_xlabel("Year")
        ax.set_ylabel("Base")
        ax.set_title(title, pad=18)
    ax.text(
        0.0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        ha="left",
        va="bottom",
        fontproperties=font_prop,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    if font_prop:
        cbar.ax.set_ylabel("Value", rotation=270, labelpad=14, fontproperties=font_prop)
    else:
        cbar.ax.set_ylabel("Value", rotation=270, labelpad=14)

    if matrix.size and len(row_labels) * len(col_labels) <= 220:
        midpoint = float(np.nanmax(matrix) + np.nanmin(matrix)) / 2.0 if matrix.size else 0.0
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                color = "white" if value > midpoint else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    format(value, value_fmt),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_yearly_overview(
    analysis_dir: Path,
    manifest: Dict[str, object],
    base_ids: Sequence[str],
    years: Sequence[int],
) -> List[dict]:
    by_year = manifest.get("high_activation_by_year_and_base_files", {})
    coactivation_files = manifest.get("coactivation_sentence_by_year_files", {})
    base_set = {str(base_id) for base_id in base_ids}
    overview: List[dict] = []

    for year in years:
        year_key = str(year)
        year_files = by_year.get(year_key, {})
        unique_sentences = set()
        raw_record_count = 0
        active_bases = 0
        for base_id in base_ids:
            rel_path = year_files.get(str(base_id))
            if not rel_path:
                continue
            base_path = analysis_dir / "without_target_high_activation_sentences_by_year" / rel_path
            records = read_jsonl(base_path)
            if not records:
                continue
            active_bases += 1
            raw_record_count += len(records)
            for record in records:
                unique_sentences.add(unique_sentence_key(record))

        multi_base_count = 0
        multi_file = coactivation_files.get(year_key)
        if multi_file:
            multi_records = read_jsonl(
                analysis_dir / "without_target_high_activation_coactivation" / multi_file
            )
            multi_base_count = len(multi_records)

        overview.append(
            {
                "year": year,
                "unique_high_activation_sentences": len(unique_sentences),
                "raw_base_hits": raw_record_count,
                "active_base_count": active_bases,
                "inactive_base_count": max(len(base_set) - active_bases, 0),
                "multi_base_sentence_count": multi_base_count,
                "multi_base_share": (
                    multi_base_count / len(unique_sentences) if unique_sentences else 0.0
                ),
            }
        )
    return overview


def plot_yearly_overview(
    overview: Sequence[dict],
    *,
    concept: str,
    quantile_tag_text: str,
    output_path: Path,
) -> None:
    font_prop = get_font_prop()
    years = [item["year"] for item in overview]
    unique_counts = [item["unique_high_activation_sentences"] for item in overview]
    multi_counts = [item["multi_base_sentence_count"] for item in overview]
    active_bases = [item["active_base_count"] for item in overview]
    raw_hits = [item["raw_base_hits"] for item in overview]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(years))
    width = 0.38
    ax.bar(
        x - width / 2,
        unique_counts,
        width=width,
        color="#2F6DB3",
        label="Unique high-activation sentences",
    )
    ax.bar(
        x + width / 2,
        multi_counts,
        width=width,
        color="#D55E00",
        label="Multi-base coactivation sentences",
    )
    ax.plot(
        x,
        raw_hits,
        color="#7A7A7A",
        linestyle="--",
        marker="o",
        linewidth=1.5,
        label="Raw base-hit count",
    )
    if font_prop:
        ax.set_title(
            f"{concept} non-target yearly high-activation overview ({quantile_tag_text})",
            fontproperties=font_prop,
        )
        ax.set_xlabel("Year", fontproperties=font_prop)
        ax.set_ylabel("Sentences / records", fontproperties=font_prop)
    else:
        ax.set_title(f"{concept} non-target yearly high-activation overview ({quantile_tag_text})")
        ax.set_xlabel("Year")
        ax.set_ylabel("Sentences / records")
    ax.set_xticks(x)
    ax.set_xticklabels([str(year) for year in years], rotation=45, ha="right")
    if font_prop:
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_prop)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        active_bases,
        color="#009E73",
        marker="s",
        linewidth=2,
        label="Active bases",
    )
    if font_prop:
        ax2.set_ylabel("Active bases", fontproperties=font_prop)
    else:
        ax2.set_ylabel("Active bases")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend = ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right",
        frameon=False,
    )
    if font_prop:
        for text in legend.get_texts():
            text.set_fontproperties(font_prop)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_coactivation_pairs(
    records: Sequence[dict],
    *,
    concept: str,
    quantile_tag_text: str,
    output_path: Path,
    top_k: int = 15,
) -> List[dict]:
    top_records = sorted(
        records,
        key=lambda item: int(item.get("coactivation_sentence_count", 0)),
        reverse=True,
    )[:top_k]
    font_prop = get_font_prop()
    if not top_records:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.axis("off")
        title = f"{concept} non-target base coactivation pairs ({quantile_tag_text})"
        if font_prop:
            ax.set_title(title, fontproperties=font_prop)
            ax.text(
                0.5,
                0.5,
                "No coactivation-pair data available for plotting",
                ha="center",
                va="center",
                fontproperties=font_prop,
            )
        else:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No coactivation pairs", ha="center", va="center")
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return []

    labels = [
        f"{item['base_id_1']} + {item['base_id_2']}"
        for item in reversed(top_records)
    ]
    values = [
        int(item.get("coactivation_sentence_count", 0))
        for item in reversed(top_records)
    ]

    fig_h = max(5.5, 0.45 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.barh(np.arange(len(labels)), values, color="#8C564B")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    if font_prop:
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_prop)
    if font_prop:
        ax.set_title(
            f"{concept} non-target Top {len(top_records)} coactivation base pairs ({quantile_tag_text})",
            fontproperties=font_prop,
        )
        ax.set_xlabel("Coactivation sentence count", fontproperties=font_prop)
    else:
        ax.set_title(f"{concept} non-target Top {len(top_records)} coactivation base pairs ({quantile_tag_text})")
        ax.set_xlabel("Coactivation sentence count")
    for idx, value in enumerate(values):
        ax.text(value, idx, f" {value}", va="center", ha="left", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return top_records


def build_visualization_summary(
    *,
    concept: str,
    years: Sequence[int],
    base_ids: Sequence[str],
    quantile: float,
    threshold_mode: str,
    yearly_overview: Sequence[dict],
    top_pairs: Sequence[dict],
    output_files: Dict[str, str],
) -> dict:
    max_unique_year = max(
        yearly_overview,
        key=lambda item: int(item.get("unique_high_activation_sentences", 0)),
    ) if yearly_overview else None
    max_multi_year = max(
        yearly_overview,
        key=lambda item: float(item.get("multi_base_share", 0.0)),
    ) if yearly_overview else None
    return {
        "concept": concept,
        "years": list(years),
        "base_ids": list(base_ids),
        "selection": {
            "high_activation_quantile": quantile,
            "high_activation_quantile_tag": quantile_tag(quantile),
            "high_activation_threshold_mode": threshold_mode,
        },
        "yearly_overview": list(yearly_overview),
        "top_coactivation_pairs": list(top_pairs),
        "highlights": {
            "max_unique_sentence_year": max_unique_year,
            "max_multi_base_share_year": max_multi_year,
        },
        "output_files": output_files,
    }


def plot_non_target_visualizations(
    *,
    corpus: str,
    concept: str,
    output_dir: str | None,
    font_path: str | None = None,
    run_root: str | None = None,
) -> Dict[str, str]:
    setup_chinese_font(font_path)

    if run_root:
        concept_dir = Path(run_root) / concept
    else:
        if not output_dir:
            raise ValueError("output_dir and run_root cannot both be empty.")
        concept_dir = resolve_concept_dir(output_dir, corpus, concept)

    analysis_dir = concept_dir / "non_target_analysis"
    visual_dir = analysis_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)

    annual_summary_path = analysis_dir / f"{concept}_annual_means_summary.json"
    manifest_path = (
        analysis_dir
        / "without_target_high_activation_sentences"
        / f"{concept}_high_activation_manifest.json"
    )
    top_bases_path = concept_dir / "top_drift_bases.json"
    base_pair_path = (
        analysis_dir
        / "without_target_high_activation_coactivation"
        / f"{concept}_base_pair_counts_q950.jsonl"
    )

    if not annual_summary_path.exists():
        raise FileNotFoundError(f"Missing yearly summary file: {annual_summary_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing high-activation manifest: {manifest_path}")
    if not top_bases_path.exists():
        raise FileNotFoundError(f"Missing top drift bases file: {top_bases_path}")

    annual_summary = load_json(annual_summary_path)
    manifest = load_json(manifest_path)
    top_bases = normalize_top_bases(load_json(top_bases_path))
    if not top_bases:
        raise RuntimeError(f"No usable bases found in top_drift_bases.json: {top_bases_path}")

    base_ids = [str(item["base_id"]) for item in top_bases]
    base_entries = manifest.get("bases", {})
    years = collect_years(annual_summary, manifest, base_entries)
    if not years:
        raise RuntimeError(f"No usable year data found: {concept_dir}")

    quantile = float(manifest.get("selection", {}).get("high_activation_quantile", 0.95))
    qtag = quantile_tag(quantile)
    threshold_mode = str(
        manifest.get("selection", {}).get("high_activation_threshold_mode", "unknown")
    )

    mean_matrix = build_mean_matrix(annual_summary, base_ids, years)
    count_matrix = build_count_matrix(base_entries, base_ids, years)
    yearly_overview = compute_yearly_overview(analysis_dir, manifest, base_ids, years)
    pair_path = (
        analysis_dir
        / "without_target_high_activation_coactivation"
        / f"{concept}_base_pair_counts_{qtag}.jsonl"
    )
    if not pair_path.exists():
        pair_path = base_pair_path
    pair_records = read_jsonl(pair_path)

    output_files = {
        "mean_heatmap": str(
            (visual_dir / f"{concept}_non_target_mean_including_zeros_heatmap.png").relative_to(
                analysis_dir
            )
        ),
        "high_activation_count_heatmap": str(
            (
                visual_dir / f"{concept}_non_target_high_activation_count_heatmap_{qtag}.png"
            ).relative_to(analysis_dir)
        ),
        "yearly_overview": str(
            (visual_dir / f"{concept}_non_target_yearly_overview_{qtag}.png").relative_to(
                analysis_dir
            )
        ),
        "top_coactivation_pairs": str(
            (
                visual_dir / f"{concept}_non_target_top_coactivation_pairs_{qtag}.png"
            ).relative_to(analysis_dir)
        ),
        "summary": str(
            (visual_dir / f"{concept}_non_target_visualization_summary.json").relative_to(
                analysis_dir
            )
        ),
    }

    render_heatmap(
        mean_matrix,
        [f"Base {base_id}" for base_id in base_ids],
        years,
        title=f"{concept} non-target yearly mean activation heatmap",
        subtitle="Values are without_target_mean_including_zeros; row order follows top_drift_bases.",
        output_path=analysis_dir / output_files["mean_heatmap"],
        cmap="YlGnBu",
        value_fmt=".3f",
    )
    render_heatmap(
        count_matrix,
        [f"Base {base_id}" for base_id in base_ids],
        years,
        title=f"{concept} non-target high-activation sentence count heatmap ({qtag})",
        subtitle=f"Threshold mode: {threshold_mode}; values are yearly counts of sentences entering the high-activation range for each base.",
        output_path=analysis_dir / output_files["high_activation_count_heatmap"],
        cmap="YlOrRd",
        value_fmt=".0f",
    )
    plot_yearly_overview(
        yearly_overview,
        concept=concept,
        quantile_tag_text=qtag,
        output_path=analysis_dir / output_files["yearly_overview"],
    )
    top_pairs = plot_top_coactivation_pairs(
        pair_records,
        concept=concept,
        quantile_tag_text=qtag,
        output_path=analysis_dir / output_files["top_coactivation_pairs"],
    )

    summary = build_visualization_summary(
        concept=concept,
        years=years,
        base_ids=base_ids,
        quantile=quantile,
        threshold_mode=threshold_mode,
        yearly_overview=yearly_overview,
        top_pairs=top_pairs,
        output_files=output_files,
    )
    summary_path = analysis_dir / output_files["summary"]
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate non-target visualizations")
    parser.add_argument("--corpus", required=True, help="Corpus name")
    parser.add_argument("--concept", required=True, help="Concept name")
    parser.add_argument("--output_dir", default=None, help="Legacy output root")
    parser.add_argument("--run_root", default=None, help="Per-run root directory")
    parser.add_argument("--font_path", default=None, help="Chinese font path")
    args = parser.parse_args()

    files = plot_non_target_visualizations(
        corpus=args.corpus,
        concept=args.concept,
        output_dir=args.output_dir,
        font_path=args.font_path,
        run_root=args.run_root,
    )


if __name__ == "__main__":
    main()
