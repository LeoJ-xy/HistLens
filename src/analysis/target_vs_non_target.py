#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare diachronic activations between target and non-target sentences on key bases.

Inputs:
- {output_dir}/{corpus}/{concept}/non_target_analysis/{concept}_annual_means_summary.json
- {output_dir}/{corpus}/{concept}/top_drift_bases.json

Output:
- {output_dir}/{concept}/visualizations/{concept}_target_vs_non_target_means.png

Usage:
    python target_vs_non_target.py \
        --corpus "<corpus_name>" \
        --concept "<concept_name>" \
        --output_dir "<output_dir>"

    # With the newer output layout, pass the per-run root directory instead.
    python target_vs_non_target.py \
        --corpus "<corpus_name>" \
        --concept "<concept_name>" \
        --run_root "<run_root>"
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Global font properties object.
FONT_PROP = None

def setup_chinese_font(font_path=None):
    """Configure font support for Chinese text.

    Returns:
        A FontProperties object that can be attached explicitly where needed.
    """
    global FONT_PROP
    
    if font_path and os.path.exists(font_path):
        try:
            # Use the font file directly for the most reliable behavior.
            FONT_PROP = font_manager.FontProperties(fname=font_path)
            # Also update rcParams as a fallback.
            plt.rcParams['axes.unicode_minus'] = False  # Render minus signs correctly.
            return FONT_PROP
        except Exception as e:
            print(f"[WARNING] Font configuration warning: {e}")
            FONT_PROP = None
            plt.rcParams['axes.unicode_minus'] = False
            return None
    else:
        if font_path:
            print(f"[WARNING] Font file does not exist: {font_path}")
        FONT_PROP = None
        plt.rcParams['axes.unicode_minus'] = False
        return None

def get_font_prop():
    """Return the global font properties object."""
    return FONT_PROP


def load_json(path: str):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_key_bases(top_drift_bases: Dict[str, Dict]) -> List[str]:
    """
    Return base IDs from top_drift_bases.json, sorted by cumulative drift.
    """
    if isinstance(top_drift_bases, dict) and "top_bases" in top_drift_bases:
        items = []
        for item in top_drift_bases.get("top_bases", []):
            if not isinstance(item, dict):
                continue
            base_id = item.get("base_id")
            if base_id is None:
                continue
            items.append((str(base_id), float(item.get("cum_drift", 0.0))))
        items.sort(key=lambda x: x[1], reverse=True)
        return [base_id for base_id, _ in items]

    # Backward compatibility: keys are base IDs and values include cum_drift metadata.
    items = []
    for base_id, info in top_drift_bases.items():
        if not isinstance(info, dict):
            continue
        items.append((str(base_id), float(info.get("cum_drift", 0.0))))
    items.sort(key=lambda x: x[1], reverse=True)
    return [base_id for base_id, _ in items]


def choose_subplot_layout(n: int) -> tuple[int, int]:
    if n <= 1:
        return 1, 1
    if n <= 10:
        ncols = 2
    elif n <= 18:
        ncols = 3
    else:
        ncols = 4
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols


def prepare_series(
    annual_summary: Dict[str, Dict], base_id: str
) -> Dict[str, List[float]]:
    """
    Build yearly series for a single base:
    - years: ascending years
    - with_target: yearly with_target_mean values
    - without_target: yearly without_target_mean values
    - diff: with_target_mean - without_target_mean
    """
    years = sorted(int(y) for y in annual_summary.keys())
    with_vals: List[float] = []
    without_vals: List[float] = []
    diff_vals: List[float] = []

    for y in years:
        year_str = str(y)
        year_entry = annual_summary.get(year_str, {})
        base_entry = year_entry.get(base_id)
        if not base_entry:
            # Fill missing year/base combinations with zero to keep lengths aligned.
            with_vals.append(0.0)
            without_vals.append(0.0)
            diff_vals.append(0.0)
            continue

        with_mean = float(base_entry.get("with_target_mean", 0.0))
        without_mean = float(base_entry.get("without_target_mean", 0.0))
        with_vals.append(with_mean)
        without_vals.append(without_mean)
        diff_vals.append(with_mean - without_mean)

    return {
        "years": years,
        "with_target": with_vals,
        "without_target": without_vals,
        "diff": diff_vals,
    }


def resolve_concept_dir(output_dir: str, corpus: str, concept: str) -> Path:
    base_dir = Path(output_dir)
    candidate = base_dir / corpus / concept
    if candidate.exists():
        return candidate
    candidate = base_dir / concept
    if candidate.exists():
        return candidate
    return base_dir / corpus / concept


def plot_target_vs_non_target(
    corpus: str,
    concept: str,
    output_dir: str | None,
    font_path: str = None,
    run_root: str = None,
):
    """
    Plot activation comparisons between target and non-target sentences.

    Args:
        corpus: corpus source
        concept: target concept
        output_dir: output root
        font_path: font path
    """
    # Configure font support.
    setup_chinese_font(font_path)

    # Build paths.
    if run_root:
        base_dir = Path(run_root) / concept
    else:
        if not output_dir:
            raise ValueError("output_dir and run_root cannot both be empty.")
        base_dir = resolve_concept_dir(output_dir, corpus, concept)
    annual_summary_path = base_dir / "non_target_analysis" / f"{concept}_annual_means_summary.json"
    top_drift_bases_path = base_dir / "top_drift_bases.json"
    output_fig_path = base_dir / "visualizations" / f"{concept}_target_vs_non_target_means.png"
    
    # Validate required inputs.
    if not annual_summary_path.exists():
        raise FileNotFoundError(f"Yearly summary file not found: {annual_summary_path}")
    
    if not top_drift_bases_path.exists():
        raise FileNotFoundError(f"Top drift bases file not found: {top_drift_bases_path}")
    
    # Load input data.
    annual_summary = load_json(str(annual_summary_path))
    top_drift_bases = load_json(str(top_drift_bases_path))
    key_bases = get_key_bases(top_drift_bases)

    if not key_bases:
        raise RuntimeError("No base information found in top_drift_bases.json.")

    n = len(key_bases)
    nrows, ncols = choose_subplot_layout(n)
    legend_fontsize = 7 if n > 10 else 8

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6.2 * ncols, 3.0 * nrows + 0.6),
        sharex=True,
    )
    
    # Normalize axes into a flat list of Axes objects.
    if n == 1:
        axes_list = [axes]
    else:
        if hasattr(axes, 'flatten'):
            axes_flat = axes.flatten()
            axes_list = [axes_flat[i] for i in range(len(axes_flat))]
        elif hasattr(axes, '__iter__') and not isinstance(axes, str):
            axes_list = []
            for item in axes:
                if hasattr(item, '__iter__') and not isinstance(item, str):
                    axes_list.extend(item)
                else:
                    axes_list.append(item)
        else:
            axes_list = [axes]
    
    axes = axes_list

    font_prop = get_font_prop()
    
    for idx, base_id in enumerate(key_bases):
        ax = axes[idx]
        series = prepare_series(annual_summary, base_id)
        years = series["years"]
        with_vals = series["with_target"]
        without_vals = series["without_target"]
        diff_vals = series["diff"]

        ax.plot(years, with_vals, marker="o", label=f'Mean for sentences containing "{concept}"')
        ax.plot(years, without_vals, marker="s", label=f'Mean for sentences without "{concept}"')
        ax_twin = ax.twinx()
        ax_twin.plot(
            years,
            diff_vals,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label="Difference (with - without)",
        )

        if font_prop:
            ax.set_title(f"Base {base_id} (#{idx + 1})", fontproperties=font_prop)
            ax.set_xlabel("Year", fontproperties=font_prop)
            ax.set_ylabel("Mean activation", fontproperties=font_prop)
            ax_twin.set_ylabel("Difference", fontproperties=font_prop)
        else:
            ax.set_title(f"Base {base_id} (#{idx + 1})")
            ax.set_xlabel("Year")
            ax.set_ylabel("Mean activation")
            ax_twin.set_ylabel("Difference")

        # Merge legends per subplot to avoid a crowded global legend.
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        if font_prop:
            ax.legend(
                lines + lines2,
                labels + labels2,
                fontsize=legend_fontsize,
                loc="upper left",
                prop=font_prop,
            )
        else:
            ax.legend(
                lines + lines2,
                labels + labels2,
                fontsize=legend_fontsize,
                loc="upper left",
            )

    # Hide unused subplots.
    for j in range(len(key_bases), len(axes)):
        axes[j].axis("off")

    title = f'{corpus}: "{concept}" target vs non-target activations over time (Top {n} drift bases)'
    if font_prop:
        fig.suptitle(title, fontsize=14, fontproperties=font_prop)
    else:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    # Ensure the output directory exists.
    output_fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_fig_path), dpi=200)


def main():
    parser = argparse.ArgumentParser(
        description="Compare diachronic activations between target and non-target sentences on key bases"
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Corpus source"
    )
    parser.add_argument(
        "--concept",
        required=True,
        help="Target concept"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Legacy output root"
    )
    parser.add_argument(
        "--run_root",
        default=None,
        help="Per-run root directory (optional)"
    )
    parser.add_argument(
        "--font_path",
        default=None,
        help="Font file path"
    )
    
    args = parser.parse_args()
    
    try:
        plot_target_vs_non_target(
            corpus=args.corpus,
            concept=args.concept,
            output_dir=args.output_dir,
            font_path=args.font_path,
            run_root=args.run_root,
        )
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
