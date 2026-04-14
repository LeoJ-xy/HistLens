#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization entry script.

Supports two modes:
- --mode full: equivalent to the original visualize_full.py
- --mode simple: equivalent to the original visualize_simple.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine, pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import seaborn as sns
except Exception:
    sns = None

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_ROOT))

from configs.load import load_config
from configs.run_utils import get_active_model, resolve_run_root

# Global font properties object.
FONT_PROP = None


def setup_chinese_font(font_path=None):
    """Configure font support for Chinese text.

    Args:
        font_path: font file path

    Returns:
        A FontProperties object that can be attached explicitly where needed.
    """
    global FONT_PROP

    if font_path and os.path.exists(font_path):
        try:
            # Use the font file directly for the most reliable behavior.
            FONT_PROP = fm.FontProperties(fname=font_path)
            # Also update rcParams as a fallback.
            plt.rcParams["axes.unicode_minus"] = False  # Render minus signs correctly.
            return FONT_PROP
        except Exception as e:
            print(f"[WARNING] Font configuration warning: {e}")
            FONT_PROP = None
            plt.rcParams["axes.unicode_minus"] = False
            return None
    else:
        if font_path:
            print(f"[WARNING] Font file does not exist: {font_path}")
        FONT_PROP = None
        plt.rcParams["axes.unicode_minus"] = False
        return None


def get_font_prop():
    """Return the global font properties object."""
    return FONT_PROP


def normalize_top_bases(obj):
    if isinstance(obj, dict) and "top_bases" in obj:
        return obj["top_bases"]
    if isinstance(obj, dict):
        normalized = []
        for base_id, info in obj.items():
            if str(base_id).isdigit() and isinstance(info, dict):
                normalized.append({"base_id": int(base_id), **info})
        normalized.sort(key=lambda item: float(item.get("cum_drift", 0.0)), reverse=True)
        return normalized
    if isinstance(obj, list):
        return obj
    return []


def get_requested_top_n(top_bases_data, top_bases):
    if isinstance(top_bases_data, dict):
        raw_top_n = top_bases_data.get("top_n")
        if raw_top_n is not None:
            try:
                return int(raw_top_n)
            except (TypeError, ValueError):
                pass
    return len(top_bases)


def render_heatmap(matrix, years, cmap, value_fmt):
    if sns is not None:
        sns.heatmap(
            matrix,
            xticklabels=years,
            yticklabels=years,
            annot=True,
            cmap=cmap,
            fmt=value_fmt,
        )
        return

    plt.imshow(matrix, cmap=cmap, aspect="auto")
    plt.xticks(range(len(years)), years, rotation=45)
    plt.yticks(range(len(years)), years)
    plt.colorbar()
    for i in range(len(years)):
        for j in range(len(years)):
            plt.text(
                j,
                i,
                format(matrix[i, j], value_fmt),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )


def generate_distance_heatmaps(centers_path, output_dir, concept_name):
    """Generate distance heatmaps."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font_prop = get_font_prop()

    with open(centers_path, "r", encoding="utf-8") as f:
        centers_data = json.load(f)

    # Support multiple input layouts.
    if "centers" in centers_data:
        yearly_centers = centers_data["centers"]
    else:
        yearly_centers = centers_data

    # Keep only numeric year keys.
    year_keys = [k for k in yearly_centers.keys() if k.isdigit()]
    years = sorted([int(y) for y in year_keys])
    centers = np.array([yearly_centers[str(year)] for year in years])

    # Euclidean distance heatmap.
    euclidean_distances = squareform(pdist(centers, metric="euclidean"))

    plt.figure(figsize=(10, 8))
    render_heatmap(euclidean_distances, years, "YlOrRd", ".2f")
    title = f"{concept_name} - Euclidean distance heatmap across years"
    if font_prop:
        plt.title(title, fontproperties=font_prop)
    else:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / f"{concept_name}_euclidean_heatmap.png", dpi=300)
    plt.close()

    # Cosine distance heatmap.
    cosine_distances = squareform(pdist(centers, metric="cosine"))

    plt.figure(figsize=(10, 8))
    render_heatmap(cosine_distances, years, "YlGnBu", ".3f")
    title = f"{concept_name} - Cosine distance heatmap across years"
    if font_prop:
        plt.title(title, fontproperties=font_prop)
    else:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / f"{concept_name}_cosine_heatmap.png", dpi=300)
    plt.close()



def generate_trajectory_plot(centers_path, output_dir, concept_name):
    """Generate a yearly trajectory plot using t-SNE."""
    output_dir = Path(output_dir)
    font_prop = get_font_prop()

    with open(centers_path, "r", encoding="utf-8") as f:
        centers_data = json.load(f)

    # Support multiple input layouts.
    if "centers" in centers_data:
        yearly_centers = centers_data["centers"]
    else:
        yearly_centers = centers_data

    # Keep only numeric year keys.
    year_keys = [k for k in yearly_centers.keys() if k.isdigit()]
    years = sorted([int(y) for y in year_keys])
    centers = np.array([yearly_centers[str(year)] for year in years])

    # t-SNE projection.
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(years) - 1))
    centers_2d = tsne.fit_transform(centers)

    plt.figure(figsize=(12, 8))

    # Draw the trajectory.
    plt.plot(centers_2d[:, 0], centers_2d[:, 1], "o-", alpha=0.7, linewidth=2)

    # Annotate each year.
    for i, year in enumerate(years):
        plt.annotate(
            year,
            (centers_2d[i, 0], centers_2d[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            ha="left",
        )

    if font_prop:
        plt.title(f"{concept_name} - semantic trajectory by year (t-SNE)", fontproperties=font_prop)
        plt.xlabel("Dimension 1", fontproperties=font_prop)
        plt.ylabel("Dimension 2", fontproperties=font_prop)
    else:
        plt.title(f"{concept_name} - semantic trajectory by year (t-SNE)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{concept_name}_trajectory.png", dpi=300)
    plt.close()



def generate_activation_lines(distances_path, output_dir, concept_name):
    """Generate activation-distance line plots."""
    output_dir = Path(output_dir)
    font_prop = get_font_prop()

    try:
        with open(distances_path, "r", encoding="utf-8") as f:
            distances_data = json.load(f)

        # Support two layouts:
        # 1. {"1922->1923": {"euclidean": ..., "cosine": ...}, ...}
        # 2. {"1922": value, "1923": value, ...}

        if isinstance(list(distances_data.values())[0], dict):
            # Layout 1: year-pair keys with Euclidean and cosine distances.
            year_pairs = sorted(distances_data.keys())
            euclidean_dists = []
            cosine_dists = []
            years = []

            for pair in year_pairs:
                # Extract the starting year from each year pair.
                start_year = int(pair.split("->")[0])
                if not years or start_year != years[-1]:
                    years.append(start_year)
                euclidean_dists.append(distances_data[pair].get("euclidean", 0))
                cosine_dists.append(distances_data[pair].get("cosine", 0))

            # Plot both distances on dual y-axes.
            fig, ax1 = plt.subplots(figsize=(12, 6))

            ax1.plot(
                range(len(year_pairs)),
                euclidean_dists,
                "o-",
                linewidth=2,
                markersize=6,
                label="Euclidean distance",
                color="blue",
            )
            if font_prop:
                ax1.set_xlabel("Year pairs", fontproperties=font_prop)
                ax1.set_ylabel("Euclidean distance", fontproperties=font_prop, color="blue")
            else:
                ax1.set_xlabel("Year pairs")
                ax1.set_ylabel("Euclidean distance", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")
            ax1.set_xticks(range(len(year_pairs)))
            ax1.set_xticklabels(year_pairs, rotation=45)
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            ax2.plot(
                range(len(year_pairs)),
                cosine_dists,
                "s-",
                linewidth=2,
                markersize=6,
                label="Cosine distance",
                color="red",
            )
            if font_prop:
                ax2.set_ylabel("Cosine distance", fontproperties=font_prop, color="red")
            else:
                ax2.set_ylabel("Cosine distance", color="red")
            ax2.tick_params(axis="y", labelcolor="red")

            title = f"{concept_name} - center distance changes across years"
            if font_prop:
                plt.title(title, fontproperties=font_prop)
            else:
                plt.title(title)
            plt.tight_layout()
            plt.savefig(output_dir / f"{concept_name}_activation_lines.png", dpi=300)
            plt.close()
        else:
            # Layout 2: direct year -> value mapping.
            years = sorted([int(k) for k in distances_data.keys() if k.isdigit()])
            distances = [distances_data[str(y)] for y in years]

            plt.figure(figsize=(12, 6))
            plt.plot(years, distances, "o-", linewidth=2, markersize=6)
            if font_prop:
                plt.title(
                    f"{concept_name} - center distance changes across years", fontproperties=font_prop
                )
                plt.xlabel("Year", fontproperties=font_prop)
                plt.ylabel("Distance", fontproperties=font_prop)
            else:
                plt.title(f"{concept_name} - center distance changes across years")
                plt.xlabel("Year")
                plt.ylabel("Distance")
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / f"{concept_name}_activation_lines.png", dpi=300)
            plt.close()

    except Exception as e:
        print(f"[WARNING] Failed to generate activation-distance line plot: {e}")
        print("[WARNING] Skipping activation-distance lines and continuing")
        import traceback

        traceback.print_exc()


def compare_concepts(centers_paths, output_dir, concept_names):
    """Compare semantic trajectories across concepts."""
    output_dir = Path(output_dir)
    font_prop = get_font_prop()

    if len(centers_paths) != len(concept_names):
        print("[ERROR] centers_paths and concept_names must have the same length")
        return

    # Load center data for each concept.
    all_centers_data = {}
    for i, path in enumerate(centers_paths):
        if Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                all_centers_data[concept_names[i]] = json.load(f)

    if len(all_centers_data) < 2:
        print("[WARNING] At least two concepts are required for comparison")
        return

    # Find the shared years across concepts.
    common_years = set.intersection(*[set(data.keys()) for data in all_centers_data.values()])
    common_years = sorted(common_years)

    if len(common_years) == 0:
        print("[WARNING] No shared years found across concepts")
        return

    # Compute inter-concept distances.
    concept_pairs = []
    for i in range(len(concept_names)):
        for j in range(i + 1, len(concept_names)):
            concept_pairs.append((concept_names[i], concept_names[j]))

    for concept1, concept2 in concept_pairs:
        if concept1 in all_centers_data and concept2 in all_centers_data:
            euclidean_dists = []
            cosine_dists = []

            for year in common_years:
                if year in all_centers_data[concept1] and year in all_centers_data[concept2]:
                    center1 = np.array(all_centers_data[concept1][year])
                    center2 = np.array(all_centers_data[concept2][year])

                    euclidean_dist = np.linalg.norm(center1 - center2)
                    cosine_dist = cosine(center1, center2)

                    euclidean_dists.append(euclidean_dist)
                    cosine_dists.append(cosine_dist)

            # Plot the comparison figure.
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            ax1.plot(common_years, euclidean_dists, "o-", linewidth=2, label="Euclidean distance")
            if font_prop:
                ax1.set_title(
                    f"{concept1} vs {concept2} - Euclidean distance changes",
                    fontproperties=font_prop,
                )
                ax1.set_xlabel("Year", fontproperties=font_prop)
                ax1.set_ylabel("Euclidean distance", fontproperties=font_prop)
            else:
                ax1.set_title(f"{concept1} vs {concept2} - Euclidean distance changes")
                ax1.set_xlabel("Year")
                ax1.set_ylabel("Euclidean distance")
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis="x", rotation=45)

            ax2.plot(
                common_years, cosine_dists, "o-", color="red", linewidth=2, label="Cosine distance"
            )
            if font_prop:
                ax2.set_title(
                    f"{concept1} vs {concept2} - Cosine distance changes",
                    fontproperties=font_prop,
                )
                ax2.set_xlabel("Year", fontproperties=font_prop)
                ax2.set_ylabel("Cosine distance", fontproperties=font_prop)
            else:
                ax2.set_title(f"{concept1} vs {concept2} - Cosine distance changes")
                ax2.set_xlabel("Year")
                ax2.set_ylabel("Cosine distance")
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis="x", rotation=45)

            plt.tight_layout()
            filename = f"{concept1}_vs_{concept2}_comparison.png"
            plt.savefig(output_dir / filename, dpi=300)
            plt.close()



def save_top10_yearly_activations(
    centers_path, top_drift_bases_path, output_dir, concept_name
):
    """Extract and save yearly activations for the Top-N drift bases."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load yearly center data.
    with open(centers_path, "r", encoding="utf-8") as f:
        centers_data = json.load(f)

    # Load top drift bases.
    with open(top_drift_bases_path, "r", encoding="utf-8") as f:
        top_bases_data = json.load(f)

    top_bases = normalize_top_bases(top_bases_data)
    requested_top_n = get_requested_top_n(top_bases_data, top_bases)
    base_meta = {int(item["base_id"]): item for item in top_bases}

    base_order = centers_data["base_order"]
    centers = centers_data["centers"]

    # Build a base_id -> center index mapping.
    base_to_idx = {int(bid): idx for idx, bid in enumerate(base_order)}

    # Extract yearly activations for the selected drift bases.
    top10_yearly_activations = {}
    top_base_ids = [int(item["base_id"]) for item in top_bases]

    years = sorted([int(y) for y in centers.keys()])

    for base_id in top_base_ids:
        if base_id not in base_to_idx:
            print(f"[WARNING] Base {base_id} is not present in base_order, skipping")
            continue

        idx = base_to_idx[base_id]
        yearly_values = {}

        for year in years:
            year_str = str(year)
            if year_str in centers:
                activation_value = centers[year_str][idx]
                yearly_values[year] = float(activation_value)

        meta = base_meta.get(base_id, {})
        top10_yearly_activations[str(base_id)] = {
            "base_id": base_id,
            "cum_drift": meta.get("cum_drift"),
            "peak_delta": meta.get("peak_delta"),
            "peak_years": meta.get("years"),
            "yearly_activations": yearly_values,
        }

    rendered_top_n = len(top10_yearly_activations) or len(top_bases)
    if rendered_top_n != requested_top_n:
        print(
            f"[WARNING] {concept_name} Top-N visualization count does not match the drift metadata: "
            f"requested={requested_top_n}, rendered={rendered_top_n}"
        )

    # Save the extracted yearly activations as JSON.
    output_file = output_dir / f"{concept_name}_top{rendered_top_n}_yearly_activations.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(top10_yearly_activations, f, ensure_ascii=False, indent=2)

    return top10_yearly_activations, years, {
        "requested_top_n": requested_top_n,
        "rendered_top_n": rendered_top_n,
        "output_file": str(output_file),
    }


def plot_top10_yearly_activations(top10_data, years, output_dir, concept_name, top_n=None):
    """Plot yearly activations for the Top-N drift bases on one figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font_prop = get_font_prop()
    rendered_top_n = int(top_n or len(top10_data) or 0)

    plt.figure(figsize=(14, 8))

    # Draw one line per base.
    colors = plt.cm.Set3(np.linspace(0, 1, len(top10_data)))

    for i, (base_id_str, base_info) in enumerate(top10_data.items()):
        base_id = base_info["base_id"]
        yearly_activations = base_info["yearly_activations"]

        # Collect yearly activations for the current base.
        activation_values = [yearly_activations.get(year, 0.0) for year in years]

        # Plot the yearly series.
        plt.plot(
            years,
            activation_values,
            "o-",
            label=f'Base {base_id} (Δ={base_info["cum_drift"]:.2f})',
            color=colors[i],
            linewidth=2,
            markersize=6,
            alpha=0.8,
        )

    if font_prop:
        plt.title(
            f"{concept_name} - yearly activation changes for Top {rendered_top_n} bases",
            fontproperties=font_prop,
            fontsize=16,
        )
        plt.xlabel("Year", fontproperties=font_prop, fontsize=12)
        plt.ylabel("Activation", fontproperties=font_prop, fontsize=12)
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=9,
            prop=font_prop,
        )
    else:
        plt.title(f"{concept_name} - yearly activation changes for Top {rendered_top_n} bases", fontsize=16)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Activation", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xticks(years, rotation=45)
    plt.tight_layout()

    output_file = output_dir / f"{concept_name}_top{rendered_top_n}_yearly_activations.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()



def generate_analysis_report(config, output_dir, top_base_count=None):
    """Generate a markdown analysis report."""
    output_dir = Path(output_dir)
    report_path = output_dir / "analysis_report.md"

    target_word = config.get("target_word", "concept")
    top_base_count = int(top_base_count) if top_base_count else None
    top_heading = (
        f"### Top {top_base_count} Drift Bases"
        if top_base_count
        else "### Top Drift Bases"
    )
    top_json_name = (
        f"{target_word}_top{top_base_count}_yearly_activations.json"
        if top_base_count
        else f"{target_word}_topN_yearly_activations.json"
    )
    top_png_name = (
        f"{target_word}_top{top_base_count}_yearly_activations.png"
        if top_base_count
        else f"{target_word}_topN_yearly_activations.png"
    )
    top_method_text = (
        f"5. **Drift-base analysis**: extract and visualize yearly activations for the top {top_base_count} most drifted bases"
        if top_base_count
        else "5. **Drift-base analysis**: extract and visualize yearly activations for the most drifted bases"
    )
    top_findings_text = (
        f"- Activation-pattern changes for the Top {top_base_count} drift bases"
        if top_base_count
        else "- Activation-pattern changes for the key drift bases"
    )

    report_content = f"""# Semantic Drift Report for {target_word}

## Overview
This report uses SAE (Sparse Autoencoder) outputs to analyze semantic drift for "{target_word}" over time.

## Generated Files
### Distance Heatmaps
- `{target_word}_euclidean_heatmap.png` - Euclidean distance heatmap across years
- `{target_word}_cosine_heatmap.png` - Cosine distance heatmap across years

### Trajectory Visualizations
- `{target_word}_trajectory.png` - semantic trajectory by year (t-SNE)
- `{target_word}_activation_lines.png` - activation-distance line plot

{top_heading}
- `{top_json_name}` - yearly activation data for drift bases
- `{top_png_name}` - yearly activation plot for drift bases

### Concept Comparisons
- `*_vs_*_comparison.png` - cross-concept comparison plots

## Method
1. **SAE feature extraction**: use a pretrained SAE model to extract semantic features
2. **Yearly center computation**: compute the semantic center for each year
3. **Distance measurement**: use Euclidean and cosine distance to quantify semantic change
4. **Dimensionality reduction**: use t-SNE to visualize the high-dimensional space
{top_method_text}

## Main Findings
- Time-localized changes in semantic drift intensity
- Evolution of semantic relationships across concepts
- Structural shifts around key years
{top_findings_text}

## Technical Notes
- SAE model: TranSirius_OpenSAE-LLaMA-3.1-Layer_29
- Base model: LLaMA-3.1-8B
- Distance metrics: Euclidean distance, cosine distance
- Visualization methods: t-SNE, heatmaps, line plots

Generated on: {np.datetime64("today")}
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)



def generate_distance_analysis(centers_path, output_dir, concept_name):
    """Generate a compact distance-analysis figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font_prop = get_font_prop()

    with open(centers_path, "r", encoding="utf-8") as f:
        centers_data = json.load(f)

    # Access the expected centers layout.
    years = sorted(centers_data["centers"].keys())

    # Compute changes between adjacent years.
    distances = []
    year_pairs = []

    for i in range(len(years) - 1):
        year1, year2 = years[i], years[i + 1]
        center1 = np.array(centers_data["centers"][year1])
        center2 = np.array(centers_data["centers"][year2])

        # Compute Euclidean distance.
        euclidean_dist = np.linalg.norm(center1 - center2)
        # Compute cosine distance.
        cosine_dist = cosine(center1, center2)

        distances.append(
            {"year_pair": f"{year1}-{year2}", "euclidean": euclidean_dist, "cosine": cosine_dist}
        )
        year_pairs.append(f"{year1}-{year2}")

    # Plot the distance changes.
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    euclidean_dists = [d["euclidean"] for d in distances]
    plt.plot(range(len(year_pairs)), euclidean_dists, marker="o", linewidth=2, markersize=8)
    if font_prop:
        plt.title(f"{concept_name} - Euclidean distance changes", fontproperties=font_prop)
        plt.xlabel("Year pairs", fontproperties=font_prop)
        plt.ylabel("Euclidean distance", fontproperties=font_prop)
    else:
        plt.title(f"{concept_name} - Euclidean Distance Changes")
        plt.xlabel("Year Pairs")
        plt.ylabel("Euclidean Distance")
    plt.xticks(range(len(year_pairs)), year_pairs, rotation=45)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    cosine_dists = [d["cosine"] for d in distances]
    plt.plot(
        range(len(year_pairs)), cosine_dists, marker="s", color="red", linewidth=2, markersize=8
    )
    if font_prop:
        plt.title(f"{concept_name} - Cosine distance changes", fontproperties=font_prop)
        plt.xlabel("Year pairs", fontproperties=font_prop)
        plt.ylabel("Cosine distance", fontproperties=font_prop)
    else:
        plt.title(f"{concept_name} - Cosine Distance Changes")
        plt.xlabel("Year Pairs")
        plt.ylabel("Cosine Distance")
    plt.xticks(range(len(year_pairs)), year_pairs, rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"{concept_name}_distance_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save the computed distances.
    with open(output_dir / f"{concept_name}_distances.json", "w", encoding="utf-8") as f:
        json.dump(distances, f, ensure_ascii=False, indent=2)

    return distances


def generate_dimensionality_reduction(centers_path, output_dir, concept_name):
    """Generate dimensionality-reduction visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font_prop = get_font_prop()

    with open(centers_path, "r", encoding="utf-8") as f:
        centers_data = json.load(f)

    years = sorted(centers_data["centers"].keys())
    centers_matrix = np.array([centers_data["centers"][year] for year in years])

    # PCA projection.
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(centers_matrix)

    # Create the trajectory figure.
    plt.figure(figsize=(12, 5))

    # PCA trajectory.
    plt.subplot(1, 2, 1)
    plt.plot(pca_result[:, 0], pca_result[:, 1], "o-", linewidth=2, markersize=10)
    for i, year in enumerate(years):
        plt.annotate(
            year,
            (pca_result[i, 0], pca_result[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=12,
        )
    if font_prop:
        plt.xlabel(f"PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})", fontproperties=font_prop)
        plt.ylabel(f"PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})", fontproperties=font_prop)
        plt.title(f"{concept_name} - PCA trajectory", fontproperties=font_prop)
    else:
        plt.xlabel(f"PC1 (Var: {pca.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"PC2 (Var: {pca.explained_variance_ratio_[1]:.2%})")
        plt.title(f"{concept_name} - PCA Trajectory")
    plt.grid(True, alpha=0.3)

    # If enough points exist, also try t-SNE.
    if len(years) >= 4:  # t-SNE needs enough samples.
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(3, len(years) - 1))
            tsne_result = tsne.fit_transform(centers_matrix)

            plt.subplot(1, 2, 2)
            plt.plot(tsne_result[:, 0], tsne_result[:, 1], "s-", color="red", linewidth=2, markersize=10)
            for i, year in enumerate(years):
                plt.annotate(
                    year,
                    (tsne_result[i, 0], tsne_result[i, 1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=12,
                )
            if font_prop:
                plt.xlabel("t-SNE dimension 1", fontproperties=font_prop)
                plt.ylabel("t-SNE dimension 2", fontproperties=font_prop)
                plt.title(f"{concept_name} - t-SNE trajectory", fontproperties=font_prop)
            else:
                plt.xlabel("t-SNE Dim 1")
                plt.ylabel("t-SNE Dim 2")
                plt.title(f"{concept_name} - t-SNE Trajectory")
            plt.grid(True, alpha=0.3)
        except Exception:
            # Fall back to explained variance if t-SNE fails.
            plt.subplot(1, 2, 2)
            explained_var = pca.explained_variance_ratio_[:10]  # First 10 principal components.
            plt.bar(range(1, len(explained_var) + 1), explained_var)
            if font_prop:
                plt.xlabel("Principal component", fontproperties=font_prop)
                plt.ylabel("Explained variance ratio", fontproperties=font_prop)
                plt.title(f"{concept_name} - PCA variance contribution", fontproperties=font_prop)
            else:
                plt.xlabel("Principal Component")
                plt.ylabel("Explained Variance Ratio")
                plt.title(f"{concept_name} - PCA Variance")
    else:
        # Too few points for t-SNE: show explained variance instead.
        plt.subplot(1, 2, 2)
        pca_full = PCA()
        pca_full.fit(centers_matrix)
        explained_var = pca_full.explained_variance_ratio_[:10]
        plt.bar(range(1, len(explained_var) + 1), explained_var)
        if font_prop:
            plt.xlabel("Principal component", fontproperties=font_prop)
            plt.ylabel("Explained variance ratio", fontproperties=font_prop)
            plt.title(f"{concept_name} - PCA variance contribution", fontproperties=font_prop)
        else:
            plt.xlabel("Principal Component")
            plt.ylabel("Explained Variance Ratio")
            plt.title(f"{concept_name} - PCA Variance")

    plt.tight_layout()
    plt.savefig(output_dir / f"{concept_name}_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_drift_analysis(drift_file, output_dir, concept_name):
    """Analyze drift patterns from the drift summary."""
    output_dir = Path(output_dir)
    font_prop = get_font_prop()

    with open(drift_file, "r", encoding="utf-8") as f:
        drift_data = json.load(f)

    # Extract base IDs and drift metadata.
    base_ids = []
    drift_values = []
    peak_years = []

    # Handle dict-style drift files.
    if isinstance(drift_data, dict):
        for base_id, info in drift_data.items():
            base_ids.append(base_id)
            drift_values.append(info["cum_drift"])
            # Extract peak years from the years list.
            if "years" in info and len(info["years"]) >= 2:
                peak_years.append(f"{info['years'][0]}-{info['years'][1]}")
            else:
                peak_years.append("Unknown")
    else:
        # Handle list-style drift files.
        for item in drift_data:
            base_ids.append(item["base_id"])
            drift_values.append(item["cumulative_drift"])
            peak_years.append(f"{item['peak_year1']}-{item['peak_year2']}")

    # Plot the drift analysis.
    plt.figure(figsize=(14, 6))

    # Drift magnitude distribution.
    plt.subplot(1, 2, 1)
    plt.bar(range(len(base_ids)), drift_values, color="skyblue", alpha=0.7)
    if font_prop:
        plt.xlabel("Base rank", fontproperties=font_prop)
        plt.ylabel("Cumulative drift", fontproperties=font_prop)
        plt.title(f"{concept_name} - Top-{len(base_ids)} drift bases", fontproperties=font_prop)
    else:
        plt.xlabel("Base Vector Rank")
        plt.ylabel("Cumulative Drift")
        plt.title(f"{concept_name} - Top-{len(base_ids)} Drift Bases")
    plt.xticks(range(len(base_ids)), [f"#{i + 1}" for i in range(len(base_ids))])

    # Peak-year distribution.
    plt.subplot(1, 2, 2)
    peak_year_counts = {}
    for year_pair in peak_years:
        peak_year_counts[year_pair] = peak_year_counts.get(year_pair, 0) + 1

    years = list(peak_year_counts.keys())
    counts = list(peak_year_counts.values())
    plt.bar(years, counts, color="lightcoral", alpha=0.7)
    if font_prop:
        plt.xlabel("Peak year pair", fontproperties=font_prop)
        plt.ylabel("Base count", fontproperties=font_prop)
        plt.title(f"{concept_name} - peak drift-year distribution", fontproperties=font_prop)
    else:
        plt.xlabel("Peak Year Pair")
        plt.ylabel("Base Vector Count")
        plt.title(f"{concept_name} - Peak Year Distribution")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / f"{concept_name}_drift_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_summary_report(output_dir, concept_name, distances):
    """Create a short markdown summary report."""
    output_dir = Path(output_dir)

    # Find the years with the largest changes.
    max_euclidean = max(distances, key=lambda x: x["euclidean"])
    max_cosine = max(distances, key=lambda x: x["cosine"])

    report = f"""
# Semantic Drift Summary for {concept_name}

## Overview
This report summarizes semantic drift patterns for "{concept_name}" across years.

## Key Findings

### Largest Changes
- **Largest Euclidean change**: {max_euclidean['year_pair']} (distance: {max_euclidean['euclidean']:.4f})
- **Largest cosine change**: {max_cosine['year_pair']} (distance: {max_cosine['cosine']:.4f})

### Year-to-Year Trend
"""

    for dist_info in distances:
        report += (
            f"- {dist_info['year_pair']}: Euclidean {dist_info['euclidean']:.4f}, "
            f"cosine {dist_info['cosine']:.4f}\n"
        )

    report += f"""
## Visualization Files
- Distance analysis: {concept_name}_distance_analysis.png
- Semantic trajectory: {concept_name}_trajectory.png
- Drift analysis: {concept_name}_drift_analysis.png

## Data Files
- Distance data: {concept_name}_distances.json
"""

    with open(output_dir / f"{concept_name}_report.md", "w", encoding="utf-8") as f:
        f.write(report)


def run_full_mode(config):
    target_word = config["target_word"]
    output_base = Path(config["output_dir"]) / target_word
    centers_path = output_base / "yearly_centers.json"
    distances_path = output_base / "yearly_distances.json"
    top_drift_bases_path = output_base / "top_drift_bases.json"
    viz_dir = output_base / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Configure the Chinese-capable font if a path is provided.
    font_path = config.get("font_path", None)
    setup_chinese_font(font_path)

    # Validate required inputs.
    if not centers_path.exists():
        print(f"[ERROR] Center data file not found: {centers_path}")
        return

    # 1. Distance heatmaps.
    generate_distance_heatmaps(centers_path, viz_dir, target_word)

    # 2. Yearly trajectory plot.
    generate_trajectory_plot(centers_path, viz_dir, target_word)

    # 3. Activation-distance line plot (optional).
    if distances_path.exists():
        try:
            generate_activation_lines(distances_path, viz_dir, target_word)
        except Exception as e:
            print(f"[WARNING] Activation-distance line plot failed; continuing: {e}")

    # 4. Save and plot yearly activations for drift bases.
    top_base_count = None
    if top_drift_bases_path.exists():
        try:
            top10_data, years, top_meta = save_top10_yearly_activations(
                centers_path, top_drift_bases_path, viz_dir, target_word
            )
            top_base_count = top_meta["rendered_top_n"]
            plot_top10_yearly_activations(
                top10_data, years, viz_dir, target_word, top_n=top_base_count
            )
        except Exception as e:
            print(f"[ERROR] Failed while processing yearly activations for Top-N bases: {e}")
            import traceback

            traceback.print_exc()
    else:
        print(f"[WARNING] top_drift_bases.json not found: {top_drift_bases_path}")
        print("[WARNING] Skipping Top-N yearly activation analysis")

    # 5. Cross-concept comparisons when additional concepts are available.
    concept_paths = config.get("compare_concepts", {})
    if concept_paths:
        centers_paths = [centers_path]  # Current concept.
        concept_names = [target_word]

        for concept_name, concept_path in concept_paths.items():
            if Path(concept_path).exists():
                centers_paths.append(concept_path)
                concept_names.append(concept_name)

        if len(concept_names) > 1:
            compare_concepts(centers_paths, viz_dir, concept_names)

    # 6. Markdown report.
    generate_analysis_report(config, viz_dir, top_base_count=top_base_count)



def run_simple_mode(config):
    corpus = config.get("corpus", None)
    concept_name = config["target_word"]
    output_base = Path(config["output_dir"])

    # Support both legacy and current directory layouts.
    if corpus:
        # Current layout: output/<concept>
        concept_output = output_base / concept_name
    else:
        # Legacy layout: output/<concept>
        concept_output = output_base / concept_name
    centers_path = concept_output / "yearly_centers.json"
    drift_file = concept_output / "top_drift_bases.json"
    viz_dir = concept_output / "visualizations"

    # Configure the Chinese-capable font if a path is provided.
    font_path = config.get("font_path", None)
    setup_chinese_font(font_path)

    try:
        # 1. Distance analysis.
        distances = generate_distance_analysis(centers_path, viz_dir, concept_name)

        # 2. Dimensionality-reduction plots.
        generate_dimensionality_reduction(centers_path, viz_dir, concept_name)

        # 3. Drift analysis.
        if drift_file.exists():
            generate_drift_analysis(drift_file, viz_dir, concept_name)

        # 4. Summary report.
        create_summary_report(viz_dir, concept_name, distances)

    except Exception as e:
        print(f"[ERROR] Visualization pipeline failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Visualization entry point")
    parser.add_argument("--config", default="configs/exp/default.yaml", help="Config file path")
    parser.add_argument("--mode", choices=["full", "simple"], default="full")
    parser.add_argument("--target-word", default=None, help="Override the target word from config")
    parser.add_argument("--output-root", default=None, help="Override the output root")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.suffix.lower() == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        overrides = {}
        if args.output_root:
            overrides.setdefault("paths", {})["output_root"] = args.output_root
        cfg = load_config(args.config, overrides=overrides or None)
        target_word = (
            args.target_word
            or cfg.experiment.target_word
            or (cfg.experiment.words[0] if cfg.experiment.words else None)
        )
        if not target_word:
            raise ValueError("Missing target word. Set experiment.target_word or experiment.words in the config.")
        model = get_active_model(cfg)
        output_dir = resolve_run_root(cfg, model)
        config = {
            "corpus": cfg.experiment.corpus,
            "target_word": target_word,
            "output_dir": str(output_dir),
            "font_path": cfg.paths.font_path,
            "compare_concepts": cfg.experiment.compare_concepts,
        }

    if args.mode == "simple":
        run_simple_mode(config)
    else:
        run_full_mode(config)


if __name__ == "__main__":
    main()
