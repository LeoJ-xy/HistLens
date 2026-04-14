#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STEP-5 (generic)
Generate word clouds and top-30 sentence exports.
"""
import argparse
from collections import Counter
import json
import heapq
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    import jieba
except Exception:
    jieba = None

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

def filter_text(text):
    # Filter Arabic numerals.
    text = re.sub(r'\d+', '', text)
    # Filter Chinese numerals (one through ten).
    text = re.sub(r'[一二三四五六七八九十]+', '', text)
    return text


def setup_font(font_path):
    if font_path and Path(font_path).exists():
        try:
            return fm.FontProperties(fname=font_path)
        except Exception:
            return None
    return None


def tokenize_fallback(text):
    cleaned = filter_text(text)
    if jieba is not None:
        tokens = [tok.strip() for tok in jieba.lcut(cleaned) if tok.strip()]
    else:
        tokens = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z]+", cleaned)
    return [tok for tok in tokens if len(tok) >= 2]


def save_frequency_plot(text_corpus, img_path, font_path, base_id):
    tokens = tokenize_fallback(text_corpus)
    freq = Counter(tokens)
    if not freq:
        print(f"[WARN] Base {base_id} has no tokens for the fallback frequency chart, skipping image")
        return

    top_items = freq.most_common(20)
    labels = [item[0] for item in top_items][::-1]
    values = [item[1] for item in top_items][::-1]
    font_prop = setup_font(font_path)

    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    ax.barh(range(len(labels)), values, color="#4C72B0", alpha=0.9)
    ax.set_yticks(range(len(labels)))
    if font_prop:
        ax.set_yticklabels(labels, fontproperties=font_prop)
        ax.set_title(f"Base {base_id} token frequency chart", fontproperties=font_prop)
        ax.set_xlabel("Count", fontproperties=font_prop)
    else:
        ax.set_yticklabels(labels)
        ax.set_title(f"Base {base_id} token frequencies")
        ax.set_xlabel("Count")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close()

def generate_wordclouds(target_word, act_dir, drift_f, txt_dir, wc_dir, font_path):
    act_dir = Path(act_dir)
    drift_f = Path(drift_f)
    txt_dir = Path(txt_dir)
    wc_dir = Path(wc_dir)
    txt_dir.mkdir(parents=True, exist_ok=True)
    wc_dir.mkdir(parents=True, exist_ok=True)

    with open(drift_f, "r", encoding="utf-8") as f:
        drift = json.load(f)

    def normalize_top_bases(obj):
        if isinstance(obj, dict) and "top_bases" in obj:
            return obj["top_bases"]
        if isinstance(obj, dict):
            return [{"base_id": int(bid), **info} for bid, info in obj.items()]
        return []

    top_bases = normalize_top_bases(drift)
    top_bids = [int(item["base_id"]) for item in top_bases]
    all_acts = {bid: [] for bid in top_bids}

    for fn in act_dir.glob(f"{target_word}_activations_*.jsonl"):
        # Filenames look like {word}_activations_1915.jsonl.
        year_match = re.search(r"(\d{4})", fn.stem)
        year = int(year_match.group(1)) if year_match else None
        for ln in fn.read_text(encoding="utf-8").splitlines():
            rec = json.loads(ln)
            sent = rec.get("sentence")
            doc_id = rec.get("doc_id")
            acts = rec.get("base_activations", {})
            for bid in top_bids:
                val = acts.get(str(bid), acts.get(bid))
                if val is None:
                    continue
                all_acts[bid].append((float(val), sent, year, doc_id))

    for bid, entries in all_acts.items():
        if not entries:
            print(f"[WARN] No matching activations found for base {bid}, skipping")
            continue
        top30 = heapq.nlargest(30, entries, key=lambda x: x[0])
        jsonl_path = txt_dir / f"base_{bid:05d}.jsonl"
        txt_path = txt_dir / f"base_{bid}_top30.txt"

        with open(jsonl_path, "w", encoding="utf-8") as jf, open(txt_path, "w", encoding="utf-8") as tf:
            for rank, (act, sent, year, doc_id) in enumerate(top30, 1):
                record = {
                    "rank": rank,
                    "activation": float(act),
                    "year": year,
                    "doc_id": doc_id,
                    "sentence": sent
                }
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")
                tf.write(f"[{rank:02d}] act={act:.4f}  year={year}  doc={doc_id}  {sent}\n")

        # Generate the word cloud after removing numbers. If the wordcloud
        # dependency is unavailable, fall back to a frequency chart so the
        # pipeline still produces a usable artifact without extra setup.
        text_corpus = "\n".join(filter_text(sent) for _, sent, _, _ in entries)
        img_path = wc_dir / f"base_{bid}_wordcloud.png"
        if WordCloud is not None:
            wc = WordCloud(
                font_path=font_path,
                width=800,
                height=600,
                background_color="white"
            )
            wc.generate(text_corpus)
            wc.to_file(str(img_path))
        else:
            print("[WARN] wordcloud is not installed; generating the fallback frequency chart instead")
            save_frequency_plot(text_corpus, img_path, font_path, bid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Step 05: generate word clouds and top-30 sentences")
    parser.add_argument('--word', required=True, help="Target word")
    parser.add_argument('--act_dir', required=True, help="Activation file directory")
    parser.add_argument('--drift_f', required=True, help="Drift file")
    parser.add_argument('--txt_dir', required=True, help="Output directory for top-30 sentence files")
    parser.add_argument('--wc_dir', required=True, help="Word cloud output directory")
    parser.add_argument('--font_path', required=True, help="Font path")
    args = parser.parse_args()
    generate_wordclouds(
        target_word=args.word,
        act_dir=args.act_dir,
        drift_f=args.drift_f,
        txt_dir=args.txt_dir,
        wc_dir=args.wc_dir,
        font_path=args.font_path
    ) 
