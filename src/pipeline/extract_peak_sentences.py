#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STEP-4 (generic)
Extract sentences for the peak-change year pairs.
"""
import argparse
import json
import heapq
from pathlib import Path

def extract_key_base_peak_sentences(target_word, act_dir, drift_f, out_json, txt_dir, top_sent=5):
    act_dir = Path(act_dir)
    drift_f = Path(drift_f)
    out_json = Path(out_json)
    txt_dir = Path(txt_dir)
    txt_dir.mkdir(parents=True, exist_ok=True)
    drift_data = json.load(open(drift_f, "r", encoding="utf-8"))

    def normalize_top_bases(obj):
        if isinstance(obj, dict) and "top_bases" in obj:
            return obj["top_bases"]
        if isinstance(obj, dict):
            return [
                {"base_id": int(bid), **info}
                for bid, info in obj.items()
            ]
        return []

    top_bases = normalize_top_bases(drift_data)
    heap_map = {}
    for item in top_bases:
        bid = int(item["base_id"])
        y1, y2 = item["years"]
        heap_map[bid] = {y1: [], y2: []}
    def mark(s: str) -> str:
        return s.replace(target_word, f"【{target_word}】")
    for act_file in act_dir.glob(f"{target_word}_activations_*.jsonl"):
        yr = int(act_file.stem.split("_")[-1])
        needed_bids = [
            int(item["base_id"])
            for item in top_bases
            if yr in item["years"]
        ]
        if not needed_bids:
            continue
        for ln in act_file.read_text(encoding="utf-8").splitlines():
            obj  = json.loads(ln)
            acts = obj["base_activations"]
            sent = mark(obj["sentence"])
            doc  = obj["doc_id"]
            for bid in needed_bids:
                if bid not in heap_map or yr not in heap_map[bid]:
                    continue
                act_dict = acts
                if str(bid) not in act_dict:
                    continue
                val = act_dict[str(bid)]
                heap = heap_map[bid][yr]
                heapq.heappush(heap, (val, doc, sent))
                if len(heap) > top_sent:
                    heapq.heappop(heap)
    result = {}
    for item in top_bases:
        bid   = int(item["base_id"])
        delta = item["peak_delta"]
        y1, y2 = item["years"]
        per_year = {}
        for y in (y1, y2):
            heap = heap_map.get(bid, {}).get(y, [])
            best = sorted(heap, key=lambda x: -x[0])
            per_year[str(y)] = [
                {"activation": float(v), "doc_id": d, "sentence": s}
                for v, d, s in best
            ]
        result[str(bid)] = {
            "years": [y1, y2],
            "peak_delta": delta,
            "sentences": per_year
        }
        txt_path = txt_dir / f"key_base_{bid}_y{y1}_{y2}.txt"
        with open(txt_path, "w", encoding="utf-8") as fout:
            fout.write(f"# Base {bid}  peak Δ={delta:.4f}  years {y1}->{y2}\n\n")
            for y in (y1, y2):
                fout.write(f"=== Year {y} ===\n")
                for rank, item in enumerate(result[str(bid)]["sentences"][str(y)], 1):
                    fout.write(f"[{rank:02d}] act={item['activation']:.4f}  {item['sentence']}\n")
    with open(out_json, "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Step 04: extract peak-year sentence pairs")
    parser.add_argument('--word', required=True, help="Target word")
    parser.add_argument('--act_dir', required=True, help="Activation file directory")
    parser.add_argument('--drift_f', required=True, help="Drift file")
    parser.add_argument('--out_json', required=True, help="Output JSON file")
    parser.add_argument('--txt_dir', required=True, help="Output TXT directory")
    parser.add_argument('--top_sent', type=int, default=5, help="Top sentences to keep per year")
    args = parser.parse_args()
    extract_key_base_peak_sentences(
        target_word=args.word,
        act_dir=args.act_dir,
        drift_f=args.drift_f,
        out_json=args.out_json,
        txt_dir=args.txt_dir,
        top_sent=args.top_sent
    ) 
