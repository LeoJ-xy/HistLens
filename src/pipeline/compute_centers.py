#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STEP-2 (generic)
Compute yearly activation centers and distances.
"""
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cosine as cos_dist

def compute_centers_and_distances(target_word, act_dir, out_center, out_dist):
    # Resolve to absolute paths up front.
    act_dir = Path(act_dir).resolve()
    out_center = Path(out_center).resolve()
    out_dist = Path(out_dist).resolve()
    
    def load_activations():
        year_map = defaultdict(list)
        files = sorted(act_dir.glob(f"{target_word}_activations_*.jsonl"))
        
        total_lines = 0
        total_valid = 0
        
        for f in sorted(files):
            try:
                # Accept any file whose name ends with a 4-digit year.
                m = re.search(r"(\d{4})\.jsonl$", f.name)
                if not m:
                    print(f"[WARN] Could not extract a year from filename, skipping: {f.name}")
                    continue
                year = int(m.group(1))
                
                lines = f.read_text(encoding="utf-8").splitlines()
                total_lines += len(lines)
                
                year_data = []
                for i, line in enumerate(lines, 1):
                    try:
                        obj = json.loads(line)
                        base_activations = obj.get("base_activations", {})
                        if base_activations:  # Accept any non-empty activation dict.
                            year_data.append(base_activations)
                            total_valid += 1
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] JSON parse failed on line {i}: {e}")
                    except Exception as e:
                        print(f"[ERROR] Failed to process line {i}: {str(e)}")

                if year_data:  # Only keep years with at least one valid record.
                    year_map[year] = year_data
            except Exception as e:
                print(f"[ERROR] Failed to process file {f.name}: {str(e)}")
                
        return year_map
    
    acts_by_year = load_activations()
    if not acts_by_year:
        print("[ERROR] No activation data was loaded.")
        return
        
    all_bases = {int(bid) for lst in acts_by_year.values() for rec in lst for bid in rec.keys()}
    base_order = sorted(all_bases)
    base_to_idx = {bid: idx for idx, bid in enumerate(base_order)}
    B = len(base_order)
    
    centers = {}
    for year, recs in acts_by_year.items():
        M = np.zeros((len(recs), B), dtype=float)
        for i, rec in enumerate(recs):
            for bid_str, act in rec.items():
                bid = int(bid_str)
                M[i, base_to_idx[bid]] = act
        centers[year] = M.mean(axis=0).tolist()
    
    with open(out_center, "w", encoding="utf-8") as f:
        json.dump({"base_order": base_order, "centers": centers}, f, ensure_ascii=False, indent=2)
    
    years = sorted(centers.keys())
    distances = {}
    for y1, y2 in zip(years[:-1], years[1:]):
        v1 = np.array(centers[y1])
        v2 = np.array(centers[y2])
        distances[f"{y1}->{y2}"] = {
            "euclidean": float(np.linalg.norm(v2 - v1)),
            "cosine":    float(cos_dist(v1, v2))
        }
    
    with open(out_dist, "w", encoding="utf-8") as f:
        json.dump(distances, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Step 02: compute yearly centers and distances")
    parser.add_argument('--word', required=True, help="Target word")
    parser.add_argument('--act_dir', required=True, help="Activation file directory")
    parser.add_argument('--out_center', required=True, help="Output center file")
    parser.add_argument('--out_dist', required=True, help="Output distance file")
    args = parser.parse_args()
    compute_centers_and_distances(
        target_word=args.word,
        act_dir=args.act_dir,
        out_center=args.out_center,
        out_dist=args.out_dist
    ) 
