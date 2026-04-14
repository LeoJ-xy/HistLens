#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STEP-3 (generic)
Select the Top-N bases by cumulative drift.
"""
import argparse
import json
import numpy as np
from pathlib import Path

def identify_top_drift_bases(
    center_f,
    drift_f,
    top_n=10,
    layer=None,
    source=None,
    concept=None,
    year_start=None,
    year_end=None
):
    center_f = Path(center_f)
    drift_f = Path(drift_f)
    with open(center_f, "r", encoding="utf-8") as f:
        data = json.load(f)
    base_order = data["base_order"]
    centers    = {int(y): np.array(v) for y, v in data["centers"].items()}
    years = sorted(centers.keys())
    B     = len(base_order)
    cum   = np.zeros(B, dtype=float)
    peak  = [(0.0, -1, -1) for _ in range(B)]
    for y1, y2 in zip(years[:-1], years[1:]):
        delta = np.abs(centers[y2] - centers[y1])
        cum  += delta
        for i, v in enumerate(delta):
            if v > peak[i][0]:
                peak[i] = (float(v), y1, y2)
    top_idx = cum.argsort()[-top_n:][::-1]

    top_bases = []
    for i in top_idx:
        bid, drift = base_order[i], float(cum[i])
        pval, y1, y2 = peak[i]
        top_bases.append({
            "base_id": int(bid),
            "cum_drift": drift,
            "peak_delta": pval,
            "years": [y1, y2]
        })

    out = {
        "layer": int(layer) if layer is not None else None,
        "source": source,
        "concept": concept,
        "time_range": {"start": int(year_start) if year_start else None, "end": int(year_end) if year_end else None},
        "top_n": int(top_n),
        "top_bases": top_bases
    }
    with open(drift_f, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Step 03: select Top-N cumulative drift bases")
    parser.add_argument('--center_f', required=True, help="Center file")
    parser.add_argument('--drift_f', required=True, help="Output drift file")
    parser.add_argument('--top_n', type=int, default=10, help="Top-N")
    parser.add_argument('--layer', type=int, default=None, help="SAE layer")
    parser.add_argument('--source', type=str, default=None, help="Corpus source")
    parser.add_argument('--concept', type=str, default=None, help="Target concept")
    parser.add_argument('--year_start', type=int, default=None, help="Start year")
    parser.add_argument('--year_end', type=int, default=None, help="End year")
    args = parser.parse_args()
    identify_top_drift_bases(
        center_f=args.center_f,
        drift_f=args.drift_f,
        top_n=args.top_n,
        layer=args.layer,
        source=args.source,
        concept=args.concept,
        year_start=args.year_start,
        year_end=args.year_end
    )
