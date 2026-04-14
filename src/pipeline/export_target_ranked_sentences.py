#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export yearly rankings for target-only sentences.

Inputs:
- activations/{word}_activations_{year}.jsonl

Outputs:
- target_sorted_sentences_by_year/year_{year}_all_by_activation_mass.jsonl
- target_sorted_sentences_by_year/year_{year}_all_by_activation_mass.txt
- target_sorted_sentences_by_year/year_{year}_all_by_max_activation.jsonl
- target_sorted_sentences_by_year/year_{year}_all_by_max_activation.txt
- target_sorted_sentences_by_year/manifest.json
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_year(path: Path) -> int | None:
    match = re.search(r"(\d{4})", path.stem)
    return int(match.group(1)) if match else None


def normalize_acts(raw_acts: Dict) -> Dict[int, float]:
    acts: Dict[int, float] = {}
    for key, value in raw_acts.items():
        try:
            acts[int(key)] = float(value)
        except Exception:
            continue
    return acts


def summarize_record(record: Dict, year: int) -> Dict:
    acts = normalize_acts(record.get("base_activations", {}))
    if acts:
        max_base_id, max_activation = max(acts.items(), key=lambda item: item[1])
    else:
        max_base_id, max_activation = None, 0.0
    return {
        "year": int(year),
        "doc_id": record.get("doc_id"),
        "sentence": record.get("sentence", ""),
        "activation_mass": float(sum(acts.values())),
        "max_activation": float(max_activation),
        "max_base_id": int(max_base_id) if max_base_id is not None else None,
        "n_active_bases": len(acts),
        "base_activations": acts,
    }


def iter_year_records(path: Path, year: int) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield summarize_record(json.loads(line), year=year)


def sort_records_by_mass(records: List[Dict]) -> List[Dict]:
    return sorted(
        records,
        key=lambda item: (
            -item["activation_mass"],
            -item["max_activation"],
            -item["n_active_bases"],
            item.get("doc_id") or "",
            item.get("sentence") or "",
        ),
    )


def sort_records_by_max(records: List[Dict]) -> List[Dict]:
    return sorted(
        records,
        key=lambda item: (
            -item["max_activation"],
            -item["activation_mass"],
            -item["n_active_bases"],
            item.get("doc_id") or "",
            item.get("sentence") or "",
        ),
    )


def write_year_outputs(
    out_dir: Path,
    year: int,
    records: List[Dict],
    sort_label: str,
) -> Tuple[Path, Path]:
    jsonl_path = out_dir / f"year_{year}_all_by_{sort_label}.jsonl"
    txt_path = out_dir / f"year_{year}_all_by_{sort_label}.txt"

    with open(jsonl_path, "w", encoding="utf-8") as jf, open(txt_path, "w", encoding="utf-8") as tf:
        for rank, record in enumerate(records, 1):
            payload = {"rank": rank, **record}
            jf.write(json.dumps(payload, ensure_ascii=False) + "\n")
            tf.write(
                f"[{rank:04d}] mass={record['activation_mass']:.4f}"
                f"  max={record['max_activation']:.4f}"
                f"  max_base={record['max_base_id']}"
                f"  n_bases={record['n_active_bases']}"
                f"  doc={record.get('doc_id')}\n"
            )
            tf.write(f"{record.get('sentence', '')}\n\n")

    return jsonl_path, txt_path


def export_target_ranked_sentences(word: str, act_dir: str, out_dir: str) -> None:
    act_dir_path = Path(act_dir)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    activation_files = sorted(act_dir_path.glob(f"{word}_activations_*.jsonl"))
    if not activation_files:
        raise FileNotFoundError(
            f"Target-only activation files not found: {act_dir_path}/{word}_activations_*.jsonl"
        )

    manifest = {
        "word": word,
        "act_dir": str(act_dir_path),
        "out_dir": str(out_dir_path),
        "sort_outputs": [
            "activation_mass_desc",
            "max_activation_desc",
        ],
        "years": [],
    }

    for act_file in activation_files:
        year = parse_year(act_file)
        if year is None:
            continue
        year_records = list(iter_year_records(act_file, year))
        records_by_mass = sort_records_by_mass(year_records)
        records_by_max = sort_records_by_max(year_records)
        mass_jsonl_path, mass_txt_path = write_year_outputs(
            out_dir=out_dir_path,
            year=year,
            records=records_by_mass,
            sort_label="activation_mass",
        )
        max_jsonl_path, max_txt_path = write_year_outputs(
            out_dir=out_dir_path,
            year=year,
            records=records_by_max,
            sort_label="max_activation",
        )
        manifest["years"].append(
            {
                "year": int(year),
                "n_sentences": len(year_records),
                "by_activation_mass": {
                    "jsonl": str(mass_jsonl_path),
                    "txt": str(mass_txt_path),
                    "top_activation_mass": float(records_by_mass[0]["activation_mass"]) if records_by_mass else 0.0,
                    "top_max_activation": float(records_by_mass[0]["max_activation"]) if records_by_mass else 0.0,
                },
                "by_max_activation": {
                    "jsonl": str(max_jsonl_path),
                    "txt": str(max_txt_path),
                    "top_activation_mass": float(records_by_max[0]["activation_mass"]) if records_by_max else 0.0,
                    "top_max_activation": float(records_by_max[0]["max_activation"]) if records_by_max else 0.0,
                },
            }
        )

    manifest_path = out_dir_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export yearly ranked target-only sentences")
    parser.add_argument("--word", required=True, help="Target word")
    parser.add_argument("--act_dir", required=True, help="Activation file directory")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()
    export_target_ranked_sentences(
        word=args.word,
        act_dir=args.act_dir,
        out_dir=args.out_dir,
    )
