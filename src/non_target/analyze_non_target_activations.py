#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze activations on top drift bases for sentences that do not contain the target word.

Main tasks:
1. Extract activations for all sentences, including non-target ones
2. Select sentences that do not contain the target word
3. Compute activations on the top drift bases for those sentences
4. Find the top 30 sentences per base with year and document metadata
5. Compute yearly means for three populations:
   - all records
   - records containing the target word
   - records without the target word
"""
import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
import torch
import transformers
from opensae.config_utils import PretrainedSaeConfig

# OpenSAE's opensae/__init__.py calls setuptools_scm.get_version() during import,
# usually relative to the current working tree. In some launch modes, such as
# nohup or a non-git directory, that can raise LookupError, so we provide a
# fallback version to keep the process running.
os.environ.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "0.0.0")
from opensae.transformer_with_sae import TransformerWithSae
from opensae import OpenSae
from opensae.saes.open_sae.configuration_open_sae import OpenSaeConfig


def patch_opensae_dtype_bug() -> None:
    def _get_torch_dtype(self) -> torch.dtype:
        if isinstance(self.torch_dtype, torch.dtype):
            return self.torch_dtype
        dtype = getattr(torch, self.torch_dtype)
        assert isinstance(dtype, torch.dtype)
        return dtype

    PretrainedSaeConfig.get_torch_dtype = _get_torch_dtype


def load_sae_checkpoint(sae_ckpt: str) -> OpenSae:
    # Work around the current OpenSAE loader bug where get_torch_dtype() fails
    # when torch_dtype is already a torch.dtype object.
    patch_opensae_dtype_bug()
    cfg = OpenSaeConfig.from_pretrained(sae_ckpt)
    cfg.torch_dtype = "float32"
    return OpenSae.from_pretrained(sae_ckpt, config=cfg)


def encode_sentence(
    tokenizer: transformers.PreTrainedTokenizerBase,
    sentence: str,
    *,
    max_tokens: int,
) -> tuple[transformers.BatchEncoding, Optional[List[Tuple[int, int]]]]:
    try:
        enc = tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_offsets_mapping=True,
        )
        offsets = [tuple(map(int, pair)) for pair in enc.pop("offset_mapping")[0].tolist()]
        return enc, offsets
    except Exception:
        enc = tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
        )
        return enc, None


def collect_sparse_rows(
    model: TransformerWithSae,
    enc: transformers.BatchEncoding,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    enc = enc.to(model.device)
    with torch.inference_mode():
        model.forward(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    if model.encoder_output is None:
        return None, None, 0

    seq_len = int(enc["attention_mask"][0].sum().item())
    all_indices = model.encoder_output.sparse_feature_indices
    all_acts = model.encoder_output.sparse_feature_activations
    available = min(seq_len, int(all_indices.shape[0]), int(all_acts.shape[0]))
    return all_indices[:available], all_acts[:available], available


def aggregate_sparse_features(
    sparse_indices: torch.Tensor,
    sparse_acts: torch.Tensor,
    token_positions: Sequence[int],
) -> dict[int, float]:
    agg: dict[int, float] = {}
    for pos in token_positions:
        for bid, val in zip(sparse_indices[pos].tolist(), sparse_acts[pos].tolist()):
            val = float(val)
            if val <= 1e-6:
                continue
            bid = int(bid)
            if bid not in agg or val > agg[bid]:
                agg[bid] = val
    return agg


def _split_csv(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def load_aliases(raw: Optional[str], path: Optional[str]) -> List[str]:
    aliases: List[str] = []
    aliases.extend(_split_csv(raw))
    if path:
        p = Path(path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                cleaned = line.strip()
                if cleaned and not cleaned.startswith("#"):
                    aliases.append(cleaned)
    seen = set()
    deduped: List[str] = []
    for alias in aliases:
        if alias in seen:
            continue
        seen.add(alias)
        deduped.append(alias)
    return deduped


def normalize_top_bases(obj) -> List[dict]:
    if isinstance(obj, dict) and "top_bases" in obj:
        return obj["top_bases"]
    if isinstance(obj, dict):
        normalized = []
        for bid, info in obj.items():
            if str(bid).isdigit():
                normalized.append({"base_id": int(bid), **info})
        return normalized
    return []


def write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def quantile_tag(value: float) -> str:
    return f"q{int(round(float(value) * 1000)):03d}"


def reset_generated_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize_sentence_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def make_text_matcher(
    target_word: str,
    aliases: List[str],
    *,
    match_mode: str,
    target_regex: Optional[str],
) -> Callable[[str], bool]:
    terms = [target_word, *aliases]
    terms = [t for t in terms if t]

    if match_mode == "regex":
        if not target_regex:
            raise ValueError("--target_regex is required when match_mode=regex")
        pattern = re.compile(target_regex)

        def _match(text: str) -> bool:
            return bool(pattern.search(text))

        return _match

    if match_mode != "substring":
        raise ValueError(f"Unknown match_mode: {match_mode}")

    def _match(text: str) -> bool:
        return any(term in text for term in terms)

    return _match


def _count_chars(text: str):
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_letters = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    digits = sum(1 for ch in text if ch.isdigit())
    non_space = sum(1 for ch in text if not ch.isspace())
    return cjk, ascii_letters, digits, non_space


def is_good_evidence_text(
    text: str,
    *,
    min_chars: int,
    min_cjk_chars: int,
    max_digit_ratio: float,
    max_punct_ratio: float,
    max_ascii_ratio: float,
) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    cjk, ascii_letters, digits, non_space = _count_chars(cleaned)
    if non_space < min_chars:
        return False
    # Purely English evidence is allowed, but strings with almost no CJK text and
    # mostly letters, digits, or symbols are often directory names, page numbers,
    # or catalog-like noise.
    if cjk < min_cjk_chars and (ascii_letters + digits) / max(non_space, 1) > max_ascii_ratio:
        return False
    # Heavy numeric or punctuation density is usually numbering, dates, or citations.
    if digits / max(non_space, 1) > max_digit_ratio:
        return False
    punct = sum(1 for ch in cleaned if not ch.isalnum() and not ("\u4e00" <= ch <= "\u9fff") and not ch.isspace())
    if punct / max(non_space, 1) > max_punct_ratio:
        return False
    return True


def extract_all_activations(
    target_word,
    years,
    sent_dir,
    meta_path,
    out_dir,
    sae_ckpt,
    llama_path,
    device,
    max_tokens: int = 4096,
    oom_recover: bool = True,
    log_truncation: bool = True,
    matcher: Optional[Callable[[str], bool]] = None,
    min_context_tokens: int = 20,
    context_mode: str = "prepend",
    shard_index: int = 0,
    num_shards: int = 1,
):
    """Extract activations for all sentences, including non-target ones."""
    sent_dir = Path(sent_dir)
    meta_path = Path(meta_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sae = load_sae_checkpoint(sae_ckpt)
    model = TransformerWithSae(llama_path, sae, device=device)
    # Reduce memory pressure: disable the KV cache because we only need forward features.
    try:
        if hasattr(model, "transformer") and hasattr(model.transformer, "config"):
            model.transformer.config.use_cache = False
    except Exception:
        pass
    tok = transformers.AutoTokenizer.from_pretrained(llama_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Metadata: filename -> year
    fname2year = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for ln in f:
            obj = json.loads(ln)
            fname2year[obj["filename"]] = int(obj["pub_date"][:4])

    def extend_line(lines, idx):
        sent = lines[idx].strip()
        ids = tok(sent)["input_ids"]
        j = idx - 1
        while len(ids) < min_context_tokens and j >= 0:
            prev = lines[j].strip()
            if not prev:
                break
            sent = prev + " " + sent
            ids = tok(sent)["input_ids"]
            j -= 1
        return sent

    def extract_acts(sentence: str, ctx: str = "") -> tuple[dict, str]:
        """Extract all base activations for a sentence, regardless of target-word presence."""
        full_len = None
        if log_truncation:
            try:
                full_len = len(tok(sentence)["input_ids"])
            except Exception:
                full_len = None

        enc, offsets = encode_sentence(tok, sentence, max_tokens=max_tokens)
        if log_truncation and full_len is not None and full_len > max_tokens:
            print(f"[WARNING] Input was truncated: orig_tokens={full_len}, max_tokens={max_tokens} | {ctx}")
        # Use the truncated text for contains_target checks so labeling stays
        # consistent with the actual model input.
        text_used = tok.decode(enc["input_ids"][0], skip_special_tokens=True)

        try:
            sparse_indices, sparse_acts, seq_len = collect_sparse_rows(model, enc)
        except torch.OutOfMemoryError:
            if oom_recover:
                # Log and skip this sentence so one OOM does not abort the run.
                seqlen = int(enc["input_ids"].shape[-1]) if "input_ids" in enc else -1
                print(f"[ERROR] OOM, skipping sentence: seqlen={seqlen}, max_tokens={max_tokens} | {ctx}")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return {}, text_used
            raise

        if sparse_indices is None or sparse_acts is None or seq_len == 0:
            return {}, text_used
        if offsets is not None:
            token_positions = [
                i for i, (start, end) in enumerate(offsets[:seq_len]) if start != end
            ]
        else:
            token_positions = list(range(seq_len))
        return aggregate_sparse_features(sparse_indices, sparse_acts, token_positions), text_used

    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got: {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index out of range: shard_index={shard_index}, num_shards={num_shards}"
        )
    
    # Collect candidate files first, then shard by file.
    candidate_files = []
    for txt in sorted(sent_dir.glob("*.txt")):
        yr = fname2year.get(txt.name)
        if yr is None and txt.name[:4].isdigit():
            yr = int(txt.name[:4])
        if yr is None or (years and yr not in years):
            continue
        candidate_files.append((txt, yr))

    # File-level sharding.
    if num_shards > 1:
        txt_files = [
            item
            for idx, item in enumerate(candidate_files)
            if (idx % num_shards) == shard_index
        ]
    else:
        txt_files = candidate_files

    total_files = len(txt_files)

    # Count lines for progress reporting.
    total_lines_count = 0
    for txt, _yr in txt_files:
        with open(txt, "r", encoding="utf-8") as f:
            total_lines_count += sum(1 for line in f if line.strip())
    
    total = kept_all = kept_with = kept_without = 0
    start_time = time.time()
    last_progress_time = start_time
    last_progress_count = 0
    
    # Open output files lazily per year to avoid holding everything in memory.
    year_handles = {}

    def get_year_handle(year: int):
        handle = year_handles.get(year)
        if handle is None:
            out_f = out_dir / f"all_activations_{year}.jsonl"
            out_f.parent.mkdir(parents=True, exist_ok=True)
            handle = open(out_f, "a", encoding="utf-8")
            year_handles[year] = handle
        return handle

    try:
        for txt, yr in txt_files:
            lines = txt.read_text(encoding="utf-8").splitlines()

            for idx, line in enumerate(lines):
                if not line.strip():
                    continue
                total += 1

                # Estimate throughput and remaining time.
                current_time = time.time()
                recent_elapsed = current_time - last_progress_time

                # Periodically refresh counters so long-running extraction does not
                # accumulate stale timing state even when running quietly.
                if total % 100 == 0 or recent_elapsed >= 5.0:
                    if recent_elapsed > 0:
                        last_progress_time = current_time
                        last_progress_count = total

                sent = lines[idx].strip() if context_mode == "none" else extend_line(lines, idx)
                ctx = f"file={txt.name} year={yr} line_idx={idx}"
                acts, text_used = extract_acts(sent, ctx=ctx)
                if not acts:
                    continue

                contains_target = matcher(text_used) if matcher is not None else (target_word in text_used)

                kept_all += 1
                if contains_target:
                    kept_with += 1
                else:
                    kept_without += 1

                record = {
                    "doc_id": txt.stem,
                    "sentence": sent,
                    "base_activations": acts,
                    "contains_target": contains_target,
                }
                handle = get_year_handle(yr)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        for handle in year_handles.values():
            try:
                handle.close()
            except Exception:
                pass
    
    return None


def analyze_top_bases_activations(
    target_word,
    top_bases_file,
    all_activations_dir,
    output_dir,
    top_n_sentences=30,
    matcher: Optional[Callable[[str], bool]] = None,
    tokenizer=None,
    max_tokens: int = 4096,
    recompute_contains_target: bool = False,
    evidence_filter: bool = True,
    evidence_min_chars: int = 12,
    evidence_min_cjk_chars: int = 3,
    evidence_max_digit_ratio: float = 0.35,
    evidence_max_punct_ratio: float = 0.5,
    evidence_max_ascii_ratio: float = 0.85,
    evidence_dedupe: bool = True,
    evidence_max_per_doc: int = 2,
    high_activation_quantile: float = 0.95,
):
    """Analyze activations on top bases for sentences without the target word."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the selected top bases.
    with open(top_bases_file, "r", encoding="utf-8") as f:
        top_bases_data = json.load(f)

    top_bases = normalize_top_bases(top_bases_data)
    top_base_ids = [int(item["base_id"]) for item in top_bases]
    base_meta = {int(item["base_id"]): item for item in top_bases}
    if not top_base_ids:
        raise ValueError(f"No base_id values could be parsed from top_bases_file: {top_bases_file}")

    ranked_dir = output_dir / "without_target_ranked_sentences"
    high_dir = output_dir / "without_target_high_activation_sentences"
    high_by_year_dir = output_dir / "without_target_high_activation_sentences_by_year"
    coactivation_dir = output_dir / "without_target_high_activation_coactivation"
    ranked_dir.mkdir(parents=True, exist_ok=True)
    reset_generated_dir(high_dir)
    reset_generated_dir(high_by_year_dir)
    reset_generated_dir(coactivation_dir)
    combined_high_records = []
    high_records_by_year = defaultdict(list)
    sentence_coactivation = {}
    high_activation_manifest = {
        "target_word": target_word,
        "selection": {
            "population": "without_target & base_present",
            "high_activation_quantile": float(high_activation_quantile),
            "high_activation_threshold_mode": "per_base_per_year",
            "top_sentences_population": (
                "without_target & base_present & evidence_filter_applied_if_enabled"
            ),
        },
        "bases": {},
    }
    
    # Load all activation files.
    all_activations_dir = Path(all_activations_dir)
    activation_files = sorted(all_activations_dir.glob("all_activations_*.jsonl"))
    
    # Organize records by year.
    year_data_all = defaultdict(list)  # All records.
    year_data_with_target = defaultdict(list)  # Records containing the target.
    year_data_without_target = defaultdict(list)  # Records without the target.
    
    for act_file in activation_files:
        # Extract the year from the filename.
        m = re.search(r"(\d{4})\.jsonl$", act_file.name)
        if not m:
            continue
        year = int(m.group(1))
        
        for line in act_file.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                text = rec.get("sentence", "") or ""
                contains = bool(rec.get("contains_target", False))
                if recompute_contains_target:
                    if matcher is None:
                        raise ValueError("recompute_contains_target=True but matcher=None")
                    text_for_match = text
                    if tokenizer is not None:
                        try:
                            enc = tokenizer(
                                text_for_match,
                                return_tensors=None,
                                truncation=True,
                                max_length=max_tokens,
                            )
                            ids = enc.get("input_ids")
                            if isinstance(ids, list) and ids:
                                text_for_match = tokenizer.decode(ids, skip_special_tokens=True)
                        except Exception:
                            # Fall back to raw text.
                            pass
                    contains = bool(matcher(text_for_match))
                rec["contains_target"] = contains

                year_data_all[year].append(rec)
                if contains:
                    year_data_with_target[year].append(rec)
                else:
                    year_data_without_target[year].append(rec)
            except:
                continue
    
    # Analyze each base independently.
    results = {}
    annual_stats = {}  # {year: {base_id: stats}}
    
    # Initialize the annual stats container.
    all_years = sorted(set(year_data_all.keys()))
    for year in all_years:
        annual_stats[year] = {}
    
    total_bases = len(top_base_ids)
    analysis_start_time = time.time()
    
    for base_idx, base_id in enumerate(top_base_ids, 1):
        base_start_time = time.time()
        
        # Collect activations for non-target sentences.
        sentences_without_target_all = []
        sentences_without_target_evidence = []
        for year, records in year_data_without_target.items():
            for rec in records:
                acts = rec.get("base_activations", {})
                if str(base_id) in acts:
                    activation = float(acts[str(base_id)])
                    sentence_text = rec.get("sentence", "")
                    item = {
                        "year": year,
                        "doc_id": rec["doc_id"],
                        "sentence": sentence_text,
                        "activation": activation
                    }
                    sentences_without_target_all.append(item)
                    if (not evidence_filter) or is_good_evidence_text(
                        sentence_text,
                        min_chars=evidence_min_chars,
                        min_cjk_chars=evidence_min_cjk_chars,
                        max_digit_ratio=evidence_max_digit_ratio,
                        max_punct_ratio=evidence_max_punct_ratio,
                        max_ascii_ratio=evidence_max_ascii_ratio,
                    ):
                        sentences_without_target_evidence.append(item)
        
        # Sort by activation and keep the strongest sentences first.
        sentences_without_target_all.sort(key=lambda x: x["activation"], reverse=True)
        sentences_without_target_evidence.sort(key=lambda x: x["activation"], reverse=True)
        ranked_records = []
        for rank, item in enumerate(sentences_without_target_all, 1):
            ranked_records.append({
                "base_id": base_id,
                "rank": rank,
                "year": item["year"],
                "doc_id": item["doc_id"],
                "sentence": item["sentence"],
                "activation": item["activation"],
            })
        ranked_path = ranked_dir / f"base_{base_id:05d}.jsonl"
        write_jsonl(ranked_path, ranked_records)

        high_sentences = []
        high_thresholds_by_year = {}
        high_counts_by_year = {}
        sentences_without_target_all_by_year = defaultdict(list)
        for item in sentences_without_target_all:
            sentences_without_target_all_by_year[int(item["year"])].append(item)
        for year, year_items in sorted(sentences_without_target_all_by_year.items()):
            activations = [float(item["activation"]) for item in year_items]
            threshold = (
                float(np.quantile(np.asarray(activations, dtype=float), high_activation_quantile))
                if activations
                else None
            )
            high_thresholds_by_year[str(year)] = threshold
            if threshold is None:
                high_counts_by_year[str(year)] = 0
                continue
            year_high = [
                item for item in year_items
                if float(item["activation"]) >= threshold
            ]
            year_high.sort(key=lambda x: float(x["activation"]), reverse=True)
            high_counts_by_year[str(year)] = len(year_high)
            for item in year_high:
                enriched = dict(item)
                enriched["high_activation_threshold_year"] = threshold
                high_sentences.append(enriched)
        high_sentences.sort(key=lambda x: float(x["activation"]), reverse=True)
        high_records = []
        for rank, item in enumerate(high_sentences, 1):
            rec = {
                "base_id": base_id,
                "rank": rank,
                "year": item["year"],
                "doc_id": item["doc_id"],
                "sentence": item["sentence"],
                "activation": item["activation"],
                "high_activation_quantile": float(high_activation_quantile),
                "high_activation_threshold_mode": "per_base_per_year",
                "high_activation_threshold_year": float(item["high_activation_threshold_year"]),
            }
            high_records.append(rec)
            combined_high_records.append(rec)
            high_records_by_year[int(item["year"])].append(rec)
            sent_key = (
                int(item["year"]),
                str(item["doc_id"]),
                normalize_sentence_key(item["sentence"]),
            )
            bucket = sentence_coactivation.setdefault(
                sent_key,
                {
                    "year": int(item["year"]),
                    "doc_id": str(item["doc_id"]),
                    "sentence": item["sentence"],
                    "bases": [],
                    "base_activations": {},
                },
            )
            bucket["bases"].append(int(base_id))
            bucket["base_activations"][str(base_id)] = float(item["activation"])
        high_path = high_dir / f"base_{base_id:05d}_{quantile_tag(high_activation_quantile)}.jsonl"
        write_jsonl(high_path, high_records)

        if evidence_dedupe:
            seen_sent = set()
            per_doc = defaultdict(int)
            deduped = []
            source_for_top = (
                sentences_without_target_evidence
                if evidence_filter
                else sentences_without_target_all
            )
            for item in source_for_top:
                doc_id = item.get("doc_id", "")
                if evidence_max_per_doc > 0 and per_doc[doc_id] >= evidence_max_per_doc:
                    continue
                key = re.sub(r"\s+", " ", str(item.get("sentence", "")).strip())
                if not key or key in seen_sent:
                    continue
                seen_sent.add(key)
                per_doc[doc_id] += 1
                deduped.append(item)
                if len(deduped) >= top_n_sentences:
                    break
            top_sentences = deduped
        else:
            source_for_top = (
                sentences_without_target_evidence
                if evidence_filter
                else sentences_without_target_all
            )
            top_sentences = source_for_top[:top_n_sentences]
        
        results[str(base_id)] = {
            "base_id": base_id,
            "top_sentences": top_sentences,
            "total_sentences_without_target": len(sentences_without_target_all),
            "total_evidence_sentences_without_target": len(sentences_without_target_evidence),
            "ranked_sentences_file": str(ranked_path.name),
            "high_activation_sentences_file": str(high_path.name),
            "high_activation_quantile": float(high_activation_quantile),
            "high_activation_threshold_mode": "per_base_per_year",
            "high_activation_threshold_by_year": high_thresholds_by_year,
            "high_activation_sentence_count": len(high_sentences),
            "high_activation_sentence_count_by_year": high_counts_by_year,
        }
        high_activation_manifest["bases"][str(base_id)] = {
            "ranked_sentences_file": ranked_path.name,
            "high_activation_sentences_file": high_path.name,
            "high_activation_quantile": float(high_activation_quantile),
            "high_activation_threshold_mode": "per_base_per_year",
            "high_activation_threshold_by_year": high_thresholds_by_year,
            "total_without_target_sentences": len(sentences_without_target_all),
            "total_evidence_sentences_without_target": len(sentences_without_target_evidence),
            "high_activation_sentence_count": len(high_sentences),
            "high_activation_sentence_count_by_year": high_counts_by_year,
        }
        
        # Compute yearly means for each population.
        for year in all_years:
            all_total = len(year_data_all[year])
            with_total = len(year_data_with_target[year])
            without_total = len(year_data_without_target[year])

            # All records.
            all_acts = []
            all_sum = 0.0
            for rec in year_data_all[year]:
                acts = rec.get("base_activations", {})
                all_sum += float(acts.get(str(base_id), 0.0) or 0.0)
                if str(base_id) in acts:
                    all_acts.append(float(acts[str(base_id)]))
            
            # Records containing the target word.
            with_acts = []
            with_sum = 0.0
            for rec in year_data_with_target[year]:
                acts = rec.get("base_activations", {})
                with_sum += float(acts.get(str(base_id), 0.0) or 0.0)
                if str(base_id) in acts:
                    with_acts.append(float(acts[str(base_id)]))
            
            # Records without the target word.
            without_acts = []
            without_sum = 0.0
            for rec in year_data_without_target[year]:
                acts = rec.get("base_activations", {})
                without_sum += float(acts.get(str(base_id), 0.0) or 0.0)
                if str(base_id) in acts:
                    without_acts.append(float(acts[str(base_id)]))
            
            annual_stats[year][str(base_id)] = {
                "all": {
                    "mean": float(np.mean(all_acts)) if all_acts else 0.0,
                    "count": len(all_acts),
                    "total": all_total,
                    "mean_including_zeros": (all_sum / all_total) if all_total > 0 else 0.0,
                },
                "with_target": {
                    "mean": float(np.mean(with_acts)) if with_acts else 0.0,
                    "count": len(with_acts),
                    "total": with_total,
                    "mean_including_zeros": (with_sum / with_total) if with_total > 0 else 0.0,
                },
                "without_target": {
                    "mean": float(np.mean(without_acts)) if without_acts else 0.0,
                    "count": len(without_acts),
                    "total": without_total,
                    "mean_including_zeros": (without_sum / without_total) if without_total > 0 else 0.0,
                }
            }
        
    # Save the structured analysis output.
    results_file = output_dir / f"{target_word}_non_target_analysis.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "target_word": target_word,
            "top_bases": top_base_ids,
            "results": results,
            "annual_statistics": annual_stats
        }, f, ensure_ascii=False, indent=2)
    
    combined_high_records.sort(
        key=lambda x: (float(x["activation"]), int(x["base_id"])),
        reverse=True,
    )
    combined_high_path = high_dir / f"{target_word}_all_top_bases_{quantile_tag(high_activation_quantile)}.jsonl"
    write_jsonl(combined_high_path, combined_high_records)
    year_combined_files = {}
    base_year_files = {}
    combined_high_records_by_base = defaultdict(list)
    for rec in combined_high_records:
        combined_high_records_by_base[int(rec["base_id"])].append(rec)
        year_combined_files.setdefault(str(rec["year"]), [])
    for base_id, base_records in sorted(combined_high_records_by_base.items()):
        base_dir = high_by_year_dir / f"base_{int(base_id):05d}"
        base_dir.mkdir(parents=True, exist_ok=True)
        year_file_map = {}
        by_year = defaultdict(list)
        for rec in base_records:
            by_year[int(rec["year"])].append(rec)
        for year, records in sorted(by_year.items()):
            records.sort(key=lambda x: float(x["activation"]), reverse=True)
            year_path = base_dir / f"year_{year}_{quantile_tag(high_activation_quantile)}.jsonl"
            write_jsonl(year_path, records)
            rel_path = str(year_path.relative_to(high_by_year_dir))
            year_file_map[str(year)] = rel_path
            year_combined_files[str(year)].append(rel_path)
        base_year_files[str(base_id)] = year_file_map
    year_base_files = defaultdict(dict)
    for base_id, year_file_map in base_year_files.items():
        for year, rel_path in year_file_map.items():
            year_base_files[year][base_id] = rel_path
    for year, rel_paths in list(year_combined_files.items()):
        year_combined_files[year] = sorted(rel_paths)

    coactivation_records = []
    pair_counts = defaultdict(int)
    pair_year_counts = defaultdict(lambda: defaultdict(int))
    coactivation_records_by_year = defaultdict(list)
    for item in sentence_coactivation.values():
        uniq_bases = sorted(set(int(bid) for bid in item["bases"]))
        if len(uniq_bases) < 2:
            continue
        activations = {str(bid): float(item["base_activations"][str(bid)]) for bid in uniq_bases}
        rec = {
            "year": int(item["year"]),
            "doc_id": item["doc_id"],
            "sentence": item["sentence"],
            "base_ids": uniq_bases,
            "num_coactivated_bases": len(uniq_bases),
            "base_activations": activations,
            "sum_activation": float(sum(activations.values())),
            "max_activation": float(max(activations.values())),
            "high_activation_quantile": float(high_activation_quantile),
        }
        coactivation_records.append(rec)
        coactivation_records_by_year[int(item["year"])].append(rec)
        for base_a, base_b in combinations(uniq_bases, 2):
            pair_key = (int(base_a), int(base_b))
            pair_counts[pair_key] += 1
            pair_year_counts[pair_key][int(item["year"])] += 1

    coactivation_records.sort(
        key=lambda x: (
            int(x["num_coactivated_bases"]),
            float(x["sum_activation"]),
            float(x["max_activation"]),
            -int(x["year"]),
        ),
        reverse=True,
    )
    coactivation_path = coactivation_dir / f"{target_word}_multi_base_sentences_{quantile_tag(high_activation_quantile)}.jsonl"
    write_jsonl(coactivation_path, coactivation_records)
    year_coactivation_files = {}
    for year, records in sorted(coactivation_records_by_year.items()):
        records.sort(
            key=lambda x: (
                int(x["num_coactivated_bases"]),
                float(x["sum_activation"]),
                float(x["max_activation"]),
            ),
            reverse=True,
        )
        year_path = coactivation_dir / f"year_{year}_multi_base_sentences_{quantile_tag(high_activation_quantile)}.jsonl"
        write_jsonl(year_path, records)
        year_coactivation_files[str(year)] = year_path.name

    pair_records = []
    for (base_a, base_b), count in sorted(
        pair_counts.items(),
        key=lambda kv: (kv[1], kv[0][0], kv[0][1]),
        reverse=True,
    ):
        year_counts = {
            str(year): int(cnt)
            for year, cnt in sorted(pair_year_counts[(base_a, base_b)].items())
        }
        pair_records.append(
            {
                "base_id_1": int(base_a),
                "base_id_2": int(base_b),
                "coactivation_sentence_count": int(count),
                "year_counts": year_counts,
                "high_activation_quantile": float(high_activation_quantile),
            }
        )
    pair_counts_path = coactivation_dir / f"{target_word}_base_pair_counts_{quantile_tag(high_activation_quantile)}.jsonl"
    write_jsonl(pair_counts_path, pair_records)

    high_activation_manifest["combined_high_activation_file"] = combined_high_path.name
    high_activation_manifest["combined_high_activation_by_year_files"] = year_combined_files
    high_activation_manifest["high_activation_by_year_and_base_files"] = dict(year_base_files)
    high_activation_manifest["high_activation_by_base_and_year_files"] = base_year_files
    high_activation_manifest["coactivation_sentence_file"] = coactivation_path.name
    high_activation_manifest["coactivation_sentence_by_year_files"] = year_coactivation_files
    high_activation_manifest["coactivation_base_pair_counts_file"] = pair_counts_path.name
    manifest_path = high_dir / f"{target_word}_high_activation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(high_activation_manifest, f, ensure_ascii=False, indent=2)
    
    # Generate the text report.
    report_file = output_dir / f"{target_word}_non_target_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# {target_word} - activation analysis on top bases for non-target sentences\n\n")
        f.write(f"Generated at: {Path(__file__).stat().st_mtime}\n\n")
        f.write("=" * 80 + "\n\n")
        
        for base_id in top_base_ids:
            base_info = base_meta[base_id]
            f.write(f"## Base {base_id}\n")
            f.write(f"Cumulative drift: {base_info['cum_drift']:.4f}\n")
            f.write(f"Peak delta: {base_info['peak_delta']:.4f} ({base_info['years'][0]}->{base_info['years'][1]})\n")
            f.write(f"Non-target sentence count: {results[str(base_id)]['total_sentences_without_target']}\n\n")
            if evidence_filter:
                f.write(f"Evidence-candidate sentences (after filtering): {results[str(base_id)]['total_evidence_sentences_without_target']}\n\n")
            f.write(
                f"High-activation threshold: q={results[str(base_id)]['high_activation_quantile']:.3f} "
                f"(computed within each year)\n"
            )
            f.write(
                f"High-activation sentence count: {results[str(base_id)]['high_activation_sentence_count']}\n"
                f"Full ranked-sentence file: without_target_ranked_sentences/{results[str(base_id)]['ranked_sentences_file']}\n"
                f"High-activation sentence file: without_target_high_activation_sentences/{results[str(base_id)]['high_activation_sentences_file']}\n\n"
            )
            
            f.write("### Top 30 activated non-target sentences\n\n")
            for i, sent_info in enumerate(results[str(base_id)]["top_sentences"], 1):
                f.write(f"[{i:2d}] Year: {sent_info['year']}, activation: {sent_info['activation']:.4f}\n")
                f.write(f"     Document: {sent_info['doc_id']}\n")
                f.write(f"     Sentence: {sent_info['sentence'][:100]}...\n\n")
            
            f.write("### Yearly mean statistics\n\n")
            f.write("Year | All mean (nz) | With-target mean (nz) | Without-target mean (nz) | All nz/total | With-target nz/total | Without-target nz/total | All mean (with 0s) | With-target mean (with 0s) | Without-target mean (with 0s)\n")
            f.write("-" * 100 + "\n")
            for year in sorted(annual_stats.keys()):
                stats = annual_stats[year][str(base_id)]
                f.write(
                    f"{year} | {stats['all']['mean']:.4f} | {stats['with_target']['mean']:.4f} | "
                    f"{stats['without_target']['mean']:.4f} | "
                    f"{stats['all']['count']}/{stats['all']['total']} | "
                    f"{stats['with_target']['count']}/{stats['with_target']['total']} | "
                    f"{stats['without_target']['count']}/{stats['without_target']['total']} | "
                    f"{stats['all']['mean_including_zeros']:.4f} | "
                    f"{stats['with_target']['mean_including_zeros']:.4f} | "
                    f"{stats['without_target']['mean_including_zeros']:.4f}\n"
                )
            f.write("\n" + "=" * 80 + "\n\n")

        f.write("## Additional Exports\n")
        f.write(f"- Combined high-activation non-target sentences: without_target_high_activation_sentences/{combined_high_path.name}\n")
        f.write(f"- Base/year high-activation non-target sentences: {high_by_year_dir.name}/base_XXXXX/year_YYYY_{quantile_tag(high_activation_quantile)}.jsonl\n")
        f.write(f"- Multi-base coactivation sentence table: {coactivation_dir.name}/{coactivation_path.name}\n")
        f.write(f"- Multi-base pair-count table: {coactivation_dir.name}/{pair_counts_path.name}\n")
    
    # Generate the yearly summary table.
    summary_file = output_dir / f"{target_word}_annual_means_summary.json"
    summary_data = {}
    for year in sorted(annual_stats.keys()):
        summary_data[year] = {}
        for base_id in top_base_ids:
            stats = annual_stats[year][str(base_id)]
            summary_data[year][str(base_id)] = {
                "all_mean": stats["all"]["mean"],
                "with_target_mean": stats["with_target"]["mean"],
                "without_target_mean": stats["without_target"]["mean"],
                "all_count": stats["all"]["count"],
                "with_target_count": stats["with_target"]["count"],
                "without_target_count": stats["without_target"]["count"],
                "all_total": stats["all"]["total"],
                "with_target_total": stats["with_target"]["total"],
                "without_target_total": stats["without_target"]["total"],
                "all_mean_including_zeros": stats["all"]["mean_including_zeros"],
                "with_target_mean_including_zeros": stats["with_target"]["mean_including_zeros"],
                "without_target_mean_including_zeros": stats["without_target"]["mean_including_zeros"],
            }
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Analyze activations on top bases for sentences without the target word")
    parser.add_argument('--word', required=True, help="Target word")
    parser.add_argument('--aliases', default=None, help="Extra aliases, comma-separated, used for contains_target checks")
    parser.add_argument('--aliases_file', default=None, help="Alias file, one per line, with optional # comments")
    parser.add_argument('--match_mode', default="substring", choices=["substring", "regex"], help="Matching mode for contains_target")
    parser.add_argument('--target_regex', default=None, help="Regular expression used when match_mode=regex")
    parser.add_argument('--years', nargs='*', type=int, default=None, help="Year list")
    parser.add_argument('--sent_dir', required=False, help="Source sentence directory (optional when --skip_extraction is set)")
    parser.add_argument('--meta_path', required=False, help="Metadata JSONL path (optional when --skip_extraction is set)")
    parser.add_argument('--top_bases_file', required=True, help="Top-bases file path")
    parser.add_argument('--sae_ckpt', required=False, help="SAE checkpoint path (optional when --skip_extraction is set)")
    parser.add_argument('--llama_path', required=False, help="LLaMA/tokenizer path (optional when --skip_extraction is set)")
    parser.add_argument('--device', default="cuda:0", help="Device")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('--all_activations_dir', default=None, help="all_activations input/output directory (default: <output_dir>/all_activations)")
    parser.add_argument('--top_n_sentences', type=int, default=30, help="Number of top sentences to keep per base")
    parser.add_argument('--skip_extraction', action='store_true', help="Skip activation extraction and analyze existing data only")
    parser.add_argument('--skip_analysis', action='store_true', help="Only extract all_activations and skip Top-bases analysis and report generation")
    parser.add_argument('--max_tokens', type=int, default=4096, help="Maximum tokens per sentence; longer inputs are truncated")
    parser.add_argument('--oom_recover', action='store_true', help="Skip a sentence and continue after OOM errors")
    parser.add_argument('--no_trunc_log', action='store_true', help="Disable truncation warning logs")
    parser.add_argument('--recompute_contains_target', action='store_true', help="Recompute contains_target during analysis using aliases, match_mode, and target_regex")
    parser.add_argument('--context_mode', default="prepend", choices=["prepend", "none"], help="Whether to prepend the previous line as context")
    parser.add_argument('--min_context_tokens', type=int, default=20, help="Minimum token target when context_mode=prepend")
    parser.add_argument('--num_shards', type=int, default=1, help="Total number of file shards when extracting all_activations in parallel")
    parser.add_argument('--shard_index', type=int, default=0, help="Current shard index for all_activations extraction, starting from 0")
    parser.add_argument('--no_evidence_filter', action='store_true', help="Disable evidence-sentence filtering")
    parser.add_argument('--no_evidence_dedupe', action='store_true', help="Disable evidence-sentence deduplication")
    parser.add_argument('--evidence_max_per_doc', type=int, default=2, help="Maximum number of evidence sentences to keep per document when deduplication is enabled")
    parser.add_argument('--high_activation_quantile', type=float, default=0.95, help="Quantile threshold used to export high-activation non-target sentences within each base/year distribution")
    
    args = parser.parse_args()
    if not (0.0 < float(args.high_activation_quantile) <= 1.0):
        raise ValueError("--high_activation_quantile must be in the interval (0, 1]")

    aliases = load_aliases(args.aliases, args.aliases_file)
    matcher = make_text_matcher(
        args.word,
        aliases,
        match_mode=args.match_mode,
        target_regex=args.target_regex,
    )

    if not args.skip_extraction:
        missing = [
            name
            for name, value in [
                ("sent_dir", args.sent_dir),
                ("meta_path", args.meta_path),
                ("sae_ckpt", args.sae_ckpt),
                ("llama_path", args.llama_path),
            ]
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing required arguments: "
                + ", ".join(missing)
                + " (required unless --skip_extraction is enabled)"
            )
    
    output_dir = Path(args.output_dir)
    all_activations_dir = (
        Path(args.all_activations_dir)
        if args.all_activations_dir
        else (output_dir / "all_activations")
    )
    
    if not args.skip_extraction:
        extract_all_activations(
            target_word=args.word,
            years=args.years,
            sent_dir=args.sent_dir,
            meta_path=args.meta_path,
            out_dir=all_activations_dir,
            sae_ckpt=args.sae_ckpt,
            llama_path=args.llama_path,
            device=args.device,
            max_tokens=args.max_tokens,
            oom_recover=args.oom_recover,
            log_truncation=(not args.no_trunc_log),
            matcher=matcher,
            min_context_tokens=args.min_context_tokens,
            context_mode=args.context_mode,
            shard_index=int(args.shard_index),
            num_shards=int(args.num_shards),
        )

    if args.skip_analysis:
        return
    
    tokenizer = None
    if args.recompute_contains_target and args.llama_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.llama_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    analyze_top_bases_activations(
        target_word=args.word,
        top_bases_file=args.top_bases_file,
        all_activations_dir=all_activations_dir,
        output_dir=output_dir,
        top_n_sentences=args.top_n_sentences,
        matcher=matcher,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        recompute_contains_target=args.recompute_contains_target,
        evidence_filter=(not args.no_evidence_filter),
        evidence_dedupe=(not args.no_evidence_dedupe),
        evidence_max_per_doc=max(0, int(args.evidence_max_per_doc)),
        high_activation_quantile=float(args.high_activation_quantile),
    )


if __name__ == '__main__':
    main()
