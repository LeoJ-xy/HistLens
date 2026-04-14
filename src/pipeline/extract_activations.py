#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STEP-1 (generic)
Extract SAE base activations for target-word tokens in sentences that contain the target word.
"""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple
import torch
import transformers

# OpenSAE may attempt to derive a version from the current working tree.
# Provide a fallback so the extractor still works when this code is copied
# into a standalone repository or launched from a non-git directory.
os.environ.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "0.0.0")
from opensae.config_utils import PretrainedSaeConfig
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


def find_target_spans(text: str, target_word: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    while True:
        pos = text.find(target_word, start)
        if pos < 0:
            break
        spans.append((pos, pos + len(target_word)))
        start = pos + max(1, len(target_word))
    return spans


def encode_sentence(
    tokenizer: transformers.PreTrainedTokenizerBase,
    sentence: str,
) -> tuple[transformers.BatchEncoding, Optional[List[Tuple[int, int]]]]:
    try:
        enc = tokenizer(
            sentence,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = [tuple(map(int, pair)) for pair in enc.pop("offset_mapping")[0].tolist()]
        return enc, offsets
    except Exception:
        enc = tokenizer(sentence, return_tensors="pt")
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


def token_positions_for_target(
    sentence: str,
    target_word: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    offsets: Optional[Sequence[Tuple[int, int]]],
    seq_len: int,
) -> List[int]:
    if offsets is not None:
        spans = find_target_spans(sentence, target_word)
        if spans:
            idxs = [
                i
                for i, (start, end) in enumerate(offsets[:seq_len])
                if start != end and any(start < span_end and end > span_start for span_start, span_end in spans)
            ]
            if idxs:
                return idxs

    toks = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    return [
        i
        for i, tk in enumerate(toks[:seq_len])
        if target_word in tokenizer.convert_tokens_to_string([tk])
    ]


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

def extract_activations_by_year(
    target_word,
    years,
    sent_dir,
    meta_path,
    out_dir,
    sae_ckpt,
    llama_path,
    device,
    corpus=None
):
    sent_dir = Path(sent_dir)
    # Accept either a corpus root (…/sentences) or an already resolved corpus
    # directory (…/sentences/<corpus>) to avoid double-appending the corpus name.
    if corpus and sent_dir.name != corpus and (sent_dir / corpus).exists():
        sent_dir = sent_dir / corpus
    meta_path = Path(meta_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sae   = load_sae_checkpoint(sae_ckpt)
    model = TransformerWithSae(llama_path, sae, device=device)
    tok   = transformers.AutoTokenizer.from_pretrained(llama_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # ---------- Metadata: filename -> year ----------
    fname2year = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for ln in f:
            obj = json.loads(ln)
            fname2year[obj["filename"]] = int(obj["pub_date"][:4])

    def extend_line(lines, idx, min_tok=20):
        sent = lines[idx].strip()
        ids  = tok(sent)["input_ids"]
        j = idx - 1
        while len(ids) < min_tok and j >= 0:
            sent = lines[j].strip() + " " + sent
            ids  = tok(sent)["input_ids"]
            j   -= 1
        return sent

    def acts_from_token_list(tokens_list):
        agg = {}
        for tk in tokens_list:
            if target_word not in tk["token"]:
                continue
            for pair in tk.get("activations", []):
                bid = int(pair["base_vector"])
                val = float(pair["activation"])
                if bid not in agg or val > agg[bid]:
                    agg[bid] = val
        return agg

    def extract_acts(sentence: str) -> dict:
        if target_word not in sentence:
            return {}
        enc, offsets = encode_sentence(tok, sentence)
        sparse_indices, sparse_acts, seq_len = collect_sparse_rows(model, enc)
        if sparse_indices is None or sparse_acts is None or seq_len == 0:
            return {}
        idxs = token_positions_for_target(
            sentence=sentence,
            target_word=target_word,
            tokenizer=tok,
            input_ids=enc["input_ids"][0],
            offsets=offsets,
            seq_len=seq_len,
        )
        if not idxs:
            return {}
        return aggregate_sparse_features(sparse_indices, sparse_acts, idxs)

    year_lines = defaultdict(list)
    total = kept = 0
    for txt in sent_dir.glob("*.txt"):
        yr = fname2year.get(txt.name)
        if yr is None and txt.name[:4].isdigit():
            yr = int(txt.name[:4])
        if yr is None or (years and yr not in years):
            continue
        lines = txt.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines):
            if target_word not in line:
                continue
            total += 1
            sent = extend_line(lines, idx)
            acts = extract_acts(sent)
            if not acts:
                continue
            kept += 1
            year_lines[yr].append(json.dumps({
                "doc_id": txt.stem,
                "sentence": sent,
                "base_activations": acts
            }, ensure_ascii=False))
    for yr, records in sorted(year_lines.items()):
        out_f = out_dir / f"{target_word}_activations_{yr}.jsonl"
        out_f.write_text("\n".join(records), encoding="utf-8")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Step 01: extract activations")
    parser.add_argument('--word', required=True, help="Target word")
    parser.add_argument('--years', nargs='*', type=int, default=None, help="Year list")
    parser.add_argument('--sent_dir', required=True, help="Root directory for source sentences")
    parser.add_argument('--meta_path', required=True, help="Path to the metadata JSONL file")
    parser.add_argument('--out_dir', required=True, help="Output directory")
    parser.add_argument('--sae_ckpt', required=True, help="Path to the SAE checkpoint")
    parser.add_argument('--llama_path', required=True, help="Path to the LLaMA model")
    parser.add_argument('--device', default="cuda:0", help="Device")
    parser.add_argument('--corpus', default=None, help="Corpus name")
    args = parser.parse_args()
    extract_activations_by_year(
        target_word=args.word,
        years=args.years,
        sent_dir=args.sent_dir,
        meta_path=args.meta_path,
        out_dir=args.out_dir,
        sae_ckpt=args.sae_ckpt,
        llama_path=args.llama_path,
        device=args.device,
        corpus=args.corpus
    ) 
