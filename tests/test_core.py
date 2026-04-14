from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from configs.load import load_config
from configs.run_utils import get_active_model, resolve_run_root
from pipeline.compute_centers import compute_centers_and_distances
from pipeline.export_target_ranked_sentences import export_target_ranked_sentences
from pipeline.identify_drift import identify_top_drift_bases
from pipeline.run import parse_stage_list


class HistSAESmokeTests(unittest.TestCase):
    def test_load_config_and_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = tmp_path / "data"
            font_path = tmp_path / "font.ttf"
            sae_ckpt = tmp_path / "sae-checkpoint"
            llama_path = tmp_path / "llm-model"
            for path in [data_root / "sentences", data_root / "metadata", data_root / "output", sae_ckpt, llama_path]:
                path.mkdir(parents=True, exist_ok=True)
            font_path.write_text("fake-font", encoding="utf-8")

            env_path = tmp_path / "local.yaml"
            env_path.write_text(
                "\n".join(
                    [
                        "paths:",
                        f"  data_root: {data_root}",
                        f"  sae_ckpt: {sae_ckpt}",
                        f"  llama_path: {llama_path}",
                        f"  font_path: {font_path}",
                        "runtime:",
                        "  device: cpu",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            exp_path = tmp_path / "exp.yaml"
            exp_path.write_text(
                "\n".join(
                    [
                        "experiment:",
                        "  name: smoke",
                        "  corpus: example_corpus",
                        "  words:",
                        "    - example_concept",
                        "  years:",
                        "    - 1915-1916",
                        f"  env_path: {env_path}",
                        "paths: {}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cfg = load_config(str(exp_path))
            self.assertEqual(cfg.experiment.years, [1915, 1916])
            model = get_active_model(cfg)
            run_root = resolve_run_root(cfg, model)

            self.assertEqual(run_root.name, "smoke")
            self.assertIn("example_corpus", run_root.parts)
            self.assertIn("llm-model", run_root.parts)
            self.assertIn("layer_-1", run_root.parts)

    def test_compute_centers_and_identify_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            act_dir = tmp_path / "activations"
            act_dir.mkdir()
            center_path = tmp_path / "yearly_centers.json"
            dist_path = tmp_path / "yearly_distances.json"
            drift_path = tmp_path / "top_drift_bases.json"

            records_1915 = [
                {"doc_id": "a", "sentence": "s1", "base_activations": {"1": 1.0, "2": 0.2}},
                {"doc_id": "b", "sentence": "s2", "base_activations": {"1": 0.5}},
            ]
            records_1916 = [
                {"doc_id": "c", "sentence": "s3", "base_activations": {"1": 2.0, "2": 0.1}},
                {"doc_id": "d", "sentence": "s4", "base_activations": {"2": 1.4}},
            ]

            for year, records in [(1915, records_1915), (1916, records_1916)]:
                path = act_dir / f"example_concept_activations_{year}.jsonl"
                path.write_text(
                    "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
                    encoding="utf-8",
                )

            compute_centers_and_distances("example_concept", act_dir, center_path, dist_path)
            identify_top_drift_bases(center_path, drift_path, top_n=2, source="example_corpus", concept="example_concept")

            centers = json.loads(center_path.read_text(encoding="utf-8"))
            drift = json.loads(drift_path.read_text(encoding="utf-8"))

            self.assertEqual(centers["base_order"], [1, 2])
            self.assertIn("1915->1916", json.loads(dist_path.read_text(encoding="utf-8")))
            self.assertEqual(drift["top_n"], 2)
            self.assertEqual(drift["top_bases"][0]["base_id"], 2)

    def test_export_target_ranked_sentences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            act_dir = tmp_path / "activations"
            out_dir = tmp_path / "exports"
            act_dir.mkdir()

            records = [
                {"doc_id": "a", "sentence": "alpha", "base_activations": {"2": 0.5, "7": 1.2}},
                {"doc_id": "b", "sentence": "beta", "base_activations": {"2": 0.1}},
            ]
            (act_dir / "example_concept_activations_1915.jsonl").write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
                encoding="utf-8",
            )

            export_target_ranked_sentences("example_concept", act_dir, out_dir)

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            by_mass = (out_dir / "year_1915_all_by_activation_mass.jsonl").read_text(encoding="utf-8").splitlines()

            self.assertEqual(manifest["word"], "example_concept")
            self.assertEqual(len(manifest["years"]), 1)
            self.assertIn("alpha", by_mass[0])

    def test_parse_stage_list(self) -> None:
        self.assertEqual(parse_stage_list(None)[0], "extract")
        self.assertEqual(parse_stage_list("extract,drift"), ["extract", "drift"])
        with self.assertRaises(ValueError):
            parse_stage_list("extract,unknown")


if __name__ == "__main__":
    unittest.main()
