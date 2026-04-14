# HistSAE

HistSAE is a toolkit for historical semantic drift analysis with sparse autoencoders (SAEs). It supports concept-centered analysis over Chinese historical corpora, from target-sentence activation extraction to drift-base discovery, evidence sentence export, visualization, and non-target analysis.

## Features

- SAE activation extraction for concept-bearing sentences
- yearly center construction and inter-year distance measurement
- drift-base ranking by cumulative change across time
- peak evidence sentence export and top-sentence ranking
- wordcloud generation for highly activated bases
- non-target analysis for sentences that do not explicitly mention the target concept
- run manifests and structured output directories for reproducibility

## Repository layout

- `src/pipeline/`: main workflow stages
- `src/non_target/`: non-target activation analysis
- `src/analysis/`: visualization and reporting utilities
- `src/configs/`: config loading, schema validation, and run-root resolution
- `configs/`: experiment and local environment config files
- `scripts/run_pipeline.sh`: shell wrapper for the pipeline runner
- `tests/`: lightweight smoke tests for config loading and post-processing logic

## Data layout

HistSAE expects sentence files and metadata in a corpus-oriented directory structure:

```text
sentences/
  example_corpus/
    *.txt
metadata/
  example_corpus/
    documents.jsonl
```

Each metadata record should include at least:

- `filename`
- `pub_date`

An example record shape is:

```json
{"doc_id":"...","filename":"...txt","pub_date":"1915-09-15","...":"..."}
```

## Installation

1. Create a Python environment.
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install OpenSAE in the environment you use for extraction. The extraction scripts import:

- `opensae`
- `opensae.transformer_with_sae`
- `opensae.config_utils`

4. Copy the example environment file and fill in local paths:

```bash
cp configs/env/local.example.yaml configs/env/local.yaml
```

## Configuration

`configs/env/local.yaml` should define:

- `paths.data_root` or explicit `sentences_dir` / `metadata_dir` / `output_root`
- `paths.sae_ckpt`
- `paths.llama_path`
- `paths.font_path`

The example experiment config is provided in `configs/exp/default.yaml`.

## Running the pipeline

Run the default pipeline:

```bash
python src/pipeline/run.py --config configs/exp/default.yaml
```

Run selected stages only:

```bash
python src/pipeline/run.py --config configs/exp/default.yaml --stages extract,centers,drift
```

Run non-target analysis after the main drift outputs exist:

```bash
python src/pipeline/run.py --config configs/exp/default.yaml --stages non_target,non_target_visualize
```

Preview the execution plan without running computation:

```bash
python src/pipeline/run.py --config configs/exp/default.yaml --dry-run
```

You can also use the shell wrapper:

```bash
bash scripts/run_pipeline.sh --config configs/exp/default.yaml
```

## Output structure

Each run is organized under:

```text
output/{corpus}/{model_name}/layer_{layer}/sae_{sae_id}/{experiment_name}/{concept}/
```

Typical per-concept outputs include:

- `activations/`
- `yearly_centers.json`
- `yearly_distances.json`
- `top_drift_bases.json`
- `key_bases_peak_change.json`
- `sentences/`
- `top30_sentences/`
- `wordclouds/`
- `target_sorted_sentences_by_year/`
- `visualizations/`
- `non_target_analysis/`

## Main entry points

The central code paths are:

- `src/pipeline/run.py`
- `src/pipeline/extract_activations.py`
- `src/pipeline/compute_centers.py`
- `src/pipeline/identify_drift.py`
- `src/non_target/analyze_non_target_activations.py`
- `src/analysis/visualize.py`
- `src/analysis/target_vs_non_target.py`
- `src/analysis/visualize_non_target.py`

## Testing

The smoke tests cover config loading and pure post-processing stages, and do not require model weights.

Run them with:

```bash
python -m unittest discover -s tests -v
```
