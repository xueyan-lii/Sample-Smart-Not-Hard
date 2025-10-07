## SAMPLE SMART, NOT HARD: CORRECTNESS-FIRST DECODING FOR BETTER REASONING IN LLMS
This repository contains utilities to:
- Evaluate models via the lm-eval-harness using vLLM (`run.py`)
- Generate per-step probability diagnostics and rank logs (`run_writeout.py`)
- Post-process logs to analyze rank/probability/accuracy relationships (`eval/post_processing/merge_low_conf_ranks_seq.py`)
- Collect token-level logits from a training/eval loop in Torchtune (`calibration_grid/get_logit_values.py`)
- Build calibration grids and simple power-law fits from logged probabilities (`eval/post_processing/per_step_calibration_visualization.ipynb`)

## Environment

- Python 3.10+
- CUDA-capable GPU recommended for vLLM and Torchtune

### Suggested setup
```bash
# Create and activate a virtualenv (or conda)
python -m venv .venv
source .venv/bin/activate

pip install vllm lm-eval torchtune torch torchdata

pip install matplotlib pandas numpy omegaconf tqdm
```

Notes:
- Install a CUDA-matching PyTorch if needed (see `https://pytorch.org/get-started/locally/`).
- Some Torchtune configs require specific model/tokenizer checkpoints you must provide.

## Directory structure (high level)
- `run.py`: vLLM-backed evaluation through lm-eval-harness
- `run_writeout.py`: vLLM-backed evaluation + per-token probability logging and low-confidence rank JSONL outputs
- `dynamic_topk.py`: unified logits processor (temperature, top-p, min-p, thresholds, calibrated constraints, etc.)
- `eval/`:
  - `post_processing/merge_low_conf_ranks_seq.py`: merges low-confidence rank JSONL with evaluation JSON to study rank vs accuracy
  - `post_processing/figure0.py`, `figure1.py`: plotting utilities
  - `post_processing/per_step_calibration_visualization.ipynb`: build calibration grids and fits from top-k traces
  - task YAMLs under `eval/*` for lm-eval (e.g., GSM8K, BBH)
- `calibration_grid/get_logit_values.py` (+ YAML): Torchtune-based loop to dump per-token top-k ids/probs JSONL

## 1) Run evaluation with lm-eval + vLLM

Example (GSM8K few-shot):
```bash
python sample_smart_not_hard/run.py \
  --task-config sample_smart_not_hard/eval/maths/gsm8k_fewshot.yaml \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir sample_smart_not_hard/eval \
```

Common knobs (all optional, forwarded to `dynamic_topk.py`):
- `--temperature`, `--top-p`, `--top-k`, `--min-p`
- `--greedy-threshold`
- `--prob-threshold` (epsilon-sampling)
- `--dynamic-top-k` (comma-separated 10 integers)
- `--edt-var` (comma-separated N,theta,T0 for entropy-dependent temperature)
- `--hewitt-epsilon`
- `--confidence-bin-only` (1..10)
- `--cal-acc` (A,B,thr for power-law calibrated accuracy: log c = A + B log p)

Outputs: if `--output-dir` is provided, metrics JSON is written there with a timestamped filename.

## 2) Run evaluation and log per-step probabilities and ranks

`run_writeout.py` writes two JSONL logs (when `--output-dir` is provided):
- `low_conf_ranks_*.jsonl`: ranks at steps where the top-1 probability was below a threshold (defaults in script)
- `all_sampled_ranks_*.jsonl`: ranks where chosen token probability was below a separate threshold

Example:
```bash
python sample_smart_not_hard/run_writeout.py \
  --task-config sample_smart_not_hard/eval/maths/gsm8k_fewshot.yaml \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir sample_smart_not_hard/eval \
```
The program prints the output JSONL paths at the end.

## 3) Merge rank logs with evaluation output and analyze rank vs accuracy

Use `eval/post_processing/merge_low_conf_ranks_seq.py` to attach the logged ranks to the evaluation samples and compute/plot:
- Accuracy vs number of low-confidence tokens
- k-sampled accuracy (1..9, `10`=10+)
- Length-filtered accuracy (len(ranks) == 1)
- Mean-rank binned accuracies and conditioned majority voting

Environment variables for inputs/outputs:
- `SAMPLES_JSON`: path to your evaluation JSON (written by `run_writeout.py` or lm-eval)
- `RANKS_JSONL`: path to `low_conf_ranks_*.jsonl`
- `OUT_DIR`: output directory for plot(s) and console summaries (default `./outputs`)

Example:
```bash
export SAMPLES_JSON=sample_smart_not_hard/eval/MyModel_20250101_123456.json
export RANKS_JSONL=sample_smart_not_hard/eval/low_conf_ranks_20250101_123500.jsonl
export OUT_DIR=sample_smart_not_hard/eval

python sample_smart_not_hard/eval/post_processing/merge_low_conf_ranks_seq.py
```
This will print summary stats and save `accuracy_vs_low_conf_tokens_seq.png` under `OUT_DIR`.

## 4) Collect per-token logits with Torchtune

`calibration_grid/get_logit_values.py` runs a single-device Torchtune loop and streams top-k per-token probabilities and token ids to a JSONL.

- Configure `calibration_grid/get_logit_values.yaml` with your model checkpoint and tokenizer paths.
- Ensure `log_topk: True` and set `topk_k` as desired.
- The script creates a timestamped file under `output_dir` unless `topk_output_path` is set.

Run:
```bash
python sample_smart_not_hard/calibration_grid/get_logit_values.py \
  --config sample_smart_not_hard/calibration_grid/get_logit_values.yaml
```
You can override entries via `--override key=value` pairs, e.g.:
```bash
python sample_smart_not_hard/calibration_grid/get_logit_values.py \
  --config sample_smart_not_hard/calibration_grid/get_logit_values.yaml \
  --override output_dir=./softmax_values log_topk=True topk_k=20
```

The JSONL will contain entries like:
```json
{"gt": 123, "top_ids": [42, 17, ...], "top_probs": ["0.93210000", "0.02130000", ...]}
```

## 5) Build calibration grids and simple fits

Open `eval/post_processing/per_step_calibration_visualization.ipynb` and set:
- Env `TOPK_TRACES_PATH` to the JSONL produced by step (4)
- Env `CALIBRATION_DIR` to an output directory for figures

The notebook will:
- Bin examples by top-1 probability into 10 bins
- Compute rank-wise correctness and (conditional/unconditional) means
- Plot a heatmap (optionally saving as `CALIBRATION_DIR/{model}-{dataset}.pdf`)
- Fit a power-law in log-space: `log c = A + B log p`

## Optional: Confidence-bin accuracy curves

`eval/post_processing/figure1.py` computes/plots majority-voted accuracy vs k for 10 confidence bins. You supply 10 JSON files (one per bin) and it will save the figure.

Output path can be customized via env `CONF_BIN_PLOT_PATH` (default `confidence_bin_plot.png`).

## Environment variables overview
- `SAMPLES_JSON`: path to evaluation samples JSON
- `RANKS_JSONL`: path to `low_conf_ranks_*.jsonl`
- `OUT_DIR`: directory for post-processing plots (`merge_low_conf_ranks_seq.py`)
- `TOPK_TRACES_PATH`: path to Torchtune top-k traces JSONL
- `CALIBRATION_DIR`: directory to save calibration plots from the notebook
- `CONF_BIN_PLOT_PATH`: output path for `figure1.py` plot

## Notes and tips
- vLLM requires sufficient GPU memory; adjust `--gpu-memory-utilization` and `--tensor-parallel-size` accordingly.
- For lm-eval tasks, pass an absolute path to the YAML so the include path is resolved correctly.
- Torchtune config requires valid `checkpointer` and `tokenizer` paths; update `get_logit_values.yaml` to match your local checkpoints.
