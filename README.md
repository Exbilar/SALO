# SALO — Sparse Aware Latent Operator

A small CNN probe over a contiguous mid-stack window of an instruction-tuned LLM's hidden states, trained to detect jailbreak / harmful prompts. The probe is supervised on (safe, unsafe) prompt pairs from **PKU-SafeRLHF** and **lmsys/toxic-chat**, calibrated on **XSTest**, and evaluated on **AdvBench (direct + prefilling)**, **GCG suffixes**, and **AutoDAN**.

This repo is a clean re-implementation suitable for reproduction. The default config targets `Qwen2.5-7B-Instruct`; the same pipeline runs on Mistral-7B and Llama-3.1-8B by editing `configs.py`.

---

## Repo layout

```
salo_repo/
├── configs.py            # per-model paths and probe layer slice
├── process_data.py       # extract per-layer hidden states -> .pt files
├── train_probe.py        # train CNN probe + XSTest calibration + AdvBench eval
├── eval_attacks.py       # evaluate trained probe on attack CSVs (GCG, AutoDAN, ...)
├── requirements.txt
└── salo/
    ├── __init__.py
    ├── model.py          # hiddenDetector (CNN), linearProbe, MLP, RePE
    └── train.py          # ActivationDataset, fit(), predict(), eval helpers
```

---

## Install

PyTorch must match your CUDA driver. The original environment used CUDA 12.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Single-GPU is sufficient for everything below; ~24 GB VRAM (e.g. one A100/H100/4090) is enough to host the 7B base LLM in fp16 plus the probe.

---

## Data

Layout expected at runtime (relative to repo root):

```
./qwen2.5-7b/                          # local HF-format checkout of the base LLM
./datasets/
    safe-rlhf/                         # PKU-SafeRLHF      (download separately)
    toxic-chat/                        # lmsys/toxic-chat  (download separately)
    xstest/                            # walledai/XSTest   (download separately)
    advbench/harmful_behaviors.csv     # AdvBench          (download separately)
    qwen-7B-gcg.csv                    # shipped in this repo (see datasets/NOTICE.md)
    qwen-7B-AutoDAN.csv                # shipped in this repo
    llama-8B-gcg.csv                   # shipped in this repo
    llama-8B-AutoDAN.csv               # shipped in this repo
    mistral-7B-gcg.csv                 # shipped in this repo
    mistral-7B-AutoDAN.csv             # shipped in this repo
```

The four training/calibration datasets (safe-rlhf, toxic-chat, xstest, advbench) are **not** redistributed here — fetch them with `huggingface-cli download <repo> --local-dir ./datasets/<name>` (or `git lfs clone`). All paths above can be overridden via CLI flags.

The six attack CSVs in `datasets/` are shipped with this repo for ICML reproducibility; see `datasets/NOTICE.md` for provenance and intended-use notes.

The GCG / AutoDAN CSVs are two-column (`prompt`, `injection`):

* **GCG** — `injection` is an adversarial suffix that lives **inside** the user turn. `eval_attacks.py` detects a `gcg` substring in the dataset name and concatenates `prompt + injection` before applying the chat template.
* **AutoDAN / prefilling** — `injection` is appended **after** the assistant generation header (i.e. assistant-side prefill). This is the default behaviour for any non-GCG dataset.

---

## Pipeline

### 1) Extract activations

```bash
python process_data.py --model qwen2.5-7b
```

Writes `./residuals/qwen2.5-7b/train/{safe,unsafe}/<idx>.pt`. Each `.pt` is a tuple of `(end_layer - start_layer)` tensors of shape `(1, seq_len, hidden_dim)` in fp16.

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--model` | `qwen2.5-7b` | Key into `configs.MODEL_CONFIGS` |
| `--output-root` | `./residuals/<model>/train` | |
| `--max-safe-rlhf` | `4000` | Cap on safe-rlhf samples |
| `--max-toxic-chat-per-class` | `1500` | Per-class cap on toxic-chat |

Tip: residuals are many small files, so I/O dominates if `./residuals` lives on a slow shared FS. Copy/symlink to a fast local SSD (or `/dev/shm` for very small experiments) before training:

```bash
cp -r ./residuals /dev/shm/ && \
python train_probe.py --dataset-path /dev/shm/residuals/qwen2.5-7b/train
```

### 2) Train the probe (+ XSTest calibration + AdvBench eval)

```bash
python train_probe.py --model qwen2.5-7b
```

This trains the CNN probe, saves it to `./salo_qwen2.5-7b.pth`, scores XSTest, picks a threshold at the configured FPR (default 10%), and evaluates AdvBench Direct + Prefilling at that threshold. The threshold is also written to `./salo_qwen2.5-7b.threshold.json` (fields: `threshold`, `target_fpr`, `xstest_auroc`) so `eval_attacks.py` can re-use it without rerunning the XSTest pass.

Useful flags: `--epochs 5`, `--batch-size 16`, `--lr 1e-3`, `--num-workers 8`, `--pooling max|mean`, `--target-fpr 0.10`.

### 3) Evaluate against attack CSVs

```bash
python eval_attacks.py --model qwen2.5-7b \
    --datasets Qwen-7B-GCG:./datasets/qwen-7B-gcg.csv \
               Qwen-7B-AutoDAN:./datasets/qwen-7B-AutoDAN.csv
```

By default the threshold is loaded from `salo_<model>.threshold.json`. Pass `--recalibrate` to recompute it from XSTest before scoring attacks.

Each `--datasets` entry is `<name>:<csv_path>`. The `<name>` controls attack interpretation: if it contains `gcg` (case-insensitive), the `injection` column is treated as an in-user adversarial suffix; otherwise it is treated as an assistant-side prefill.

---

## Notes on probe placement (`start_layer` / `end_layer`)

The probe consumes a contiguous slice `outputs.hidden_states[start_layer:end_layer]` (end-exclusive). Mid-stack layers tend to expose the most linearly-separable safety features. Reasonable defaults already shipped:

| Model | start | end |
|---|---|---|
| Qwen2.5-7B | 10 | 20 |
| Mistral-7B | 10 | 20 |
| Llama-3.1-8B | 5 | 15 |

Changing the slice changes `hiddenDetector`'s effective receptive field along the layer axis but keeps `input_dim` (hidden size) unchanged — you only need to re-run `process_data.py` and `train_probe.py`.

---

## Ethics & intended use

The attack CSVs in `datasets/` (GCG / AutoDAN against Qwen2.5-7B, Llama-3.1-8B, Mistral-7B) are released **solely** to allow reproduction of the SALO detector. Harmful-behaviour prompts originate from AdvBench (Zou et al., 2023); the adversarial payloads were re-derived with the published GCG and AutoDAN algorithms — both of which already ship comparable attack strings in their official repositories, so this release does not introduce new offensive capability. GCG suffixes are model-specific gibberish with poor cross-model transfer. Please do not redistribute these files as standalone jailbreaks or use them against production systems without authorization. See `datasets/NOTICE.md` for per-file provenance.
