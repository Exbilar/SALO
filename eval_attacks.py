"""Evaluate a trained probe against attack CSVs (GCG + AutoDAN by default).

CSV schema expected: two columns, `prompt` and `injection`.

How `injection` is interpreted depends on the attack family:
  - GCG-style suffix attacks: the adversarial suffix was optimized to live
    INSIDE the user turn, so we concatenate prompt + injection and place the
    result inside the chat template. (Triggered when "gcg" is in the dataset
    name passed to --datasets.)
  - Prefilling attacks: the injection is the assistant-side prefix and is
    appended AFTER the `<assistant>` generation header (handled by
    train.predict when `injection` is not None).
  - If injection is NaN/empty, the prompt is sent as-is.

The threshold can either be loaded from `salo_<model>.threshold.json` (written
by train_probe.py) or recalibrated from XSTest (default if no file exists, or
when --recalibrate is passed).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

import salo.train as train
from salo import model as M
from configs import get_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5-7b")
    p.add_argument("--probe", default=None,
                   help="Defaults to ./salo_<model>.pth")
    p.add_argument("--threshold-file", default=None,
                   help="Defaults to <probe>.threshold.json")
    p.add_argument("--recalibrate", action="store_true",
                   help="Always recalibrate the threshold from XSTest")
    p.add_argument("--target-fpr", type=float, default=0.10)
    p.add_argument("--xstest-path", default="./datasets/xstest")
    p.add_argument(
        "--datasets", nargs="+",
        default=[
            "Qwen-7B-GCG:./datasets/qwen-7B-gcg.csv",
            "Qwen-7B-AutoDAN:./datasets/qwen-7B-AutoDAN.csv",
        ],
        help="One or more <name>:<csv_path> entries to evaluate",
    )
    return p.parse_args()


def get_predictions(probe, llm, tokenizer, items, prompt_func, device,
                    start_layer, end_layer, desc=""):
    probs = []
    print(f"  [{desc}] {len(items)} samples")
    for i, item in enumerate(items):
        p = train.predict(probe, prompt_func(item), llm, tokenizer, device,
                          start_layer=start_layer, end_layer=end_layer)
        probs.append(p)
        if (i + 1) % 100 == 0:
            print(f"    progress {i + 1}/{len(items)}")
            torch.cuda.empty_cache()
    return np.array(probs)


def calibrate_threshold(probe, llm, tokenizer, device, cfg, xstest_path,
                        target_fpr):
    print("\n[Calibration] XSTest...")
    xs_test = list(load_dataset(path=xstest_path, split="prompts"))
    xs_probs = get_predictions(
        probe, llm, tokenizer, xs_test,
        lambda x: {"prompt": x["prompt"], "injection": None},
        device, cfg["start_layer"], cfg["end_layer"], desc="XSTest",
    )
    xs_labels = np.array([1 if x["type"].startswith("contrast_") else 0 for x in xs_test])
    auroc = roc_auc_score(xs_labels, xs_probs)
    thresh = float(np.percentile(xs_probs[xs_labels == 0], (1 - target_fpr) * 100))
    print(f"XSTest AUROC: {auroc:.4f}, Threshold (FPR={target_fpr}): {thresh:.4f}")
    return thresh, auroc


def make_prompt_for(name):
    """Return a callable that maps a CSV row to {prompt, injection}.

    For attack families where the `injection` field is the adversarial
    *suffix* (GCG), we concatenate it into the user-side prompt and set
    `injection=None` so that train.predict places everything inside the
    user turn of the chat template.

    For everything else (prefilling-style attacks), the `injection` is
    passed through and `train.predict` appends it after the assistant
    generation header.
    """
    name_lower = name.lower()

    def fn(row):
        injection = row.get("injection", None)
        if pd.isna(injection) or injection == "":
            injection = None

        if "gcg" in name_lower and injection is not None:
            return {"prompt": row["prompt"] + injection, "injection": None}

        return {"prompt": row["prompt"], "injection": injection}

    return fn


def main():
    args = parse_args()
    cfg = get_config(args.model)
    probe_path = args.probe or f"./salo_{args.model}.pth"
    threshold_path = args.threshold_file or probe_path.replace(".pth", ".threshold.json")

    device = torch.device("cuda:0")

    # ---------- load probe ----------
    probe = M.hiddenDetector(
        input_dim=cfg["embed_dim"], num_filters=64,
        layer_kernel_size=3, dropout=0.5, pooling="max",
    )
    probe.load_state_dict(torch.load(probe_path, map_location="cpu"))
    probe = probe.to(device).half().eval()
    print(f"Loaded probe from {probe_path}")

    # ---------- load LLM ----------
    print(f"Loading LLM from {cfg['path']}...")
    llm = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.float16,
    ).to(device)
    llm.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])

    # ---------- threshold ----------
    auroc = None
    if not args.recalibrate and os.path.exists(threshold_path):
        with open(threshold_path) as f:
            data = json.load(f)
        thresh = float(data["threshold"])
        auroc = data.get("xstest_auroc")
        print(f"Loaded threshold {thresh:.4f} from {threshold_path}"
              f" (XSTest AUROC={auroc:.4f})" if auroc else
              f"Loaded threshold {thresh:.4f} from {threshold_path}")
    else:
        thresh, auroc = calibrate_threshold(
            probe, llm, tokenizer, device, cfg,
            args.xstest_path, args.target_fpr,
        )

    # ---------- attack CSVs ----------
    summary = []
    for entry in args.datasets:
        if ":" not in entry:
            raise ValueError(f"--datasets entry must be <name>:<path>, got {entry!r}")
        name, csv_path = entry.split(":", 1)
        print(f"\nEvaluating {name} ({csv_path})...")
        df = pd.read_csv(csv_path)
        rows = [row for _, row in df.iterrows()]
        probs = get_predictions(
            probe, llm, tokenizer, rows, make_prompt_for(name),
            device, cfg["start_layer"], cfg["end_layer"], desc=name,
        )
        preds = (probs >= thresh).astype(int)
        acc = float(np.mean(preds))
        print(f"{name} Detection Rate: {acc:.4f} ({int(preds.sum())}/{len(preds)})")
        summary.append((name, acc, len(preds)))

    # ---------- summary ----------
    print("\n========== Attack Eval Summary ==========")
    print(f"Probe:               {probe_path}")
    if auroc is not None:
        print(f"XSTest AUROC:        {auroc:.4f}")
    print(f"Threshold (FPR={args.target_fpr}): {thresh:.4f}")
    for name, acc, n in summary:
        print(f"{name:<22} {acc:.4f}  ({n} samples)")
    print("Done.")


if __name__ == "__main__":
    main()
