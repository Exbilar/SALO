"""Train the SALO CNN probe and report XSTest AUROC + AdvBench detection.

Pipeline:
  1. Train hiddenDetector on saved activations (residuals/<model>/train/{safe,unsafe})
  2. Save probe weights to salo_<model>.pth (fp32)
  3. Load the base LLM and score XSTest to compute AUROC
  4. Calibrate a threshold at TARGET_FPR on the XSTest safe split
  5. Evaluate AdvBench Direct and AdvBench Prefilling at that threshold

Threshold is also written to salo_<model>.threshold.json so eval_attacks.py can
reuse it without rerunning the XSTest pass.
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

import salo.train as train
from configs import get_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5-7b")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pooling", default="max", choices=["max", "mean"])
    p.add_argument("--dataset-path", default=None,
                   help="Defaults to ./residuals/<model>/train")
    p.add_argument("--probe-out", default=None,
                   help="Defaults to ./salo_<model>.pth")
    p.add_argument("--xstest-path", default="./datasets/xstest")
    p.add_argument("--advbench-path",
                   default="./datasets/advbench/harmful_behaviors.csv")
    p.add_argument("--target-fpr", type=float, default=0.10)
    p.add_argument("--num-workers", type=int, default=8)
    return p.parse_args()


def get_predictions(probe, llm, tokenizer, items, prompt_func, device,
                    start_layer, end_layer, desc=""):
    probs = []
    print(f"  [{desc}] {len(items)} samples")
    for i, item in enumerate(items):
        p = train.predict(probe, prompt_func(item), llm, tokenizer, device,
                          start_layer=start_layer, end_layer=end_layer)
        probs.append(p)
        if (i + 1) % 50 == 0:
            print(f"    progress {i + 1}/{len(items)}")
            torch.cuda.empty_cache()
    return np.array(probs)


def main():
    args = parse_args()
    cfg = get_config(args.model)
    dataset_path = args.dataset_path or f"./residuals/{args.model}/train"
    probe_out = args.probe_out or f"./salo_{args.model}.pth"
    threshold_out = probe_out.replace(".pth", ".threshold.json")

    device = torch.device("cuda:0")
    print(f"Detected {torch.cuda.device_count()} GPU(s). Running on {device}.")

    # ---------- 1) train the probe ----------
    print("Starting Training (Single GPU)...")
    probe, _ = train.fit(
        num_epochs=args.epochs,
        input_dim=cfg["embed_dim"],
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        pooling=args.pooling,
        dataset_path=dataset_path,
        num_workers=args.num_workers,
    )
    torch.save(probe.state_dict(), probe_out)
    print(f"Probe saved to {probe_out}")
    probe = probe.to(device).half().eval()

    # ---------- 2) load LLM ----------
    print(f"Loading LLM from {cfg['path']}...")
    llm = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.float16,
    ).to(device)
    llm.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])

    # ---------- 3) XSTest calibration ----------
    print("\n[Step 1] Calibrating on XSTest...")
    xs_test = list(load_dataset(path=args.xstest_path, split="prompts"))
    xs_probs = get_predictions(
        probe, llm, tokenizer, xs_test,
        prompt_func=lambda x: {"prompt": x["prompt"], "injection": None},
        device=device, start_layer=cfg["start_layer"], end_layer=cfg["end_layer"],
        desc="XSTest",
    )
    # In XSTest, `contrast_*` types are the unsafe twins of safe-looking prompts.
    xs_labels = np.array([1 if x["type"].startswith("contrast_") else 0 for x in xs_test])
    safe_scores = xs_probs[xs_labels == 0]
    unsafe_scores = xs_probs[xs_labels == 1]
    print(f"   Avg Safe Score:   {np.mean(safe_scores):.4f}")
    print(f"   Avg Unsafe Score: {np.mean(unsafe_scores):.4f}")

    xs_auroc = roc_auc_score(xs_labels, xs_probs)
    print(f"XSTest AUROC: {xs_auroc:.4f}")

    threshold = float(np.percentile(safe_scores, (1 - args.target_fpr) * 100))
    print(f"Calibrated Threshold (FPR={args.target_fpr}): {threshold:.4f}")
    with open(threshold_out, "w") as f:
        json.dump({"threshold": threshold, "target_fpr": args.target_fpr,
                   "xstest_auroc": xs_auroc}, f, indent=2)
    print(f"Threshold written to {threshold_out}")

    # ---------- 4) AdvBench eval ----------
    def evaluate_attack(name, items, prompt_func):
        print(f"\nEvaluating {name}...")
        probs = get_predictions(probe, llm, tokenizer, items, prompt_func,
                                device, cfg["start_layer"], cfg["end_layer"],
                                desc=name)
        acc = float(np.mean((probs >= threshold).astype(int)))
        print(f"{name} Detection Rate: {acc:.4f}")
        return acc

    adv_df = pd.read_csv(args.advbench_path)
    adv_rows = [row for _, row in adv_df.iterrows()]

    direct = evaluate_attack(
        "AdvBench-Direct", adv_rows,
        lambda x: {"prompt": x["goal"], "injection": None},
    )
    prefilling = evaluate_attack(
        "AdvBench-Prefilling", adv_rows,
        lambda x: {"prompt": x["goal"], "injection": x["target"]},
    )

    print("\n========== Summary ==========")
    print(f"Model:               {args.model}")
    print(f"XSTest AUROC:        {xs_auroc:.4f}")
    print(f"Threshold (FPR={args.target_fpr}): {threshold:.4f}")
    print(f"AdvBench Direct:     {direct:.4f}")
    print(f"AdvBench Prefilling: {prefilling:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
