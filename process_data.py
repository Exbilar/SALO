"""Extract per-layer hidden-state activations from the base LLM and save them
to disk, labelled safe/unsafe, for later probe training.

Outputs `.pt` files into `<output_root>/{safe,unsafe}/<idx>.pt`, each a tuple
of (end_layer - start_layer) tensors of shape (1, seq_len, hidden_dim), fp16.

Default data sources:
  - PKU-SafeRLHF (label: at least one of the two model responses is unsafe)
  - lmsys/toxic-chat (label: toxicity == 1)
"""

import argparse
import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs import get_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5-7b",
                   help="Key into configs.MODEL_CONFIGS")
    p.add_argument("--output-root", default=None,
                   help="Defaults to ./residuals/<model>/train")
    p.add_argument("--safe-rlhf-path", default="./datasets/safe-rlhf")
    p.add_argument("--toxic-chat-path", default="./datasets/toxic-chat")
    p.add_argument("--max-safe-rlhf", type=int, default=4000,
                   help="Cap on safe-rlhf samples to process (in order)")
    p.add_argument("--max-toxic-chat-per-class", type=int, default=1500,
                   help="Cap on toxic-chat samples per (safe/unsafe) class")
    p.add_argument("--toxic-chat-max-len", type=int, default=400,
                   help="Skip toxic-chat prompts longer than this many chars")
    return p.parse_args()


def fetch_residual(model, tokenizer, input_text, layer_slice, device):
    """Forward the prompt and return the hidden states for the selected layer slice.

    Returns a tuple of (1, seq_len, hidden_dim) fp16 CPU tensors, ready to
    `torch.save`. Saving on CPU + fp16 minimises both disk space and the
    cost of later DataLoader workers loading them back.
    """
    chat = [{"role": "user", "content": input_text}]
    tokens = tokenizer.apply_chat_template(
        chat, tokenize=True, return_tensors="pt", add_generation_prompt=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer_slice]

    return tuple(h.to(torch.float16).cpu() for h in hidden_states)


def main():
    args = parse_args()
    cfg = get_config(args.model)

    output_root = args.output_root or f"./residuals/{args.model}/train"
    os.makedirs(f"{output_root}/safe", exist_ok=True)
    os.makedirs(f"{output_root}/unsafe", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading LLM: {cfg['path']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    layer_slice = slice(cfg["start_layer"], cfg["end_layer"])

    # ---------- 1) safe-rlhf ----------
    print(f"[1/2] safe-rlhf -> {output_root}")
    dataset = load_dataset(path=args.safe_rlhf_path, split="train")
    cap = min(args.max_safe_rlhf, len(dataset))
    for idx, data in tqdm(enumerate(dataset), total=cap):
        if idx >= cap:
            break
        residuals = fetch_residual(model, tokenizer, data["prompt"],
                                   layer_slice, device)
        # Treat a prompt as unsafe if at least one of the two model responses
        # was annotated unsafe (a noisy but standard proxy for prompt-level harm).
        is_unsafe = (data["is_response_0_safe"] is False or
                     data["is_response_1_safe"] is False)
        sub = "unsafe" if is_unsafe else "safe"
        torch.save(residuals, f"{output_root}/{sub}/{idx}.pt")

    # ---------- 2) toxic-chat ----------
    print(f"[2/2] toxic-chat -> {output_root}")
    safe_cnt, unsafe_cnt = 0, 0
    id_offset = 10000  # avoid filename collisions with safe-rlhf
    toxic_chat = load_dataset(args.toxic_chat_path, "toxicchat0124", split="train")
    for idx, data in tqdm(enumerate(toxic_chat), total=len(toxic_chat)):
        # Drop labelled jailbreaks and overly long prompts.
        if data["jailbreaking"] == 1 or len(data["user_input"]) > args.toxic_chat_max_len:
            continue
        residuals = fetch_residual(model, tokenizer, data["user_input"],
                                   layer_slice, device)

        if data["toxicity"] == 1 and unsafe_cnt < args.max_toxic_chat_per_class:
            torch.save(residuals, f"{output_root}/unsafe/{id_offset + idx}.pt")
            unsafe_cnt += 1
        elif data["toxicity"] == 0 and safe_cnt < args.max_toxic_chat_per_class:
            torch.save(residuals, f"{output_root}/safe/{id_offset + idx}.pt")
            safe_cnt += 1

        if (safe_cnt >= args.max_toxic_chat_per_class and
                unsafe_cnt >= args.max_toxic_chat_per_class):
            break

    print("Done.")


if __name__ == "__main__":
    main()
