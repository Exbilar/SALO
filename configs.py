"""Per-LLM configuration used across process_data.py / train_probe.py / eval_attacks.py.

`path`            : local HF-format directory of the base LLM
`start/end_layer` : Python slice over `outputs.hidden_states` (end exclusive).
                    Pick a contiguous mid-stack window where safety features
                    are most linearly separable.
`embed_dim`       : LLM hidden size (== input_dim of the probe).
"""

MODEL_CONFIGS = {
    "qwen2.5-7b": {
        "path": "./qwen2.5-7b",
        "start_layer": 10,
        "end_layer": 20,
        "embed_dim": 3584,
    },
    "mistral-7B": {
        "path": "./mistral-7B",
        "start_layer": 10,
        "end_layer": 20,
        "embed_dim": 4096,
    },
    "llama3.1-8B": {
        "path": "./llama3.1-8B",
        "start_layer": 5,
        "end_layer": 15,
        "embed_dim": 4096,
    },
}


def get_config(model_key: str) -> dict:
    if model_key not in MODEL_CONFIGS:
        raise KeyError(
            f"Unknown model '{model_key}'. Available: {list(MODEL_CONFIGS)}"
        )
    return MODEL_CONFIGS[model_key]
