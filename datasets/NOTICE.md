# Attack CSV provenance

These CSVs are evaluation benchmarks for the SALO probe. Each file has two columns:

| Column | Meaning |
|---|---|
| `prompt` | A harmful behaviour drawn from **AdvBench** (Zou et al., 2023). |
| `injection` | The adversarial payload produced by the attack listed in the filename, optimized against the model named in the filename. |

| File | Attack | Target model | # rows |
|---|---|---|---|
| `qwen-7B-gcg.csv`       | GCG     | Qwen2.5-7B-Instruct  | 520 |
| `qwen-7B-AutoDAN.csv`   | AutoDAN | Qwen2.5-7B-Instruct  | 252 |
| `llama-8B-gcg.csv`      | GCG     | Llama-3.1-8B-Instruct | 520 |
| `llama-8B-AutoDAN.csv`  | AutoDAN | Llama-3.1-8B-Instruct | 250 |
| `mistral-7B-gcg.csv`    | GCG     | Mistral-7B-Instruct  | 520 |
| `mistral-7B-AutoDAN.csv`| AutoDAN | Mistral-7B-Instruct  | 258 |

How `injection` is interpreted at eval time:

* **GCG** — adversarial *suffix*; concatenated to the user prompt **inside** the chat template's user turn (`eval_attacks.py` detects `gcg` in the dataset name and does this automatically).

## Source attribution

* Harmful-behaviour prompts: **AdvBench** — Zou, Wang, Kolter, Fredrikson (2023), *Universal and Transferable Adversarial Attacks on Aligned Language Models*. https://github.com/llm-attacks/llm-attacks
* GCG suffixes: re-derived with the GCG algorithm from the same paper.
* AutoDAN templates: re-derived with **AutoDAN** — Liu et al. (2024), *AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models*. https://github.com/SheltonLiu-N/AutoDAN

## Intended use

These files are released **only** for evaluating jailbreak-detection methods (such as SALO). They do not contain new harmful capabilities beyond what is already public in the GCG and AutoDAN papers and their official repositories. GCG suffixes are model-specific gibberish and transfer poorly across models. Please do not redistribute as standalone "ready-to-use jailbreaks" or use them to attack production systems without authorization.
