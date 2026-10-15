# Puns-LLM

**Evaluating Large Language Models for Portuguese Pun Location and Linguistic Interpretation**

This repository contains the code, prompts, predictions, and evaluation results for **Gemma 3 27B** and **Qwen3 32B** on 2,850 Portuguese puns from *Puntuguese*. The study compares zero-shot and one-shot prompting, with and without gold pun locations (oracle). No fine-tuning was performed.

## Installation

Requirements: **Python 3.12**, [uv](https://docs.astral.sh/uv/getting-started/installation/), and an [Ollama](https://ollama.com/) server for new model inference. An Ollama server is **not** needed to evaluate the saved predictions.

From the repository root:

```bash
uv python install 3.12
uv sync --locked
uv run --locked pytest -q
```

The dependency versions are recorded in `uv.lock`. The environment used for the reported evaluation included Python **3.12.14**, uv **0.13.0**, NumPy **2.5.4**, pandas **2.3.3**, scikit-learn **1.9.1**, pytest **9.1.1**, and Ruff **0.17.0**.

## Dataset and tasks

The experiments use all **2,850 original pun texts** (2,909 annotated pun signs) from the public *Puntuguese* corpus, excluding their non-punning micro-edited counterparts. Each model is asked to:

1. Locate the punning expression (*pun sign*).
2. Recover its alternative written expression(s).
3. Identify whether the forms are homographic and/or homophonic.

In the oracle protocols, the gold pun signs are supplied; the alternatives and linguistic properties must still be predicted. The one-shot protocols use a fixed in-context example involving *notas* (musical notes/banknotes), not model training.

The prepared evaluation data are in `data/processed/puns_only.json`. The original public data used for preparation are in `data/raw/`.

## Prompts and configuration

| Protocol | Prompt file |
| --- | --- |
| Zero-shot (`zero_shot_v2`) | `configs/prompts/zero_shot_v2.txt` |
| One-shot (`one_shot`) | `configs/prompts/one_shot.txt` |
| Oracle zero-shot (`oracle_zero_shot_v2`) | `configs/prompts/oracle_zero_shot_v2.txt` |
| Oracle one-shot (`oracle_one_shot`) | `configs/prompts/oracle_one_shot.txt` |

Model identifiers and settings are in `configs/models/ollama.json`; inference parameters and the Ollama API address are in `configs/experiments/zero_shot.json`. The implementation is in `src/puns_llm/`.

## Reproduce the reported evaluations

The saved results are available in `results/official/`. To recalculate metrics from the stored predictions **without running the models**, use:

```bash
uv run --locked puns-llm --action evaluate-official \
  --model gemma3:27b --protocol one_shot
```

Replace `--model` with `gemma3:27b` or `qwen3:32b`, and `--protocol` with one of the four identifiers above to evaluate any of the eight recorded conditions.

Each `results/official/<model>/<protocol>/` directory contains:

- `predictions.jsonl` — prediction for each input text.
- `evaluation.json` — aggregate metrics reported in the paper.
- `details.jsonl` — per-text scoring and error-analysis information.
- `summary.json` and `metadata.json` — execution summary and settings.

### Main results

Strict accuracy is the percentage of texts for which all required alternative expressions and linguistic labels are correct (and, in conventional prompting, the pun signs must also be located correctly).

| Prompting condition | Gemma 3 27B | Qwen3 32B |
| --- | ---: | ---: |
| Zero-shot, strict accuracy | 0.42% | 5.16% |
| One-shot, strict accuracy | 6.67% | 7.02% |
| Oracle zero-shot, strict interpretation accuracy | 0.32% | 8.91% |
| Oracle one-shot, strict interpretation accuracy | 7.72% | 15.79% |

The highest conventional exact-location F1 was **60.36%** (Gemma, one-shot). Oracle and conventional strict accuracy measure different conditions and should not be interpreted as directly interchangeable. All 2,850 pun texts were processed in each condition; the token-level metric uses only records with valid token alignment.

## Run new inference

The Ollama server must be reachable at the address in `configs/experiments/zero_shot.json` (default: `http://127.0.0.1:11434`). The models used are `gemma3:27b` and `qwen3:32b`. The Ollama CLI may be installed on a host rather than inside the Python container.

To run one condition and then evaluate it:

```bash
uv run --locked puns-llm --action run-official \
  --model gemma3:27b --protocol one_shot

uv run --locked puns-llm --action evaluate-official \
  --model gemma3:27b --protocol one_shot
```

To run all eight conditions:

```bash
bash scripts/run_official.sh
```

New inference requires access to the corresponding configured model versions. Existing recorded results are preserved and can be evaluated independently of Ollama.

## Experimental environment

The following specifications were observed in the environment inspected for this project (15 October 2026):

| Component | Specification |
| --- | --- |
| OS / kernel | Ubuntu 26.04 LTS / `7.0.0-31-generic` |
| CPU | AMD Ryzen 9 9900X (12 cores / 24 threads) |
| RAM / swap | 125 GiB / 8 GiB |
| GPU | NVIDIA GeForce RTX 5090 (32,607 MiB VRAM) |
| NVIDIA driver | 595.84 |
| Driver-reported CUDA compatibility | 13.2 |
| Ollama server (HTTP API) | 0.40.3 |

The CUDA version above is reported by the NVIDIA driver, not a separately verified CUDA toolkit version. The Ollama API was accessible from the container, although the `ollama` executable was not installed there.

## Dataset citation

The dataset was created by **Inácio et al. (2024)**:

> Inacio, M. L., et al. *Puntuguese: A Corpus of Puns in Portuguese with Micro-edits*. Proceedings of LREC-COLING 2024, pp. 13332–13343. [Paper](https://aclanthology.org/2024.lrec-main.1167/) · [Dataset](https://huggingface.co/datasets/Superar/Puntuguese)

Please cite the original corpus when using these data.

## License

The original code in this repository is provided under the [MIT License](LICENSE). Third-party datasets and pretrained models remain subject to their respective terms; this license does not replace those terms.
