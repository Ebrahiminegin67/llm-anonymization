# LLM-Based Adversarial Text Anonymization
### Decomposing Adversarial Attacks via Explicit and Implicit Signal Reasoning

**Master's Thesis** — Negin Ebrahimi  
Stockholm University, Department of Computer and Systems Sciences  
Supervised by Zara Karazian and Amir Payberah

---

## Overview

This repository contains the code and results for a master's thesis that extends the Adversarial Anonymization (AA) framework proposed by Staab et al. (ICLR 2025). The thesis investigates whether decomposing the adversarial attacker into specialized components — one targeting explicit identifying signals and one targeting implicit stylistic signals — improves post-anonymization privacy protection.

Four pipeline architectures are evaluated on 100 synthetic profiles from the SynthPAI dataset using GPT-4o for all roles (attacker, anonymizer, utility judge).

---

## The Four Pipelines

| Pipeline | Description |
|----------|-------------|
| **P1 — Explicit-Only Baseline** | Single attacker using the original Staab et al. prompt targeting explicit signals only |
| **P2 — Combined Single-Prompt** | Single attacker using an enriched prompt targeting both explicit and implicit signals |
| **P3 — Parallel Dual-Attacker** | Two independent attackers running simultaneously — Attack A (explicit) and Attack B (implicit) — results merged before anonymization |
| **P4 — Sequential Dual-Attacker** | Attack A (explicit) runs first; Attack B (implicit) receives Attack A's findings and targets what was missed |

Each pipeline runs for 2 adversarial anonymization rounds on the same 100 SynthPAI profiles.

---

## Key Findings

- All four pipelines converge to post-anonymization Top-3 adversarial accuracy of **0.24–0.27**
- No architecture significantly outperforms the baseline (McNemar's test, all p > 0.05)
- **Explicit signals are suppressed** by phrase-level rewriting (P3 Attack A: 0.42 → 0.33 evidence rate)
- **Implicit signals persist unchanged** (P3 Attack B: 0.46 → 0.46 evidence rate — zero reduction)
- The bottleneck is the anonymizer, not the attacker

---

## Setup

**1. Install the environment:**

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
mamba env create -f environment.yaml
```

**2. Add your API key:**

Copy `credentials_clean.py` to `credentials.py` and add your OpenAI API key.

```bash
cp credentials_clean.py credentials.py
# Edit credentials.py and add your key
```

**3. Download the SynthPAI dataset:**

Download `inference_0.jsonl` from the [SynthPAI repository](https://github.com/eth-sri/SynthPAI) and place it under `data/base_inferences/synthetic/`.

---

## Running the Pipelines

### P1 — Explicit-Only Baseline
```bash
python main.py --config_path configs/anonymization/baseline_single_attack.yaml
```

### P2 — Combined Single-Prompt
```bash
python run_enhanced_baseline.py --config_path configs/anonymization/enhanced_baseline.yaml
```

### P3 — Parallel Dual-Attacker
```bash
python run_parallel_gpt4o_explicit_implicit.py --config_path configs/anonymization/parallel_gpt4o_explicit_implicit.yaml
```

### P4 — Sequential Dual-Attacker
```bash
python run_sequential_gpt4o_explicit_implicit.py --config_path configs/anonymization/sequential_gpt4o_explicit_implicit.yaml
```

---

## Evaluation

Results are stored in `anonymized_results/`. Each pipeline folder contains:
- `inference_0.jsonl` — pre-anonymization attack results
- `inference_1.jsonl`, `inference_2.jsonl` — post-anonymization attack results per round
- `anonymized_0.jsonl`, `anonymized_1.jsonl` — anonymized text outputs
- `utility_0.jsonl`, `utility_1.jsonl` — utility scores per round
- `paper_metrics_report.html` — full metrics report

---

## Repository Structure

```
├── main.py                                  # P1 baseline runner
├── run_enhanced_baseline.py                 # P2 runner
├── run_parallel_gpt4o_explicit_implicit.py  # P3 runner
├── run_sequential_gpt4o_explicit_implicit.py# P4 runner
├── src/
│   ├── anonymized/anonymizers/              # Anonymizer implementations
│   └── reddit/                              # Profile loading and types
├── configs/anonymization/                   # Pipeline configuration files
├── anonymized_results/                      # Output results per pipeline
└── data/base_inferences/synthetic/          # SynthPAI dataset (not included)
```

---

## Citation

If you use this code, please cite the original AA framework:

```bibtex
@inproceedings{staab25lmanon,
    title={Language Models are Advanced Anonymizers},
    author={Robin Staab and Mark Vero and Mislav Balunović and Martin Vechev},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
}
```

And the SynthPAI dataset:

```bibtex
@article{munzel2024synthpai,
    title={SynthPAI: A Synthetic Dataset for Personal Attribute Inference},
    author={Munzel, Nicolas and Staab, Robin and Vechev, Martin},
    year={2024},
    journal={arXiv preprint arXiv:2406.07217}
}
```
