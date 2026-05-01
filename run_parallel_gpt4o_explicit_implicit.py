"""
Parallel GPT-4o × GPT-4o Pipeline (Explicit vs Implicit)
=========================================================

Supervisor's Pipeline 3: split the responsibilities of the combined prompt
(Pipeline 2) into two specialised GPT-4o attackers running in parallel:

  Attack A — GPT-4o, explicit-cues prompt
    The original paper's analytical prompt: step-by-step deduction from
    directly stated facts (names, ages, locations, job titles, income
    references, relationship statements).

  Attack B — GPT-4o, implicit-cues prompt
    Sociolinguistic prompt: infer from HOW the person writes — vocabulary
    register, syntax complexity, cultural references, slang, discourse
    patterns, topic selection.

Both attackers use GPT-4o (no Claude). The difference from Pipeline 2 is
that the attacker's workload is split between two specialised models rather
than handled by a single combined prompt.

This is an ablation of Pipeline 2:
  Pipeline 2 : single GPT-4o, one combined prompt covering both surfaces
  Pipeline 3 : two GPT-4o, each responsible for only one signal surface

Usage:
    python run_parallel_gpt4o_explicit_implicit.py

    # Re-generate report only:
    python run_parallel_gpt4o_explicit_implicit.py --report_only
"""

import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.utils.initialization import (
    read_config_from_yaml,
    seed_everything,
    set_credentials,
)
from run_parallel_inference import run_parallel_inference_pipeline, generate_parallel_report
from evaluate_parallel_paper_metrics import run as run_paper_metrics

CONFIG_PATH = "configs/anonymization/parallel_gpt4o_explicit_implicit.yaml"


def _patch_report_labels(out_dir: str) -> None:
    """Fix the HTML report labels to use GPT-4o for both attacks."""
    report_path = f"{out_dir}/paper_metrics_report.html"
    if not os.path.exists(report_path):
        return

    with open(report_path, "r", encoding="utf-8") as f:
        html = f.read()

    replacements = [
        ("Attack B (Claude Sociolinguistic)",
         "Attack B (GPT-4o Sociolinguistic/Implicit)"),
        ("attack_b (Claude)",
         "Attack B (GPT-4o)"),
        ("Paper-Aligned Metrics — Parallel Inference Pipeline",
         "Paper-Aligned Metrics — Parallel GPT-4o × GPT-4o (Explicit vs Implicit)"),
    ]
    for old, new in replacements:
        html = html.replace(old, new)

    banner = (
        '<div style="background:#e8f5e9;border-left:4px solid #388e3c;'
        'padding:14px 18px;margin:16px 0;border-radius:4px;">'
        '<strong>Pipeline 3 — Parallel GPT-4o × GPT-4o.</strong> '
        'Both Attack A and Attack B use GPT-4o. '
        'Attack A uses the original paper\'s explicit-reasoning prompt. '
        'Attack B uses the sociolinguistic implicit-cues prompt. '
        'No Claude is involved.</div>'
    )
    html = html.replace("<body>", f"<body>\n{banner}", 1)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Patched report labels → {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run parallel GPT-4o × GPT-4o pipeline (explicit vs implicit)"
    )
    parser.add_argument("--config_path", type=str, default=CONFIG_PATH)
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="Only (re-)generate reports from existing results",
    )
    args = parser.parse_args()

    cfg = read_config_from_yaml(args.config_path)
    seed_everything(cfg.seed)
    set_credentials(cfg)

    out_dir = cfg.task_config.outpath

    if args.report_only:
        run_paper_metrics(out_dir)
        _patch_report_labels(out_dir)
        generate_parallel_report(out_dir)
    else:
        run_parallel_inference_pipeline(cfg, num_rounds=2)
        generate_parallel_report(out_dir)
        _patch_report_labels(out_dir)
