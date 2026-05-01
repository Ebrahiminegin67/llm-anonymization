"""
Sequential GPT-4o × GPT-4o Pipeline (Explicit → Implicit)
==========================================================

Supervisor's Pipeline 4: same sequential architecture as the original
sequential pipeline (GPT-4o + Claude), but both attackers are GPT-4o.

  Attack A — GPT-4o, explicit-cues prompt (runs FIRST, blind)
    The original paper's analytical prompt: step-by-step deduction from
    directly stated facts.

  Attack B — GPT-4o, implicit/sociolinguistic prompt (runs SECOND)
    Receives the original text AND Attack A's full findings.
    Must confirm, challenge, or extend A's conclusions using implicit
    writing style signals only.

Key difference from Parallel GPT-4o × GPT-4o (Pipeline 3):
  Parallel  — A and B are independent; results merged afterward
  Sequential — B sees A's findings before producing its own; B's prompt
               is customised with A's per-type inferences injected

This is a controlled ablation:
  P3 Parallel  : 2× GPT-4o, independent specialised prompts
  P4 Sequential: 2× GPT-4o, B is informed and reactive to A

Usage:
    python run_sequential_gpt4o_explicit_implicit.py

    # Report only (inference already completed):
    python run_sequential_gpt4o_explicit_implicit.py --report_only
"""

import json
import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.utils.initialization import (
    read_config_from_yaml,
    seed_everything,
    set_credentials,
)
from run_sequential_inference import (
    run_sequential_inference_pipeline,
    generate_sequential_report,
)
from evaluate_parallel_paper_metrics import (
    compute_sequential_paper_metrics,
    generate_paper_metrics_report,
)

CONFIG_PATH = "configs/anonymization/sequential_gpt4o_explicit_implicit.yaml"


def _compute_and_save_paper_metrics(out_dir: str) -> None:
    """
    Run paper-aligned metrics on the sequential output and write an HTML report.

    compute_sequential_paper_metrics() returns metrics with key 'accumulated'
    instead of 'merged'.  We remap it so generate_paper_metrics_report() works
    unchanged (it expects 'merged').
    """
    try:
        metrics = compute_sequential_paper_metrics(out_dir)
    except FileNotFoundError as e:
        print(f"  Skipping paper metrics: {e}")
        return

    # Remap 'accumulated' → 'merged' for the shared HTML report generator
    for stage in ["pre_anon", "post_anon"]:
        if "accumulated" in metrics[stage]:
            metrics[stage]["merged"] = metrics[stage].pop("accumulated")
    if "accumulated" in metrics.get("tradeoff", {}):
        metrics["tradeoff"]["merged"] = metrics["tradeoff"].pop("accumulated")

    json_path = f"{out_dir}/paper_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Saved paper metrics JSON → {json_path}")

    report_path = f"{out_dir}/paper_metrics_report.html"
    generate_paper_metrics_report(out_dir, metrics, report_path)

    _patch_report_labels(report_path)


def _patch_report_labels(report_path: str) -> None:
    """Fix hardcoded label strings to reflect GPT-4o × GPT-4o sequential setup."""
    if not os.path.exists(report_path):
        return

    with open(report_path, "r", encoding="utf-8") as f:
        html = f.read()

    replacements = [
        ("Attack B (Claude Sociolinguistic)",
         "Attack B (GPT-4o Sociolinguistic/Implicit, informed by A)"),
        ("Merged",
         "Accumulated (A + B)"),
        ("Paper-Aligned Metrics — Parallel Inference Pipeline",
         "Paper-Aligned Metrics — Sequential GPT-4o × GPT-4o (Explicit → Implicit)"),
    ]
    for old, new in replacements:
        html = html.replace(old, new)

    banner = (
        '<div style="background:#f3e5f5;border-left:4px solid #7b1fa2;'
        'padding:14px 18px;margin:16px 0;border-radius:4px;">'
        '<strong>Pipeline 4 — Sequential GPT-4o × GPT-4o.</strong> '
        'Both attacks use GPT-4o. '
        'Attack A (explicit prompt) runs first on the original text. '
        'Attack B (implicit prompt) then runs on the same text '
        '<em>with Attack A\'s findings injected into its prompt</em> — '
        'it confirms, challenges, or extends A\'s conclusions. '
        'No Claude is involved.</div>'
    )
    html = html.replace("<body>", f"<body>\n{banner}", 1)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Patched report labels → {report_path}")


def _patch_sequential_report(out_dir: str) -> None:
    """Patch the sequential analysis HTML report (not the paper metrics one)."""
    report_path = f"{out_dir}/sequential_inference_report.html"
    if not os.path.exists(report_path):
        return

    with open(report_path, "r", encoding="utf-8") as f:
        html = f.read()

    replacements = [
        ("Attack B (Claude sociolinguistic)",
         "Attack B (GPT-4o Sociolinguistic, informed by A)"),
        ("Attack B (Claude)",
         "Attack B (GPT-4o)"),
        ("Claude sociolinguistic",
         "GPT-4o Sociolinguistic"),
    ]
    for old, new in replacements:
        html = html.replace(old, new)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run sequential GPT-4o × GPT-4o pipeline (explicit → implicit)"
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
        generate_sequential_report(out_dir)
        _patch_sequential_report(out_dir)
        _compute_and_save_paper_metrics(out_dir)
    else:
        run_sequential_inference_pipeline(cfg, num_rounds=2)
        generate_sequential_report(out_dir)
        _patch_sequential_report(out_dir)
        _compute_and_save_paper_metrics(out_dir)
