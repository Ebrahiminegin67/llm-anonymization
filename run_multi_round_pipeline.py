"""
Multi-Round Adversarial Loop Pipeline
======================================

Iteratively attacks and anonymizes text across N rounds. In each round:
  1. Run parallel attack (GPT-4o analytical + Claude sociolinguistic) on current text
  2. Anonymize using the merged attack findings
  3. Score utility
  4. Re-attack the anonymized text and measure accuracy
  5. If accuracy is still above the stop threshold, carry forward into round N+1

The hypothesis: repeated attack-anonymize cycles should progressively strip more
demographic signal, driving adversarial accuracy below the 55% floor observed with
single-round pipelines. Each round the attacker has less evidence to work with, so
the anonymizer targets increasingly subtle signals.

Usage:
  python run_multi_round_pipeline.py \\
      --config_path configs/anonymization/multi_round.yaml \\
      --max_rounds 5 \\
      --stop_threshold 0.3

  # Only regenerate the summary report from existing round data:
  python run_multi_round_pipeline.py \\
      --config_path configs/anonymization/multi_round.yaml \\
      --report_only
"""

import json
import os
import sys
import argparse
from copy import deepcopy
from typing import Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.utils.initialization import read_config_from_yaml, seed_everything, set_credentials
from src.configs import AnonymizationConfig, Config
from src.models.model_factory import get_model
from src.reddit.reddit_types import Profile
from src.anonymized.anonymized import anonymize, score_utility, load_profiles
from src.anonymized.anonymizers.anonymizer_factory import get_anonymizer

from run_parallel_inference import (
    run_parallel_inference,
    create_prompts_analytical,
    create_prompts_creative,
    _parse_certainty,
)
from evaluate_parallel_paper_metrics import (
    compute_adversarial_accuracy,
    compute_evidence_rate,
    extract_utility_scores,
    aggregate_utility,
)


# ══════════════════════════════════════════════════════════════════════════════
# Per-round metrics helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_attack_results_for_metrics(results: Dict) -> Dict:
    """Convert parallel pipeline results dict to the flat format expected by
    compute_adversarial_accuracy / compute_evidence_rate (uses 'merged' view)."""
    flat = {}
    for username, r in results.items():
        merged = r.get("merged", {})
        flat[username] = {k: v for k, v in merged.items() if k != "full_answer"}
    return flat


def compute_round_metrics(
    results_pre: Dict,
    results_post: Dict,
    profiles: List[Profile],
    utility_jsonl_path: str,
) -> Dict:
    """Compute paper-aligned metrics for one round."""
    flat_pre  = _build_attack_results_for_metrics(results_pre)
    flat_post = _build_attack_results_for_metrics(results_post)

    acc_pre  = compute_adversarial_accuracy(flat_pre,  profiles)
    acc_post = compute_adversarial_accuracy(flat_post, profiles)
    ev_pre   = compute_evidence_rate(flat_pre,  profiles)
    ev_post  = compute_evidence_rate(flat_post, profiles)

    util_agg = {"avg_combined": None}
    if os.path.exists(utility_jsonl_path):
        with open(utility_jsonl_path) as f:
            utility_data = [json.loads(line) for line in f if line.strip()]
        utility_scores = extract_utility_scores(utility_data)
        util_agg = aggregate_utility(utility_scores)

    return {
        "pre_anon": {
            "accuracy": acc_pre,
            "certainty": ev_pre,
        },
        "post_anon": {
            "accuracy": acc_post,
            "certainty": ev_post,
        },
        "utility": util_agg,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HTML report
# ══════════════════════════════════════════════════════════════════════════════

def generate_multi_round_report(round_metrics: List[Dict], out_dir: str) -> None:
    """Generate an HTML report showing how metrics evolve across rounds."""

    def _pct(v):
        if v is None: return "N/A"
        try: return f"{float(v)*100:.1f}%"
        except: return str(v)

    def _f(v):
        if v is None: return "N/A"
        try: return f"{float(v):.3f}"
        except: return str(v)

    # Build table rows
    table_rows = ""
    for i, rm in enumerate(round_metrics):
        pre_acc  = rm["pre_anon"]["accuracy"].get("overall_top3")
        post_acc = rm["post_anon"]["accuracy"].get("overall_top3")
        pre_ev   = rm["pre_anon"]["certainty"].get("overall_evidence_rate")
        post_ev  = rm["post_anon"]["certainty"].get("overall_evidence_rate")
        utility  = rm["utility"].get("avg_combined")

        # Colour post-anon accuracy: green if lower than round 1
        baseline_acc = round_metrics[0]["post_anon"]["accuracy"].get("overall_top3")
        if post_acc is not None and baseline_acc is not None and i > 0:
            delta = post_acc - baseline_acc
            if delta < -0.04:
                acc_colour = "color:#2e7d32;font-weight:bold"
            elif delta < 0:
                acc_colour = "color:#f9a825;font-weight:bold"
            else:
                acc_colour = "color:#888"
            delta_str = f"({delta*100:+.1f}pp)"
        else:
            acc_colour = "font-weight:bold"
            delta_str = "(baseline)"

        table_rows += f"""
        <tr>
          <td><strong>Round {i+1}</strong></td>
          <td>{_pct(pre_acc)}</td>
          <td style="{acc_colour}">{_pct(post_acc)} {delta_str}</td>
          <td>{_pct(pre_ev)}</td>
          <td>{_pct(post_ev)}</td>
          <td>{_f(utility)}</td>
        </tr>"""

    # Per-PII-type breakdown across rounds
    pii_types = set()
    for rm in round_metrics:
        per_type = rm["post_anon"]["accuracy"].get("per_type", {})
        pii_types.update(per_type.keys())

    pii_header = "".join(f"<th>Round {i+1}</th>" for i in range(len(round_metrics)))
    pii_rows = ""
    for pii_type in sorted(pii_types):
        cells = ""
        for rm in round_metrics:
            v = rm["post_anon"]["accuracy"].get("per_type", {}).get(pii_type, {}).get("top3")
            cells += f"<td>{_pct(v)}</td>"
        pii_rows += f"<tr><td>{pii_type}</td>{cells}</tr>"

    pii_table = f"""
    <table>
      <thead><tr><th>PII Type</th>{pii_header}</tr></thead>
      <tbody>{pii_rows}</tbody>
    </table>""" if pii_rows else "<p>No per-type data.</p>"

    # Chart data (for a simple SVG sparkline)
    post_accs = [rm["post_anon"]["accuracy"].get("overall_top3") for rm in round_metrics]
    chart_points = ""
    if all(v is not None for v in post_accs):
        n = len(post_accs)
        w, h = 500, 150
        pad = 30
        x_step = (w - 2*pad) / max(n - 1, 1)
        y_min, y_max = 0.0, 1.0
        def _y(v): return h - pad - (v - y_min) / (y_max - y_min) * (h - 2*pad)
        def _x(i): return pad + i * x_step
        pts = " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(post_accs))
        # Reference line at 55% (single-round floor)
        ref_y = _y(0.55)
        chart_points = f"""
        <svg width="{w}" height="{h}" style="border:1px solid #ddd;border-radius:6px;background:#fafafa">
          <!-- grid lines -->
          <line x1="{pad}" y1="{_y(0.3):.1f}" x2="{w-pad}" y2="{_y(0.3):.1f}" stroke="#eee" stroke-dasharray="4"/>
          <line x1="{pad}" y1="{_y(0.4):.1f}" x2="{w-pad}" y2="{_y(0.4):.1f}" stroke="#eee" stroke-dasharray="4"/>
          <line x1="{pad}" y1="{_y(0.5):.1f}" x2="{w-pad}" y2="{_y(0.5):.1f}" stroke="#eee" stroke-dasharray="4"/>
          <line x1="{pad}" y1="{_y(0.6):.1f}" x2="{w-pad}" y2="{_y(0.6):.1f}" stroke="#eee" stroke-dasharray="4"/>
          <!-- single-round floor reference -->
          <line x1="{pad}" y1="{ref_y:.1f}" x2="{w-pad}" y2="{ref_y:.1f}"
                stroke="#e57373" stroke-dasharray="6,3" stroke-width="1.5"/>
          <text x="{w-pad+4}" y="{ref_y+4:.1f}" font-size="10" fill="#e57373">55% floor</text>
          <!-- accuracy line -->
          <polyline points="{pts}" fill="none" stroke="#1976d2" stroke-width="2.5"/>
          {"".join(f'<circle cx="{_x(i):.1f}" cy="{_y(v):.1f}" r="5" fill="#1976d2"/><text x="{_x(i)-10:.1f}" y="{_y(v)-8:.1f}" font-size="11" fill="#1976d2">{v*100:.0f}%</text>' for i, v in enumerate(post_accs))}
          <!-- x labels -->
          {"".join(f'<text x="{_x(i):.1f}" y="{h-6}" text-anchor="middle" font-size="11" fill="#555">R{i+1}</text>' for i in range(n))}
          <!-- y labels -->
          {"".join(f'<text x="{pad-4}" y="{_y(v)+4:.1f}" text-anchor="end" font-size="10" fill="#888">{int(v*100)}%</text>' for v in [0.3, 0.4, 0.5, 0.6])}
        </svg>"""

    # Verdict
    if len(round_metrics) >= 2:
        first_acc = round_metrics[0]["post_anon"]["accuracy"].get("overall_top3", 0)
        last_acc  = round_metrics[-1]["post_anon"]["accuracy"].get("overall_top3", 0)
        improvement = first_acc - last_acc if first_acc and last_acc else 0
        if improvement > 0.1:
            verdict = (f"<b style='color:#2e7d32'>CLEAR IMPROVEMENT</b>: post-anonymization accuracy "
                       f"dropped {improvement*100:.1f}pp over {len(round_metrics)} rounds. "
                       f"Multi-round adversarial looping is effective.")
        elif improvement > 0.02:
            verdict = (f"<b style='color:#f9a825'>MARGINAL IMPROVEMENT</b>: {improvement*100:.1f}pp drop "
                       f"over {len(round_metrics)} rounds — trend in right direction but within "
                       f"noise range for n=20.")
        elif improvement >= 0:
            verdict = (f"<b style='color:#888'>NO IMPROVEMENT</b>: accuracy unchanged after "
                       f"{len(round_metrics)} rounds. The 55% floor persists — writing style "
                       f"encodes demographics that repeated anonymization cannot remove.")
        else:
            verdict = (f"<b style='color:#c62828'>REGRESSION</b>: accuracy increased by "
                       f"{-improvement*100:.1f}pp. Over-anonymization may be degrading text "
                       f"in ways that make the attacker more confident.")
    else:
        verdict = "Only one round completed — run more rounds to evaluate the trend."

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Multi-Round Adversarial Loop: Privacy Metrics by Round</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1000px; margin: 0 auto; padding: 20px; color: #212121; }}
  h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }}
  h2 {{ color: #283593; margin-top: 32px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 10px 14px; text-align: center; }}
  th {{ background: #e8eaf6; color: #1a237e; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  .verdict {{ background: #e8f5e9; border-left: 4px solid #388e3c;
              padding: 16px; border-radius: 4px; margin: 20px 0; }}
  .method-box {{ background: #e3f2fd; border-radius: 6px; padding: 14px;
                 margin: 12px 0; border-left: 4px solid #1976d2; }}
</style>
</head>
<body>
<h1>Multi-Round Adversarial Loop: Privacy Metrics by Round</h1>

<div class="method-box">
  <strong>How it works:</strong> each round runs a fresh parallel attack on the current
  anonymized text, then anonymizes again using those new findings. If writing style
  encodes demographic signals that survive one round of anonymization, repeated rounds
  should progressively strip them — or confirm that the floor is truly irreducible.
</div>

<h2>Post-Anonymization Adversarial Accuracy by Round</h2>
{chart_points if chart_points else "<p>Not enough data for chart.</p>"}

<h2>Full Metrics Table</h2>
<table>
  <thead>
    <tr>
      <th>Round</th>
      <th>Pre-anon Accuracy</th>
      <th>Post-anon Accuracy</th>
      <th>Pre-anon Evidence Rate</th>
      <th>Post-anon Evidence Rate</th>
      <th>Combined Utility</th>
    </tr>
  </thead>
  <tbody>{table_rows}</tbody>
</table>

<h2>Per-PII-Type Post-anon Accuracy by Round</h2>
{pii_table}

<div class="verdict">
  <strong>Verdict:</strong> {verdict}
  <p style="margin-top:8px;font-size:.9em;color:#555">
    Note: n=20 profiles. A 10pp accuracy change requires ~100 profiles for statistical significance.
    Treat these as directional trends.
  </p>
</div>

<hr>
<p style="color:#888;font-size:.85em">
  Metrics: Adversarial Accuracy (Top-3) = fraction of ground-truth PII correctly guessed;
  Evidence Rate = fraction with certainty &ge; 3;
  Combined Utility = mean(Readability/10, Meaning/10, ROUGE-1 F1).
</p>
</body>
</html>"""

    report_path = f"{out_dir}/multi_round_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote report -> {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_multi_round_pipeline(cfg: Config, max_rounds: int = 5, stop_threshold: float = 0.25) -> None:
    """
    Iterative attack-anonymize loop.

    Each round:
      1. Parallel attack on current text (A + B + merge)
      2. Anonymize with merged findings
      3. Utility score
      4. Re-attack anonymized text → measure accuracy
      5. If accuracy <= stop_threshold, stop early
      6. Otherwise, carry the re-attack findings forward as input for round N+1
    """
    assert isinstance(cfg.task_config, AnonymizationConfig)

    model_a    = get_model(cfg.task_config.inference_model)
    model_b    = get_model(cfg.task_config.eval_inference_model)
    util_model = get_model(cfg.task_config.utility_model)
    anonymizer = get_anonymizer(cfg.task_config)

    profiles = load_profiles(cfg.task_config)
    out_dir  = cfg.task_config.outpath
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLoaded {len(profiles)} profiles")
    print(f"Output directory: {out_dir}")
    print(f"Max rounds: {max_rounds}  |  Stop threshold: {stop_threshold*100:.0f}% accuracy")
    print(f"Attack models: {model_a.config.name} (analytical) + {model_b.config.name} (sociolinguistic)")

    round_metrics = []
    inference_model_name = cfg.task_config.inference_model.name

    for round_num in range(max_rounds):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num + 1} / {max_rounds}")
        print(f"{'='*60}")

        # ── Step 1: Attack current text ────────────────────────────────────
        print(f"\n[Round {round_num+1}] Parallel attack on {'original' if round_num == 0 else 'anonymized'} text...")
        results_pre = run_parallel_inference(
            profiles, model_a, model_b, cfg,
            prompt_strategy_a=create_prompts_analytical,
            prompt_strategy_b=create_prompts_creative,
        )

        # Store merged inference so the anonymizer can read it
        for profile in profiles:
            merged = results_pre[profile.username]["merged"]
            profile.get_latest_comments().predictions[inference_model_name] = merged
            profile.get_latest_comments().predictions[inference_model_name]["full_answer"] = "PARALLEL_MERGED"

        with open(f"{out_dir}/attacks_pre_round_{round_num+1}.json", "w") as f:
            json.dump(results_pre, f, indent=2, default=str)

        for profile in profiles:
            with open(f"{out_dir}/inference_pre_{round_num+1}.jsonl", "a") as f:
                f.write(json.dumps(profile.to_json()) + "\n")

        # ── Step 2: Anonymize ──────────────────────────────────────────────
        print(f"\n[Round {round_num+1}] Anonymizing...")
        anonymize(profiles, anonymizer, cfg)

        # ── Step 3: Utility scoring ────────────────────────────────────────
        print(f"\n[Round {round_num+1}] Scoring utility...")
        score_utility(profiles, util_model, cfg)

        utility_path = f"{out_dir}/utility_{round_num}.jsonl"

        # ── Step 4: Re-attack anonymized text ─────────────────────────────
        print(f"\n[Round {round_num+1}] Re-attacking anonymized text...")
        results_post = run_parallel_inference(
            profiles, model_a, model_b, cfg,
            prompt_strategy_a=create_prompts_analytical,
            prompt_strategy_b=create_prompts_creative,
        )

        with open(f"{out_dir}/attacks_post_round_{round_num+1}.json", "w") as f:
            json.dump(results_post, f, indent=2, default=str)

        for profile in profiles:
            with open(f"{out_dir}/inference_post_{round_num+1}.jsonl", "a") as f:
                f.write(json.dumps(profile.to_json()) + "\n")

        # ── Step 5: Compute metrics ────────────────────────────────────────
        rm = compute_round_metrics(results_pre, results_post, profiles, utility_path)
        rm["round"] = round_num + 1
        round_metrics.append(rm)

        post_acc = rm["post_anon"]["accuracy"].get("overall_top3", 1.0)
        post_ev  = rm["post_anon"]["certainty"].get("overall_evidence_rate", 1.0)
        utility  = rm["utility"].get("avg_combined")

        print(f"\n[Round {round_num+1}] Results:")
        print(f"  Pre-anon  accuracy:      {rm['pre_anon']['accuracy'].get('overall_top3', 0)*100:.1f}%")
        print(f"  Post-anon accuracy:      {post_acc*100:.1f}%")
        print(f"  Post-anon evidence rate: {post_ev*100:.1f}%")
        print(f"  Combined utility:        {utility:.3f}" if utility else "  Combined utility: N/A")

        # ── Step 6: Early stopping ─────────────────────────────────────────
        if post_acc <= stop_threshold:
            print(f"\n[Round {round_num+1}] Early stop: accuracy {post_acc*100:.1f}% <= threshold {stop_threshold*100:.0f}%")
            break

        if round_num < max_rounds - 1:
            # Carry post-attack findings into the next round's anonymizer input.
            # After anonymize(), profile.get_latest_comments() is the NEW anonymized
            # AnnotatedComments with empty predictions — store re-attack results there.
            print(f"\n[Round {round_num+1}] Carrying attack findings forward to round {round_num+2}...")
            for profile in profiles:
                merged = results_post[profile.username]["merged"]
                profile.get_latest_comments().predictions[inference_model_name] = merged
                profile.get_latest_comments().predictions[inference_model_name]["full_answer"] = "PARALLEL_MERGED"

    # ── Save summary and report ────────────────────────────────────────────
    summary = {
        "max_rounds": max_rounds,
        "stop_threshold": stop_threshold,
        "rounds_completed": len(round_metrics),
        "rounds": round_metrics,
    }
    with open(f"{out_dir}/multi_round_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    generate_multi_round_report(round_metrics, out_dir)

    # Print terminal summary
    print("\n" + "=" * 60)
    print("MULTI-ROUND SUMMARY")
    print("=" * 60)
    print(f"{'Round':<8} {'Pre-acc':>10} {'Post-acc':>10} {'Evid-Rate':>12} {'Utility':>10}")
    print("-" * 60)
    for rm in round_metrics:
        r   = rm["round"]
        pre = rm["pre_anon"]["accuracy"].get("overall_top3")
        pst = rm["post_anon"]["accuracy"].get("overall_top3")
        ev  = rm["post_anon"]["certainty"].get("overall_evidence_rate")
        ut  = rm["utility"].get("avg_combined")
        def _p(v): return f"{v*100:.1f}%" if v is not None else "N/A"
        def _f(v): return f"{v:.3f}" if v is not None else "N/A"
        print(f"{r:<8} {_p(pre):>10} {_p(pst):>10} {_p(ev):>12} {_f(ut):>10}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Report-only mode: regenerate HTML from saved round data
# ══════════════════════════════════════════════════════════════════════════════

def report_only(out_dir: str) -> None:
    summary_path = f"{out_dir}/multi_round_summary.json"
    if not os.path.exists(summary_path):
        print(f"No multi_round_summary.json found in {out_dir}")
        print("Run the full pipeline first.")
        return
    with open(summary_path) as f:
        summary = json.load(f)
    generate_multi_round_report(summary["rounds"], out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Round Adversarial Loop Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument("--max_rounds", type=int, default=5,
                        help="Maximum number of attack-anonymize rounds (default: 5)")
    parser.add_argument("--stop_threshold", type=float, default=0.25,
                        help="Stop early if post-anon accuracy drops below this (default: 0.25)")
    parser.add_argument("--report_only", action="store_true",
                        help="Regenerate HTML report from existing round data without re-running")
    args = parser.parse_args()

    cfg = read_config_from_yaml(args.config_path)
    seed_everything(cfg.seed)
    set_credentials(cfg)

    if args.report_only:
        report_only(cfg.task_config.outpath)
        return

    run_multi_round_pipeline(cfg, max_rounds=args.max_rounds, stop_threshold=args.stop_threshold)


if __name__ == "__main__":
    main()
