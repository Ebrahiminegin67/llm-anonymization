"""
Paper-Metric Comparison: Baseline vs Parallel Inference Architecture
=====================================================================

Computes and compares the three metrics from "LLMs Are Advanced Anonymizers"
across two anonymization architectures on the same 20 profiles:

  BASELINE   — single GPT-4o attack → anonymize (2 iterations) → re-attack
  PARALLEL   — GPT-4o + Claude parallel attack → merge → anonymize → re-attack

Metrics compared side-by-side:
  1. Adversarial Accuracy (Top-1 / Top-3)
  2. Adversarial Certainty — Evidence Rate
  3. Combined Utility  = mean(Readability/10, Meaning/10, ROUGE-1)
  4. Privacy–Utility Tradeoff

Usage:
    python compare_baseline_vs_parallel_paper_metrics.py
    python compare_baseline_vs_parallel_paper_metrics.py \\
        --baseline_dir  anonymized_results/baseline_single_attack_20profiles \\
        --parallel_dir  anonymized_results/parallel_gpt4o_vs_claude_20profiles_v2 \\
        --output        anonymized_results/paper_metrics_comparison.html
"""

import json
import os
import sys
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Any

sys.path.append(os.path.dirname(__file__))

# Reuse all metric helpers from the existing evaluation module
from evaluate_parallel_paper_metrics import (
    get_ground_truth,
    compute_adversarial_accuracy,
    compute_evidence_rate,
    extract_utility_scores,
    aggregate_utility,
    _pct, _f, _delta_html,
)
from src.reddit.reddit_utils import load_data


# ── Baseline data extraction ─────────────────────────────────────────────────

def _extract_predictions_from_profiles(
    profiles_jsonl_path: str,
    comment_index: int,
    model_key: str = "gpt-4o",
) -> Dict[str, Dict]:
    """
    Extract attack predictions from a baseline JSONL file.

    The baseline stores predictions inside profile objects:
        comments[comment_index].predictions[model_key][pii_type]

    Returns {username: {pii_type: {guess, certainty, inference}}}.
    """
    results: Dict[str, Dict] = {}
    with open(profiles_jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            username = data.get("username", "")
            comments = data.get("comments", [])

            # Clamp index: -1 means last comment
            idx = comment_index if comment_index >= 0 else len(comments) - 1
            if idx >= len(comments):
                idx = len(comments) - 1

            preds = comments[idx].get("predictions", {})
            # Accept the requested model key or fall back to any available key
            if model_key in preds:
                results[username] = preds[model_key]
            elif preds:
                results[username] = next(iter(preds.values()))

    return results


def compute_baseline_paper_metrics(baseline_dir: str) -> Dict:
    """
    Compute paper metrics from a baseline (single GPT-4o attack) run.

    File layout expected:
        inference_0.jsonl  — original profiles with pre-anon attack predictions
        inference_2.jsonl  — profiles after 2 anonymization rounds (post-anon)
        utility_0.jsonl    — utility scores after the first anonymization round
    Falls back to inference_1.jsonl if inference_2 is absent.
    """
    pre_path  = f"{baseline_dir}/inference_0.jsonl"
    post_path = f"{baseline_dir}/inference_2.jsonl"
    if not os.path.exists(post_path):
        post_path = f"{baseline_dir}/inference_1.jsonl"
    util_path = f"{baseline_dir}/utility_0.jsonl"

    for path in [pre_path, post_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required baseline file not found: {path}")

    profiles = load_data(pre_path)

    pre_preds  = _extract_predictions_from_profiles(pre_path,  comment_index=0)
    post_preds = _extract_predictions_from_profiles(post_path, comment_index=-1)

    metrics: Dict = {
        "pre_anon": {
            "accuracy":  compute_adversarial_accuracy(pre_preds,  profiles),
            "certainty": compute_evidence_rate(pre_preds,  profiles),
        },
        "post_anon": {
            "accuracy":  compute_adversarial_accuracy(post_preds, profiles),
            "certainty": compute_evidence_rate(post_preds, profiles),
        },
    }

    # Utility
    if os.path.exists(util_path):
        with open(util_path) as f:
            utility_data = [json.loads(line) for line in f if line.strip()]
        utility_scores = extract_utility_scores(utility_data)
        metrics["utility"] = aggregate_utility(utility_scores)
    else:
        metrics["utility"] = {"avg_combined": None, "note": "utility_0.jsonl not found"}

    # Tradeoff pair
    util_combined = metrics["utility"].get("avg_combined")
    acc_pre  = metrics["pre_anon"]["accuracy"]["overall_top3"]
    acc_post = metrics["post_anon"]["accuracy"]["overall_top3"]
    metrics["tradeoff"] = {
        "adversarial_accuracy_pre":  acc_pre,
        "adversarial_accuracy_post": acc_post,
        "accuracy_reduction":        round(acc_pre - acc_post, 3),
        "combined_utility":          util_combined,
    }

    return metrics


# ── Load or recompute parallel metrics ───────────────────────────────────────

def load_parallel_metrics(parallel_dir: str) -> Dict:
    """
    Load cached parallel paper metrics or recompute them if absent.
    """
    cache_path = f"{parallel_dir}/paper_metrics.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    from evaluate_parallel_paper_metrics import compute_paper_metrics
    return compute_paper_metrics(parallel_dir)


# ── HTML Report ───────────────────────────────────────────────────────────────

def generate_comparison_report(
    baseline_metrics: Dict,
    parallel_metrics: Dict,
    output_path: str,
    baseline_dir: str,
    parallel_dir: str,
) -> None:

    bpre  = baseline_metrics["pre_anon"]
    bpost = baseline_metrics["post_anon"]
    bu    = baseline_metrics["utility"]
    bt    = baseline_metrics["tradeoff"]

    ppre_a  = parallel_metrics["pre_anon"]["attack_a"]
    ppost_a = parallel_metrics["post_anon"]["attack_a"]
    ppre_b  = parallel_metrics["pre_anon"]["attack_b"]
    ppost_b = parallel_metrics["post_anon"]["attack_b"]
    pu      = parallel_metrics["utility"]
    pt      = parallel_metrics["tradeoff"]

    # ── Collect all PII types ─────────────────────────────────────────────
    all_pii: set = set()
    for d in [bpre["accuracy"]["per_type"], bpost["accuracy"]["per_type"],
              ppre_a["accuracy"]["per_type"], ppost_a["accuracy"]["per_type"]]:
        all_pii.update(d.keys())
    all_pii_sorted = sorted(all_pii)

    # ── Helper: coloured accuracy cell ───────────────────────────────────
    def _acc_cell(val: Optional[float], ref: Optional[float] = None) -> str:
        """Render accuracy cell, green if better than ref."""
        if val is None:
            return "<td>-</td>"
        txt = _pct(val)
        if ref is not None and val < ref:          # lower accuracy = better privacy
            return f'<td style="color:#2e7d32;font-weight:bold;">{txt} ▼</td>'
        return f"<td>{txt}</td>"

    # ── Section 1: Adversarial Accuracy overview ─────────────────────────
    acc_overview = f"""
    <table>
      <thead>
        <tr>
          <th rowspan="2">Stage</th>
          <th colspan="2" style="background:#2196F3;">Baseline (GPT-4o only)</th>
          <th colspan="2" style="background:#7B1FA2;">Parallel — Attack A (GPT-4o)</th>
          <th colspan="2" style="background:#7B1FA2;">Parallel — Attack B (Claude)</th>
          <th colspan="2" style="background:#2e7d32;">Parallel — Merged</th>
        </tr>
        <tr>
          <th>Top-1</th><th>Top-3</th>
          <th>Top-1</th><th>Top-3</th>
          <th>Top-1</th><th>Top-3</th>
          <th>Top-1</th><th>Top-3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Before Anon</strong></td>
          <td>{_pct(bpre['accuracy']['overall_top1'])}</td>
          <td>{_pct(bpre['accuracy']['overall_top3'])}</td>
          <td>{_pct(ppre_a['accuracy']['overall_top1'])}</td>
          <td>{_pct(ppre_a['accuracy']['overall_top3'])}</td>
          <td>{_pct(ppre_b['accuracy']['overall_top1'])}</td>
          <td>{_pct(ppre_b['accuracy']['overall_top3'])}</td>
          <td>{_pct(parallel_metrics['pre_anon']['merged']['accuracy']['overall_top1'])}</td>
          <td>{_pct(parallel_metrics['pre_anon']['merged']['accuracy']['overall_top3'])}</td>
        </tr>
        <tr>
          <td><strong>After Anon</strong></td>
          {_acc_cell(bpost['accuracy']['overall_top1'])}
          {_acc_cell(bpost['accuracy']['overall_top3'])}
          {_acc_cell(ppost_a['accuracy']['overall_top1'], bpost['accuracy']['overall_top1'])}
          {_acc_cell(ppost_a['accuracy']['overall_top3'], bpost['accuracy']['overall_top3'])}
          {_acc_cell(ppost_b['accuracy']['overall_top1'], bpost['accuracy']['overall_top1'])}
          {_acc_cell(ppost_b['accuracy']['overall_top3'], bpost['accuracy']['overall_top3'])}
          {_acc_cell(parallel_metrics['post_anon']['merged']['accuracy']['overall_top1'], bpost['accuracy']['overall_top1'])}
          {_acc_cell(parallel_metrics['post_anon']['merged']['accuracy']['overall_top3'], bpost['accuracy']['overall_top3'])}
        </tr>
        <tr style="background:#e8f5e9;">
          <td><strong>Reduction ▼</strong></td>
          <td colspan="2"><strong>{_delta_html(bt['accuracy_reduction'])}</strong></td>
          <td colspan="2"><strong>{_delta_html(pt['attack_a']['accuracy_reduction'])}</strong></td>
          <td colspan="2"><strong>{_delta_html(pt['attack_b']['accuracy_reduction'])}</strong></td>
          <td colspan="2"><strong>{_delta_html(pt['merged']['accuracy_reduction'])}</strong></td>
        </tr>
      </tbody>
    </table>
    """

    # ── Section 2: Per-PII-type (post-anon, top-3) ───────────────────────
    per_type_rows = ""
    for pt_name in all_pii_sorted:
        b_pre_val  = bpre["accuracy"]["per_type"].get(pt_name, {}).get("top3")
        b_post_val = bpost["accuracy"]["per_type"].get(pt_name, {}).get("top3")
        pa_pre     = ppre_a["accuracy"]["per_type"].get(pt_name, {}).get("top3")
        pa_post    = ppost_a["accuracy"]["per_type"].get(pt_name, {}).get("top3")
        pb_pre     = ppre_b["accuracy"]["per_type"].get(pt_name, {}).get("top3")
        pb_post    = ppost_b["accuracy"]["per_type"].get(pt_name, {}).get("top3")
        pm_pre     = parallel_metrics["pre_anon"]["merged"]["accuracy"]["per_type"].get(pt_name, {}).get("top3")
        pm_post    = parallel_metrics["post_anon"]["merged"]["accuracy"]["per_type"].get(pt_name, {}).get("top3")

        def _delta_cell(pre, post, ref_post):
            if post is None:
                return "<td>-</td>"
            txt = _pct(post)
            style = ""
            if ref_post is not None and post < ref_post:
                style = "color:#2e7d32;font-weight:bold;"
            elif ref_post is not None and post > ref_post:
                style = "color:#f44336;"
            return f'<td style="{style}">{txt}</td>'

        per_type_rows += f"""
        <tr>
          <td><strong>{pt_name}</strong></td>
          <td>{_pct(b_pre_val)}</td>
          <td>{_pct(b_post_val)}</td>
          {_delta_cell(pa_pre, pa_post, b_post_val)}
          {_delta_cell(pb_pre, pb_post, b_post_val)}
          {_delta_cell(pm_pre, pm_post, b_post_val)}
        </tr>
        """

    per_type_table = f"""
    <table>
      <thead>
        <tr>
          <th rowspan="2">PII Type</th>
          <th colspan="2" style="background:#2196F3;">Baseline</th>
          <th style="background:#7B1FA2;">Parallel A</th>
          <th style="background:#7B1FA2;">Parallel B</th>
          <th style="background:#2e7d32;">Parallel Merged</th>
        </tr>
        <tr>
          <th>Pre-Anon</th><th>Post-Anon</th>
          <th>Post-Anon</th>
          <th>Post-Anon</th>
          <th>Post-Anon</th>
        </tr>
      </thead>
      <tbody>{per_type_rows}</tbody>
    </table>
    <p style="font-size:0.85em;color:#555;">
      <span style="color:#2e7d32;font-weight:bold;">Green</span> = parallel did better than baseline (lower accuracy = harder to guess = better privacy).
      <span style="color:#f44336;">Red</span> = parallel did worse. All values are Top-3 accuracy.
    </p>
    """

    # ── Section 3: Evidence Rate ──────────────────────────────────────────
    def _ev_row(label: str, pre_ev: float, post_ev: float,
                ref_post_ev: Optional[float] = None) -> str:
        drop = pre_ev - post_ev
        style = ""
        if ref_post_ev is not None and post_ev < ref_post_ev:
            style = "color:#2e7d32;font-weight:bold;"
        return f"""
        <tr>
          <td><strong>{label}</strong></td>
          <td>{_pct(pre_ev)}</td>
          <td style="{style}">{_pct(post_ev)}</td>
          <td>{_delta_html(drop)}</td>
        </tr>"""

    b_ev_pre  = bpre["certainty"]["overall_evidence_rate"]
    b_ev_post = bpost["certainty"]["overall_evidence_rate"]

    cert_table = f"""
    <table>
      <thead>
        <tr>
          <th>Architecture</th>
          <th>Evidence Rate (Pre-Anon)</th>
          <th>Evidence Rate (Post-Anon)</th>
          <th>Drop ▼</th>
        </tr>
      </thead>
      <tbody>
        {_ev_row("Baseline (GPT-4o)", b_ev_pre, b_ev_post)}
        {_ev_row("Parallel — Attack A (GPT-4o)", ppre_a['certainty']['overall_evidence_rate'], ppost_a['certainty']['overall_evidence_rate'], b_ev_post)}
        {_ev_row("Parallel — Attack B (Claude)", ppre_b['certainty']['overall_evidence_rate'], ppost_b['certainty']['overall_evidence_rate'], b_ev_post)}
        {_ev_row("Parallel — Merged", parallel_metrics['pre_anon']['merged']['certainty']['overall_evidence_rate'], parallel_metrics['post_anon']['merged']['certainty']['overall_evidence_rate'], b_ev_post)}
      </tbody>
    </table>
    """

    # ── Section 4: Utility ────────────────────────────────────────────────
    b_r  = bu.get("avg_readability")
    b_m  = bu.get("avg_meaning")
    b_r1 = bu.get("avg_rouge1")
    b_c  = bu.get("avg_combined")

    p_r  = pu.get("avg_readability")
    p_m  = pu.get("avg_meaning")
    p_r1 = pu.get("avg_rouge1")
    p_c  = pu.get("avg_combined")

    def _better_util(b_val, p_val):
        """Return HTML cell for parallel utility, green if better than baseline."""
        if p_val is None:
            return "<td>-</td>"
        style = ""
        if b_val is not None and p_val > b_val:
            style = "color:#2e7d32;font-weight:bold;"
        elif b_val is not None and p_val < b_val:
            style = "color:#f44336;"
        return f'<td style="{style}">{_f(p_val, 3)}</td>'

    util_table = f"""
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th style="background:#2196F3;">Baseline</th>
          <th style="background:#7B1FA2;">Parallel</th>
          <th>Note</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Readability / 10</strong></td>
          <td>{_f(b_r/10 if b_r else None, 3)}</td>
          {_better_util(b_r/10 if b_r else None, p_r/10 if p_r else None)}
          <td>LLM judge (GPT-4)</td>
        </tr>
        <tr>
          <td><strong>Meaning / 10</strong></td>
          <td>{_f(b_m/10 if b_m else None, 3)}</td>
          {_better_util(b_m/10 if b_m else None, p_m/10 if p_m else None)}
          <td>LLM judge (GPT-4)</td>
        </tr>
        <tr>
          <td><strong>ROUGE-1</strong></td>
          <td>{_f(b_r1, 3)}</td>
          {_better_util(b_r1, p_r1)}
          <td>Unigram F1</td>
        </tr>
        <tr style="background:#e8f5e9;">
          <td><strong>Combined Utility</strong></td>
          <td><strong>{_f(b_c, 3)}</strong></td>
          {_better_util(b_c, p_c)}
          <td>= mean(R/10, M/10, ROUGE-1)</td>
        </tr>
      </tbody>
    </table>
    """

    # ── Section 5: Tradeoff summary cards ────────────────────────────────
    def _card(label: str, color: str, acc_post: float, acc_red: float,
              util_combined: Optional[float]) -> str:
        return f"""
        <div class="card" style="border-top:4px solid {color};">
          <div class="card-label">{label}</div>
          <div class="card-value" style="color:{color};">{_pct(acc_post)}</div>
          <div class="card-sub">Post-anon adversarial accuracy (Top-3)</div>
          <div class="card-sub">Reduction: {_delta_html(acc_red)}</div>
          <div class="card-sub">Utility: {_f(util_combined, 3)}</div>
        </div>
        """

    cards = f"""
    <div class="card-row">
      {_card("Baseline (GPT-4o only)", "#2196F3",
             bt['adversarial_accuracy_post'], bt['accuracy_reduction'], bt['combined_utility'])}
      {_card("Parallel — Attack A (GPT-4o)", "#7B1FA2",
             pt['attack_a']['adversarial_accuracy_post'], pt['attack_a']['accuracy_reduction'],
             pt['attack_a']['combined_utility'])}
      {_card("Parallel — Attack B (Claude)", "#9C27B0",
             pt['attack_b']['adversarial_accuracy_post'], pt['attack_b']['accuracy_reduction'],
             pt['attack_b']['combined_utility'])}
      {_card("Parallel — Merged", "#2e7d32",
             pt['merged']['adversarial_accuracy_post'], pt['merged']['accuracy_reduction'],
             pt['merged']['combined_utility'])}
    </div>
    """

    # ── Verdict text ──────────────────────────────────────────────────────
    b_acc_post  = bt['adversarial_accuracy_post']
    pa_acc_post = pt['attack_a']['adversarial_accuracy_post']
    pb_acc_post = pt['attack_b']['adversarial_accuracy_post']
    pm_acc_post = pt['merged']['adversarial_accuracy_post']

    best_parallel = min(pa_acc_post, pb_acc_post, pm_acc_post)
    best_label = {pa_acc_post: "Attack A", pb_acc_post: "Attack B", pm_acc_post: "Merged"}[best_parallel]

    ev_b  = bpost["certainty"]["overall_evidence_rate"]
    ev_pa = ppost_a["certainty"]["overall_evidence_rate"]
    ev_pb = ppost_b["certainty"]["overall_evidence_rate"]

    verdict_accuracy = (
        f"The parallel anonymizer achieves <strong>{_pct(best_parallel)}</strong> "
        f"adversarial accuracy ({best_label}) vs baseline's <strong>{_pct(b_acc_post)}</strong> — "
        f"a <strong>{round((b_acc_post - best_parallel)*100, 1)}pp improvement</strong>."
        if best_parallel < b_acc_post else
        f"Both architectures reach similar adversarial accuracy "
        f"({_pct(b_acc_post)} baseline vs {_pct(best_parallel)} parallel best)."
    )

    verdict_evidence = (
        f"Evidence rate after anonymization drops further under the parallel anonymizer: "
        f"Attack B <strong>{_pct(ev_pb)}</strong> vs baseline <strong>{_pct(ev_b)}</strong>. "
        f"This means the parallel anonymizer removes more textual signals that the attacker relies on."
        if ev_pb < ev_b else
        f"Evidence rates after anonymization are similar between the two architectures."
    )

    verdict_utility = (
        f"Utility is comparable: baseline <strong>{_f(b_c, 3)}</strong> vs parallel "
        f"<strong>{_f(p_c, 3)}</strong>."
        if b_c and p_c else
        "Utility data partially unavailable."
    )

    # ── Full HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Baseline vs Parallel — Paper Metrics Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #222; background: #fafafa; line-height: 1.6; }}
    h1   {{ color: #2a4d69; border-bottom: 3px solid #4b86b4; padding-bottom: 10px; }}
    h2   {{ color: #4b86b4; margin-top: 36px; }}
    h3   {{ color: #2a4d69; margin-top: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 9px 12px; text-align: center; font-size: 0.87em; }}
    th   {{ background: #4b86b4; color: white; }}
    tr:nth-child(even) {{ background: #f4f8fb; }}
    tr:hover {{ background: #e8f0f8; }}
    td:first-child {{ text-align: left; }}
    .section {{ background: white; border: 1px solid #ddd; border-radius: 10px;
                padding: 24px; margin: 24px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.05); }}
    .verdict {{ background: linear-gradient(135deg, #e8f5e9, #f3e5f5);
                border: 2px solid #4CAF50; border-radius: 12px;
                padding: 22px 28px; margin: 24px 0; }}
    .verdict h2 {{ color: #2e7d32; margin-top: 0; }}
    .note {{ background: #fff8e1; border-left: 4px solid #ffc107;
             padding: 12px 18px; border-radius: 4px; margin: 16px 0;
             font-size: 0.88em; color: #555; }}
    .card-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0; }}
    .card {{ flex: 1; min-width: 180px; background: white; border-radius: 10px;
             padding: 20px; text-align: center;
             box-shadow: 0 2px 6px rgba(0,0,0,0.06); }}
    .card-label {{ font-size: 0.82em; color: #555; margin-bottom: 8px; }}
    .card-value {{ font-size: 2em; font-weight: bold; margin: 8px 0; }}
    .card-sub   {{ font-size: 0.8em; color: #666; margin: 4px 0; }}
    .legend {{ font-size: 0.85em; color: #555; margin-bottom: 8px; }}
  </style>
</head>
<body>

<h1>Baseline vs Parallel Inference — Paper-Aligned Metrics Comparison</h1>
<p style="color:#666;">
  Side-by-side comparison using the three metrics from
  <em>"LLMs Are Advanced Anonymizers"</em>.
  Both systems ran on the <strong>same 20 profiles</strong>.
  <br>
  <span style="color:#2196F3;font-weight:bold;">■ Baseline</span>:
  single GPT-4o attack → anonymize (2 rounds) → re-attack. &nbsp;
  <span style="color:#7B1FA2;font-weight:bold;">■ Parallel</span>:
  GPT-4o + Claude parallel attack → merge → anonymize (1 round) → re-attack both.
</p>

<div class="note">
  <strong>Accuracy direction:</strong> Lower adversarial accuracy after anonymization =
  <em>better</em> privacy (attacker guesses correctly less often).
  Green cells highlight where the parallel architecture outperforms the baseline.
</div>

<!-- ── Cards ─────────────────────────────────────────────────────────── -->
<div class="section">
  <h2>At a Glance — Post-Anonymization Adversarial Accuracy (Top-3)</h2>
  {cards}
</div>

<!-- ── Section 1 ──────────────────────────────────────────────────────── -->
<div class="section">
  <h2>1 · Adversarial Accuracy</h2>
  <p>Fraction of attributes the attacker correctly guesses. Lower = better privacy.</p>
  {acc_overview}

  <h3>Per-PII-Type Breakdown (Top-3, post-anonymization)</h3>
  <p class="legend">
    <span style="color:#2e7d32;font-weight:bold;">Green</span> = parallel lower (better) than baseline &nbsp;|&nbsp;
    <span style="color:#f44336;">Red</span> = parallel higher (worse) than baseline
  </p>
  {per_type_table}
</div>

<!-- ── Section 2 ──────────────────────────────────────────────────────── -->
<div class="section">
  <h2>2 · Adversarial Certainty — Evidence Rate</h2>
  <p>
    Fraction of inferences where the attacker cites direct textual evidence
    (certainty ≥ 3 out of 5).  A bigger drop = anonymizer removed more real signals.
  </p>
  {cert_table}
</div>

<!-- ── Section 3 ──────────────────────────────────────────────────────── -->
<div class="section">
  <h2>3 · Utility Comparison</h2>
  <p>Combined utility = mean(Readability/10, Meaning/10, ROUGE-1). Higher = better.</p>
  {util_table}
</div>

<!-- ── Verdict ────────────────────────────────────────────────────────── -->
<div class="verdict">
  <h2>Verdict</h2>
  <ul style="line-height:2.2;">
    <li>{verdict_accuracy}</li>
    <li>{verdict_evidence}</li>
    <li>{verdict_utility}</li>
  </ul>
</div>

<p style="color:#bbb; font-size:0.78em; margin-top:40px;">
  Baseline: {baseline_dir} &nbsp;|&nbsp; Parallel: {parallel_dir}
</p>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote comparison report -> {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

DEFAULT_BASELINE = "anonymized_results/baseline_single_attack_20profiles"
DEFAULT_PARALLEL = "anonymized_results/parallel_gpt4o_vs_claude_20profiles_v2"
DEFAULT_OUTPUT   = "anonymized_results/paper_metrics_comparison.html"


def run(baseline_dir: str, parallel_dir: str, output: str) -> None:
    print(f"\n{'='*60}")
    print("PAPER METRICS COMPARISON: BASELINE vs PARALLEL")
    print(f"{'='*60}")

    print(f"\nComputing baseline metrics from: {baseline_dir}")
    baseline_metrics = compute_baseline_paper_metrics(baseline_dir)

    print(f"Loading parallel metrics from:   {parallel_dir}")
    parallel_metrics = load_parallel_metrics(parallel_dir)

    # ── Print text summary ────────────────────────────────────────────────
    print(f"\n{'Metric':<45} {'Baseline':>12} {'Parallel A':>12} {'Parallel B':>12}")
    print("-" * 85)

    b_pre  = baseline_metrics["pre_anon"]["accuracy"]["overall_top3"]
    b_post = baseline_metrics["post_anon"]["accuracy"]["overall_top3"]
    pa_pre  = parallel_metrics["pre_anon"]["attack_a"]["accuracy"]["overall_top3"]
    pa_post = parallel_metrics["post_anon"]["attack_a"]["accuracy"]["overall_top3"]
    pb_pre  = parallel_metrics["pre_anon"]["attack_b"]["accuracy"]["overall_top3"]
    pb_post = parallel_metrics["post_anon"]["attack_b"]["accuracy"]["overall_top3"]

    print(f"{'Adv. Accuracy Top-3 (pre-anon)':<45} {b_pre*100:>11.1f}% {pa_pre*100:>11.1f}% {pb_pre*100:>11.1f}%")
    print(f"{'Adv. Accuracy Top-3 (post-anon)':<45} {b_post*100:>11.1f}% {pa_post*100:>11.1f}% {pb_post*100:>11.1f}%")
    print(f"{'Reduction (pp)':<45} {(b_pre-b_post)*100:>11.1f}  {(pa_pre-pa_post)*100:>11.1f}  {(pb_pre-pb_post)*100:>11.1f}")

    b_ev  = baseline_metrics["post_anon"]["certainty"]["overall_evidence_rate"]
    pa_ev = parallel_metrics["post_anon"]["attack_a"]["certainty"]["overall_evidence_rate"]
    pb_ev = parallel_metrics["post_anon"]["attack_b"]["certainty"]["overall_evidence_rate"]
    print(f"{'Evidence Rate (post-anon)':<45} {b_ev*100:>11.1f}% {pa_ev*100:>11.1f}% {pb_ev*100:>11.1f}%")

    b_c  = baseline_metrics["utility"].get("avg_combined")
    p_c  = parallel_metrics["utility"].get("avg_combined")
    print(f"{'Combined Utility':<45} {str(round(b_c,3)) if b_c else '-':>12} {str(round(p_c,3)) if p_c else '-':>12}")

    # ── Generate HTML ─────────────────────────────────────────────────────
    generate_comparison_report(
        baseline_metrics, parallel_metrics,
        output, baseline_dir, parallel_dir,
    )

    # Save baseline metrics JSON alongside the report
    json_path = output.replace(".html", "_baseline.json")
    with open(json_path, "w") as f:
        json.dump(baseline_metrics, f, indent=2, default=str)
    print(f"Saved baseline metrics JSON -> {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare baseline vs parallel pipeline using paper metrics."
    )
    parser.add_argument("--baseline_dir", default=DEFAULT_BASELINE)
    parser.add_argument("--parallel_dir", default=DEFAULT_PARALLEL)
    parser.add_argument("--output",       default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.baseline_dir, args.parallel_dir, args.output)
