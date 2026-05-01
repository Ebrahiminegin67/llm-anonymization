"""
Four-Architecture Paper Metrics Comparison
==========================================

Compares all four anonymization architectures on the paper's metrics:

  1. Baseline       — single GPT-4o attack, generic anonymizer, 2 rounds
  2. Parallel       — GPT-4o + Claude parallel attack, generic anonymizer
  3. Sequential     — GPT-4o then Claude sequential attack, generic anonymizer
  4. Evid-Targeted  — GPT-4o + Claude parallel attack, evidence-targeted anonymizer  ← new

The thesis claim: architecture #4 should show lower post-anonymization adversarial
accuracy because the anonymizer explicitly targets the evidence passages flagged by
the multi-model attack, rather than doing generic inference-aware anonymization.

Usage:
  python compare_evidence_targeted_metrics.py
  python compare_evidence_targeted_metrics.py \\
      --baseline_dir  anonymized_results/baseline_single_attack_20profiles \\
      --parallel_dir  anonymized_results/parallel_gpt4o_vs_claude_20profiles_v2 \\
      --sequential_dir anonymized_results/sequential_gpt4o_then_claude_20profiles \\
      --evidence_dir  anonymized_results/evidence_targeted_20profiles \\
      --output        anonymized_results/evidence_targeted_comparison.html
"""

import os
import sys
import json
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__)))

from compare_baseline_vs_parallel_paper_metrics import compute_baseline_paper_metrics
from evaluate_parallel_paper_metrics import (
    compute_paper_metrics,
    compute_sequential_paper_metrics,
)


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_or_compute(d: dict, dir_path: str, compute_fn) -> dict:
    cache = f"{dir_path}/paper_metrics.json"
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    result = compute_fn(dir_path)
    with open(cache, "w") as f:
        json.dump(result, f, indent=2)
    return result


def load_parallel_metrics(parallel_dir: str) -> dict:
    return load_or_compute({}, parallel_dir, compute_paper_metrics)


def load_sequential_metrics(sequential_dir: str) -> dict:
    return load_or_compute({}, sequential_dir, compute_sequential_paper_metrics)


def load_evidence_metrics(evidence_dir: str) -> dict:
    """Evidence-targeted output has the same format as parallel (uses same attack structure)."""
    return load_or_compute({}, evidence_dir, compute_paper_metrics)


# ── HTML report ───────────────────────────────────────────────────────────────

def _pct(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v)*100:.1f}%"
    except (TypeError, ValueError):
        return str(v)


def _f(v, decimals=3) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _delta_cell(new_val, ref_val, lower_is_better=True) -> str:
    """Return an HTML <td> coloured green/red based on improvement direction."""
    try:
        delta = float(new_val) - float(ref_val)
    except (TypeError, ValueError):
        return f"<td>N/A</td>"
    if abs(delta) < 0.001:
        colour = "#888"
        sign = ""
    elif (delta < 0) == lower_is_better:
        colour = "#2e7d32"   # green = improvement
        sign = ""
    else:
        colour = "#c62828"   # red   = regression
        sign = "+"
    pct_str = f"{sign}{delta*100:+.1f}pp" if abs(delta) <= 1 else f"{sign}{delta:+.3f}"
    return f'<td style="color:{colour};font-weight:bold">{pct_str}</td>'


def generate_comparison_report(bm, pm, sm, em, output_path):
    """Generate HTML four-way comparison report."""

    def _safe(metrics, *keys, default=None):
        obj = metrics
        for k in keys:
            if not isinstance(obj, dict):
                return default
            obj = obj.get(k, default)
            if obj is None:
                return default
        return obj

    # ── Summary values ────────────────────────────────────────────────────────
    # Adversarial accuracy (Top-3, post-anonymization)
    b_acc  = _safe(bm,  "post_anon", "accuracy", "overall_top3")
    pa_acc = _safe(pm,  "post_anon", "merged",  "accuracy", "overall_top3")
    sa_acc = _safe(sm,  "post_anon", "accumulated", "accuracy", "overall_top3")
    ea_acc = _safe(em,  "post_anon", "merged",  "accuracy", "overall_top3")

    # Evidence rate (post-anonymization)
    b_ev   = _safe(bm,  "post_anon", "certainty", "overall_evidence_rate")
    pa_ev  = _safe(pm,  "post_anon", "merged",  "certainty", "overall_evidence_rate")
    sa_ev  = _safe(sm,  "post_anon", "accumulated", "certainty", "overall_evidence_rate")
    ea_ev  = _safe(em,  "post_anon", "merged",  "certainty", "overall_evidence_rate")

    # Combined utility
    b_ut   = _safe(bm,  "utility", "avg_combined")
    pa_ut  = _safe(pm,  "utility", "avg_combined")
    sa_ut  = _safe(sm,  "utility", "avg_combined")
    ea_ut  = _safe(em,  "utility", "avg_combined")

    # Pre-anon accuracy (to show the attack itself)
    b_pre  = _safe(bm,  "pre_anon", "accuracy", "overall_top3")
    pa_pre = _safe(pm,  "pre_anon", "merged", "accuracy", "overall_top3")
    sa_pre = _safe(sm,  "pre_anon", "accumulated", "accuracy", "overall_top3")
    ea_pre = _safe(em,  "pre_anon", "merged", "accuracy", "overall_top3")

    cards_html = f"""
    <div class="cards">
      <div class="card baseline">
        <div class="card-title">Baseline</div>
        <div class="card-subtitle">Single GPT-4o attack<br>Generic anonymizer (2 rounds)</div>
        <div class="metric-row"><span>Adv. Accuracy (post)</span><span class="val">{_pct(b_acc)}</span></div>
        <div class="metric-row"><span>Evidence Rate (post)</span><span class="val">{_pct(b_ev)}</span></div>
        <div class="metric-row"><span>Combined Utility</span><span class="val">{_f(b_ut)}</span></div>
      </div>
      <div class="card parallel">
        <div class="card-title">Parallel</div>
        <div class="card-subtitle">GPT-4o + Claude parallel<br>Generic anonymizer</div>
        <div class="metric-row"><span>Adv. Accuracy (post)</span><span class="val">{_pct(pa_acc)}</span></div>
        <div class="metric-row"><span>Evidence Rate (post)</span><span class="val">{_pct(pa_ev)}</span></div>
        <div class="metric-row"><span>Combined Utility</span><span class="val">{_f(pa_ut)}</span></div>
      </div>
      <div class="card sequential">
        <div class="card-title">Sequential</div>
        <div class="card-subtitle">GPT-4o then Claude<br>Generic anonymizer</div>
        <div class="metric-row"><span>Adv. Accuracy (post)</span><span class="val">{_pct(sa_acc)}</span></div>
        <div class="metric-row"><span>Evidence Rate (post)</span><span class="val">{_pct(sa_ev)}</span></div>
        <div class="metric-row"><span>Combined Utility</span><span class="val">{_f(sa_ut)}</span></div>
      </div>
      <div class="card evidence">
        <div class="card-title">Evidence-Targeted</div>
        <div class="card-subtitle">GPT-4o + Claude parallel<br>Evidence-targeted anonymizer &#x2605;</div>
        <div class="metric-row"><span>Adv. Accuracy (post)</span><span class="val highlight">{_pct(ea_acc)}</span></div>
        <div class="metric-row"><span>Evidence Rate (post)</span><span class="val highlight">{_pct(ea_ev)}</span></div>
        <div class="metric-row"><span>Combined Utility</span><span class="val highlight">{_f(ea_ut)}</span></div>
      </div>
    </div>
    """

    main_table = f"""
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>Baseline</th>
          <th>Parallel</th>
          <th>Sequential</th>
          <th>Evid-Targeted</th>
          <th>vs Baseline</th>
          <th>vs Parallel</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Pre-anon Accuracy (Top-3)</strong></td>
          <td>{_pct(b_pre)}</td>
          <td>{_pct(pa_pre)}</td>
          <td>{_pct(sa_pre)}</td>
          <td>{_pct(ea_pre)}</td>
          {_delta_cell(ea_pre, b_pre,  lower_is_better=False)}
          {_delta_cell(ea_pre, pa_pre, lower_is_better=False)}
        </tr>
        <tr class="highlight-row">
          <td><strong>Post-anon Accuracy (Top-3)</strong><br><small>Lower = better privacy</small></td>
          <td>{_pct(b_acc)}</td>
          <td>{_pct(pa_acc)}</td>
          <td>{_pct(sa_acc)}</td>
          <td><strong>{_pct(ea_acc)}</strong></td>
          {_delta_cell(ea_acc, b_acc,  lower_is_better=True)}
          {_delta_cell(ea_acc, pa_acc, lower_is_better=True)}
        </tr>
        <tr class="highlight-row">
          <td><strong>Post-anon Evidence Rate</strong><br><small>Lower = less textual evidence</small></td>
          <td>{_pct(b_ev)}</td>
          <td>{_pct(pa_ev)}</td>
          <td>{_pct(sa_ev)}</td>
          <td><strong>{_pct(ea_ev)}</strong></td>
          {_delta_cell(ea_ev, b_ev,  lower_is_better=True)}
          {_delta_cell(ea_ev, pa_ev, lower_is_better=True)}
        </tr>
        <tr>
          <td><strong>Combined Utility</strong><br><small>Higher = better text quality</small></td>
          <td>{_f(b_ut)}</td>
          <td>{_f(pa_ut)}</td>
          <td>{_f(sa_ut)}</td>
          <td><strong>{_f(ea_ut)}</strong></td>
          {_delta_cell(ea_ut, b_ut,  lower_is_better=False)}
          {_delta_cell(ea_ut, pa_ut, lower_is_better=False)}
        </tr>
      </tbody>
    </table>
    """

    # Per-PII-type breakdown
    pii_types = set()
    for metrics, path in [(pm, ("post_anon", "merged", "accuracy", "per_type")),
                          (em, ("post_anon", "merged", "accuracy", "per_type"))]:
        d = _safe(metrics, *path) or {}
        pii_types.update(d.keys())

    pii_rows = ""
    for pii_type in sorted(pii_types):
        b_p  = _safe(bm, "post_anon", "accuracy", "per_type", pii_type, "top3")
        pa_p = _safe(pm, "post_anon", "merged", "accuracy", "per_type", pii_type, "top3")
        sa_p = _safe(sm, "post_anon", "accumulated", "accuracy", "per_type", pii_type, "top3")
        ea_p = _safe(em, "post_anon", "merged", "accuracy", "per_type", pii_type, "top3")
        pii_rows += f"""
        <tr>
          <td>{pii_type}</td>
          <td>{_pct(b_p)}</td>
          <td>{_pct(pa_p)}</td>
          <td>{_pct(sa_p)}</td>
          <td><strong>{_pct(ea_p)}</strong></td>
          {_delta_cell(ea_p, b_p,  lower_is_better=True)}
          {_delta_cell(ea_p, pa_p, lower_is_better=True)}
        </tr>"""

    pii_table = f"""
    <table>
      <thead>
        <tr>
          <th>PII Type</th>
          <th>Baseline</th>
          <th>Parallel</th>
          <th>Sequential</th>
          <th>Evid-Targeted</th>
          <th>vs Baseline</th>
          <th>vs Parallel</th>
        </tr>
      </thead>
      <tbody>{pii_rows}</tbody>
    </table>
    """ if pii_rows else "<p>No per-type data available.</p>"

    # Verdict
    improvement_vs_baseline = None
    improvement_vs_parallel = None
    if ea_acc is not None and b_acc is not None:
        improvement_vs_baseline = (b_acc - ea_acc)
    if ea_acc is not None and pa_acc is not None:
        improvement_vs_parallel = (pa_acc - ea_acc)

    def _verdict_line(improvement, opponent):
        if improvement is None:
            return f"<li>Cannot compare vs {opponent} (missing data).</li>"
        if improvement > 0.05:
            return (f"<li><b style='color:#2e7d32'>IMPROVEMENT vs {opponent}</b>: "
                    f"adversarial accuracy reduced by {improvement*100:.1f}pp — "
                    f"evidence-targeted anonymization is working.</li>")
        elif improvement > 0:
            return (f"<li><b style='color:#f9a825'>MARGINAL IMPROVEMENT vs {opponent}</b>: "
                    f"{improvement*100:.1f}pp reduction — trend in right direction "
                    f"but within noise range for n=20.</li>")
        elif improvement > -0.02:
            return (f"<li><b style='color:#888'>NO DIFFERENCE vs {opponent}</b>: "
                    f"{improvement*100:.1f}pp — evidence-targeted anonymizer performs "
                    f"similarly to generic anonymizer on this dataset.</li>")
        else:
            return (f"<li><b style='color:#c62828'>REGRESSION vs {opponent}</b>: "
                    f"accuracy higher by {-improvement*100:.1f}pp — check whether "
                    f"the anonymizer is being too cautious with targeted edits.</li>")

    verdict_html = f"""
    <div class="verdict">
      <h3>Thesis Verdict: Does Evidence-Targeted Anonymization Improve Privacy?</h3>
      <ul>
        {_verdict_line(improvement_vs_baseline, "Baseline")}
        {_verdict_line(improvement_vs_parallel, "Parallel (same attack)")}
      </ul>
      <p><b>Note on utility:</b>
        Evid-targeted utility = {_f(ea_ut)} vs baseline = {_f(b_ut)}.
        {"Utility is maintained — surgical targeting preserved text quality."
          if ea_ut is not None and b_ut is not None and abs(float(ea_ut or 0) - float(b_ut or 0)) < 0.02
          else "There is a utility difference — check whether the anonymizer is over-editing."}
      </p>
      <p><b>Sample size caveat:</b>
        n=20 profiles, one PII attribute each = 20 data points.
        A 10pp accuracy difference requires ~100 profiles for 80% statistical power (two-proportion z-test).
        Treat all differences as directional trends unless the gap exceeds ~15pp.
      </p>
    </div>
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Evidence-Targeted Anonymization: Four-Architecture Comparison</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; color: #212121; }}
  h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }}
  h2 {{ color: #283593; margin-top: 32px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 10px 14px; text-align: center; }}
  th {{ background: #e8eaf6; color: #1a237e; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  .highlight-row {{ background: #fff8e1 !important; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0; }}
  .card {{ flex: 1; min-width: 200px; border-radius: 8px; padding: 16px;
           box-shadow: 0 2px 8px rgba(0,0,0,.12); }}
  .card.baseline  {{ background: #fce4ec; border-top: 4px solid #e91e63; }}
  .card.parallel  {{ background: #e3f2fd; border-top: 4px solid #1976d2; }}
  .card.sequential {{ background: #e8f5e9; border-top: 4px solid #388e3c; }}
  .card.evidence  {{ background: #ede7f6; border-top: 4px solid #7b1fa2; }}
  .card-title {{ font-size: 1.1em; font-weight: bold; margin-bottom: 4px; }}
  .card-subtitle {{ font-size: 0.8em; color: #555; margin-bottom: 12px; }}
  .metric-row {{ display: flex; justify-content: space-between; padding: 4px 0;
                 border-bottom: 1px solid rgba(0,0,0,.06); }}
  .val {{ font-weight: bold; }}
  .highlight {{ color: #7b1fa2; font-weight: bold; font-size: 1.1em; }}
  .verdict {{ background: #f3e5f5; border-left: 4px solid #7b1fa2;
              padding: 16px; border-radius: 4px; margin: 20px 0; }}
  .verdict h3 {{ margin-top: 0; color: #4a148c; }}
  .method-box {{ background: #ede7f6; border-radius: 6px; padding: 14px;
                 margin: 12px 0; border-left: 4px solid #7b1fa2; }}
</style>
</head>
<body>
<h1>Evidence-Targeted Anonymization: Four-Architecture Comparison</h1>
<p>Paper-aligned metrics from <em>LLMs Are Advanced Anonymizers</em>.
   Lower adversarial accuracy = stronger privacy protection.</p>

<div class="method-box">
  <strong>New contribution — Evidence-Targeted Anonymizer:</strong>
  instead of giving the anonymizer generic inference summaries (Type / Inference / Guess),
  the anonymizer receives both attackers' full reasoning chains and is explicitly instructed
  to (1) identify the exact phrases the attackers are using as evidence, then (2) surgically
  replace only those phrases. The hypothesis: explicit evidence targeting should yield lower
  post-anonymization adversarial accuracy than generic anonymization, even with the same
  attack architecture.
</div>

<h2>Summary Cards</h2>
{cards_html}

<h2>Full Metrics Comparison</h2>
{main_table}

<h2>Per-PII-Type Accuracy (post-anonymization, Top-3)</h2>
{pii_table}

{verdict_html}

<hr>
<p style="color:#888;font-size:.85em">
  Metrics: Adversarial Accuracy = fraction of ground-truth PII correctly guessed (Top-3);
  Evidence Rate = fraction with direct textual evidence (certainty &ge; 3);
  Combined Utility = mean(Readability/10, Meaning/10, ROUGE-1 F1).
</p>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote report -> {output_path}")


# ── Terminal summary ──────────────────────────────────────────────────────────

def print_summary(bm, pm, sm, em):
    def _p(m, *keys):
        v = m
        for k in keys:
            if not isinstance(v, dict):
                return None
            v = v.get(k)
        return v

    def _pct(v):
        if v is None: return "N/A  "
        return f"{float(v)*100:5.1f}%"

    def _ff(v):
        if v is None: return "  N/A"
        return f"{float(v):.3f}"

    print()
    print("=" * 80)
    print("FOUR-ARCHITECTURE PAPER METRICS COMPARISON")
    print("=" * 80)
    hdr = f"{'Metric':<40} {'Baseline':>10} {'Parallel':>10} {'Sequntl':>10} {'Evid-Tgt':>10}"
    sep = "-" * 80
    print(hdr)
    print(sep)

    rows = [
        ("Adv. Accuracy PRE-anon (Top-3)",
         _pct(_p(bm,  "pre_anon",  "accuracy", "overall_top3")),
         _pct(_p(pm,  "pre_anon",  "merged",  "accuracy", "overall_top3")),
         _pct(_p(sm,  "pre_anon",  "accumulated", "accuracy", "overall_top3")),
         _pct(_p(em,  "pre_anon",  "merged",  "accuracy", "overall_top3")),
        ),
        ("Adv. Accuracy POST-anon (Top-3)",
         _pct(_p(bm,  "post_anon", "accuracy", "overall_top3")),
         _pct(_p(pm,  "post_anon", "merged",  "accuracy", "overall_top3")),
         _pct(_p(sm,  "post_anon", "accumulated", "accuracy", "overall_top3")),
         _pct(_p(em,  "post_anon", "merged",  "accuracy", "overall_top3")),
        ),
        ("Evidence Rate POST-anon",
         _pct(_p(bm,  "post_anon", "certainty", "overall_evidence_rate")),
         _pct(_p(pm,  "post_anon", "merged",  "certainty", "overall_evidence_rate")),
         _pct(_p(sm,  "post_anon", "accumulated", "certainty", "overall_evidence_rate")),
         _pct(_p(em,  "post_anon", "merged",  "certainty", "overall_evidence_rate")),
        ),
        ("Combined Utility",
         _ff(_p(bm, "utility", "avg_combined")),
         _ff(_p(pm, "utility", "avg_combined")),
         _ff(_p(sm, "utility", "avg_combined")),
         _ff(_p(em, "utility", "avg_combined")),
        ),
    ]

    for row in rows:
        label, *vals = row
        print(f"{label:<40} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

    print("=" * 80)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    baseline_dir:  str = "anonymized_results/baseline_single_attack_20profiles",
    parallel_dir:  str = "anonymized_results/parallel_gpt4o_vs_claude_20profiles_v2",
    sequential_dir: str = "anonymized_results/sequential_gpt4o_then_claude_20profiles",
    evidence_dir:  str = "anonymized_results/evidence_targeted_20profiles",
    output_path:   str = "anonymized_results/evidence_targeted_comparison.html",
):
    print(f"Baseline   : {baseline_dir}")
    print(f"Parallel   : {parallel_dir}")
    print(f"Sequential : {sequential_dir}")
    print(f"Evid-Tgt   : {evidence_dir}")

    if not os.path.exists(baseline_dir):
        print(f"ERROR: baseline directory not found: {baseline_dir}"); return
    if not os.path.exists(parallel_dir):
        print(f"ERROR: parallel directory not found: {parallel_dir}"); return
    if not os.path.exists(evidence_dir):
        print(f"ERROR: evidence-targeted directory not found: {evidence_dir}")
        print("       Run the pipeline first:")
        print("       python run_evidence_targeted_pipeline.py --config_path configs/anonymization/evidence_targeted.yaml")
        return

    bm = compute_baseline_paper_metrics(baseline_dir)
    pm = load_parallel_metrics(parallel_dir)
    sm = load_sequential_metrics(sequential_dir) if os.path.exists(sequential_dir) else {}
    em = load_evidence_metrics(evidence_dir)

    print_summary(bm, pm, sm, em)
    generate_comparison_report(bm, pm, sm, em, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Four-architecture paper metrics comparison")
    parser.add_argument("--baseline_dir",   default="anonymized_results/baseline_single_attack_20profiles")
    parser.add_argument("--parallel_dir",   default="anonymized_results/parallel_gpt4o_vs_claude_20profiles_v2")
    parser.add_argument("--sequential_dir", default="anonymized_results/sequential_gpt4o_then_claude_20profiles")
    parser.add_argument("--evidence_dir",   default="anonymized_results/evidence_targeted_20profiles")
    parser.add_argument("--output",         default="anonymized_results/evidence_targeted_comparison.html")
    args = parser.parse_args()

    run(
        baseline_dir=args.baseline_dir,
        parallel_dir=args.parallel_dir,
        sequential_dir=args.sequential_dir,
        evidence_dir=args.evidence_dir,
        output_path=args.output,
    )
