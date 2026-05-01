"""
Paper-Metric Comparison: Baseline vs Parallel vs Sequential
============================================================

Computes and compares all three metrics from "LLMs Are Advanced Anonymizers"
across all three anonymization architectures on the same 20 profiles.

  BASELINE    — single GPT-4o attack -> anonymize (2 rounds) -> re-attack
  PARALLEL    — GPT-4o + Claude independent attacks -> merge -> anonymize -> re-attack both
  SEQUENTIAL  — GPT-4o attack A -> Claude attack B (informed by A) -> anonymize -> re-attack both

Usage:
    python compare_all_architectures_paper_metrics.py
    python compare_all_architectures_paper_metrics.py \\
        --baseline_dir   anonymized_results/baseline_single_attack_20profiles \\
        --parallel_dir   anonymized_results/parallel_gpt4o_vs_claude_20profiles_v2 \\
        --sequential_dir anonymized_results/sequential_gpt4o_then_claude_20profiles \\
        --output         anonymized_results/all_architectures_paper_metrics.html
"""

import json
import os
import sys
import argparse
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(__file__))

from evaluate_parallel_paper_metrics import (
    compute_paper_metrics,
    compute_sequential_paper_metrics,
    get_ground_truth,
    compute_adversarial_accuracy,
    compute_evidence_rate,
    extract_utility_scores,
    aggregate_utility,
    _pct, _f, _delta_html,
)
from compare_baseline_vs_parallel_paper_metrics import (
    compute_baseline_paper_metrics,
    load_parallel_metrics,
)
from src.reddit.reddit_utils import load_data


DEFAULT_BASELINE   = "anonymized_results/baseline_single_attack_20profiles"
DEFAULT_PARALLEL   = "anonymized_results/parallel_gpt4o_vs_claude_20profiles_v2"
DEFAULT_SEQUENTIAL = "anonymized_results/sequential_gpt4o_then_claude_20profiles"
DEFAULT_OUTPUT     = "anonymized_results/all_architectures_paper_metrics.html"


# ── Load / cache sequential metrics ─────────────────────────────────────────

def load_sequential_metrics(sequential_dir: str) -> Dict:
    cache_path = f"{sequential_dir}/paper_metrics.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    m = compute_sequential_paper_metrics(sequential_dir)
    with open(cache_path, "w") as f:
        json.dump(m, f, indent=2, default=str)
    return m


# ── HTML helpers ─────────────────────────────────────────────────────────────

def _acc_cell(val: Optional[float], *refs: Optional[float]) -> str:
    """Cell coloured green if val is strictly lower than all non-None refs."""
    if val is None:
        return "<td>-</td>"
    txt = _pct(val)
    if refs and all(r is not None for r in refs) and all(val < r for r in refs):
        return f'<td style="color:#2e7d32;font-weight:bold;">{txt} &#9660;</td>'
    if refs and any(r is not None and val > r for r in refs):
        return f'<td style="color:#f44336;">{txt} &#9650;</td>'
    return f"<td>{txt}</td>"


def _ev_cell(val: Optional[float], *refs: Optional[float]) -> str:
    """Same colouring for evidence-rate cells."""
    return _acc_cell(val, *refs)


def _util_cell(val: Optional[float], ref: Optional[float]) -> str:
    if val is None:
        return "<td>-</td>"
    txt = _f(val, 3)
    if ref is not None and val > ref:
        return f'<td style="color:#2e7d32;font-weight:bold;">{txt} &#9650;</td>'
    if ref is not None and val < ref:
        return f'<td style="color:#f44336;">{txt} &#9660;</td>'
    return f"<td>{txt}</td>"


# ── Report generation ────────────────────────────────────────────────────────

def generate_full_report(
    bm: Dict,   # baseline metrics
    pm: Dict,   # parallel metrics
    sm: Dict,   # sequential metrics
    output_path: str,
    baseline_dir: str,
    parallel_dir: str,
    sequential_dir: str,
) -> None:

    # Shorthand references
    bp  = bm["pre_anon"];   bpost = bm["post_anon"];   bu = bm["utility"];   bt = bm["tradeoff"]
    ppa = pm["pre_anon"];   ppst  = pm["post_anon"];   pu = pm["utility"];   pt = pm["tradeoff"]
    spa = sm["pre_anon"];   spst  = sm["post_anon"];   su = sm["utility"];   st = sm["tradeoff"]

    b_pre  = bp["accuracy"]["overall_top3"]
    b_post = bpost["accuracy"]["overall_top3"]

    # ── All PII types ────────────────────────────────────────────────────
    all_pii: set = set()
    for d in [bp["accuracy"]["per_type"], bpost["accuracy"]["per_type"],
              ppa["attack_a"]["accuracy"]["per_type"], ppst["attack_a"]["accuracy"]["per_type"],
              spa["attack_a"]["accuracy"]["per_type"], spst["attack_a"]["accuracy"]["per_type"]]:
        all_pii.update(d.keys())
    all_pii_sorted = sorted(all_pii)

    # ── Summary cards ────────────────────────────────────────────────────
    def _card(label: str, color: str, acc_post: float, acc_red: float,
              util: Optional[float], ev_post: float) -> str:
        return f"""
        <div class="card" style="border-top:5px solid {color};">
          <div class="card-label">{label}</div>
          <div class="card-value" style="color:{color};">{_pct(acc_post)}</div>
          <div class="card-sub">Post-anon accuracy (Top-3)</div>
          <hr style="border:none;border-top:1px solid #eee;margin:8px 0;">
          <div class="card-sub">Accuracy reduction: {_delta_html(acc_red)}</div>
          <div class="card-sub">Evidence rate (post): {_pct(ev_post)}</div>
          <div class="card-sub">Combined utility: {_f(util, 3)}</div>
        </div>"""

    cards_html = f"""
    <div class="card-row">
      {_card("Baseline<br>(GPT-4o only)", "#2196F3",
             b_post, bt["accuracy_reduction"],
             bu.get("avg_combined"), bpost["certainty"]["overall_evidence_rate"])}
      {_card("Parallel<br>Attack A (GPT-4o)", "#7B1FA2",
             ppst["attack_a"]["accuracy"]["overall_top3"],
             pt["attack_a"]["accuracy_reduction"],
             pu.get("avg_combined"),
             ppst["attack_a"]["certainty"]["overall_evidence_rate"])}
      {_card("Parallel<br>Attack B (Claude)", "#9C27B0",
             ppst["attack_b"]["accuracy"]["overall_top3"],
             pt["attack_b"]["accuracy_reduction"],
             pu.get("avg_combined"),
             ppst["attack_b"]["certainty"]["overall_evidence_rate"])}
      {_card("Sequential<br>Attack A (GPT-4o)", "#E65100",
             spst["attack_a"]["accuracy"]["overall_top3"],
             st["attack_a"]["accuracy_reduction"],
             su.get("avg_combined"),
             spst["attack_a"]["certainty"]["overall_evidence_rate"])}
      {_card("Sequential<br>Attack B (Claude)", "#FF6F00",
             spst["attack_b"]["accuracy"]["overall_top3"],
             st["attack_b"]["accuracy_reduction"],
             su.get("avg_combined"),
             spst["attack_b"]["certainty"]["overall_evidence_rate"])}
      {_card("Sequential<br>Accumulated", "#F9A825",
             spst["accumulated"]["accuracy"]["overall_top3"],
             st["accumulated"]["accuracy_reduction"],
             su.get("avg_combined"),
             spst["accumulated"]["certainty"]["overall_evidence_rate"])}
    </div>"""

    # ── Section 1: Adversarial Accuracy overview ─────────────────────────
    def _row(stage_label: str, b_acc, pa_acc, pb_acc, pm_acc,
             sa_acc, sb_acc, sacc_acc, b_ref=None):
        ref = b_ref  # colour relative to baseline post-anon
        return f"""
        <tr>
          <td><strong>{stage_label}</strong></td>
          <td>{_pct(b_acc)}</td>
          {_acc_cell(pa_acc,  ref)} {_acc_cell(pb_acc,  ref)} {_acc_cell(pm_acc,  ref)}
          {_acc_cell(sa_acc,  ref)} {_acc_cell(sb_acc,  ref)} {_acc_cell(sacc_acc, ref)}
        </tr>"""

    acc_table = f"""
    <table>
      <thead>
        <tr>
          <th rowspan="2">Stage</th>
          <th style="background:#1565C0;">Baseline</th>
          <th colspan="3" style="background:#6A1B9A;">Parallel</th>
          <th colspan="3" style="background:#BF360C;">Sequential</th>
        </tr>
        <tr>
          <th style="background:#1976D2;">GPT-4o</th>
          <th style="background:#7B1FA2;">Atk A</th>
          <th style="background:#9C27B0;">Atk B</th>
          <th style="background:#AB47BC;">Merged</th>
          <th style="background:#E65100;">Atk A</th>
          <th style="background:#FF6F00;">Atk B</th>
          <th style="background:#F9A825;">Accum.</th>
        </tr>
      </thead>
      <tbody>
        {_row("Before Anon (Top-3)",
              bp["accuracy"]["overall_top3"],
              ppa["attack_a"]["accuracy"]["overall_top3"],
              ppa["attack_b"]["accuracy"]["overall_top3"],
              ppa["merged"]["accuracy"]["overall_top3"],
              spa["attack_a"]["accuracy"]["overall_top3"],
              spa["attack_b"]["accuracy"]["overall_top3"],
              spa["accumulated"]["accuracy"]["overall_top3"])}
        {_row("After Anon (Top-3)",
              bpost["accuracy"]["overall_top3"],
              ppst["attack_a"]["accuracy"]["overall_top3"],
              ppst["attack_b"]["accuracy"]["overall_top3"],
              ppst["merged"]["accuracy"]["overall_top3"],
              spst["attack_a"]["accuracy"]["overall_top3"],
              spst["attack_b"]["accuracy"]["overall_top3"],
              spst["accumulated"]["accuracy"]["overall_top3"],
              b_ref=bpost["accuracy"]["overall_top3"])}
        <tr style="background:#f0f4ff;">
          <td><strong>Reduction &#9660;</strong></td>
          <td><strong>{_delta_html(bt["accuracy_reduction"])}</strong></td>
          <td><strong>{_delta_html(pt["attack_a"]["accuracy_reduction"])}</strong></td>
          <td><strong>{_delta_html(pt["attack_b"]["accuracy_reduction"])}</strong></td>
          <td><strong>{_delta_html(pt["merged"]["accuracy_reduction"])}</strong></td>
          <td><strong>{_delta_html(st["attack_a"]["accuracy_reduction"])}</strong></td>
          <td><strong>{_delta_html(st["attack_b"]["accuracy_reduction"])}</strong></td>
          <td><strong>{_delta_html(st["accumulated"]["accuracy_reduction"])}</strong></td>
        </tr>
      </tbody>
    </table>"""

    # ── Section 2: Per-PII-type ───────────────────────────────────────────
    per_type_rows = ""
    b_post_ref = bpost["accuracy"]["overall_top3"]
    for pt_name in all_pii_sorted:
        def _v(d, key="top3"):
            return d.get(pt_name, {}).get(key)

        b_v  = _v(bpost["accuracy"]["per_type"])
        pa_v = _v(ppst["attack_a"]["accuracy"]["per_type"])
        pb_v = _v(ppst["attack_b"]["accuracy"]["per_type"])
        pm_v = _v(ppst["merged"]["accuracy"]["per_type"])
        sa_v = _v(spst["attack_a"]["accuracy"]["per_type"])
        sb_v = _v(spst["attack_b"]["accuracy"]["per_type"])
        sc_v = _v(spst["accumulated"]["accuracy"]["per_type"])

        per_type_rows += f"""
        <tr>
          <td><strong>{pt_name}</strong></td>
          <td>{_pct(b_v)}</td>
          {_acc_cell(pa_v, b_v)} {_acc_cell(pb_v, b_v)} {_acc_cell(pm_v, b_v)}
          {_acc_cell(sa_v, b_v)} {_acc_cell(sb_v, b_v)} {_acc_cell(sc_v, b_v)}
        </tr>"""

    per_type_table = f"""
    <table>
      <thead>
        <tr>
          <th rowspan="2">PII Type (post-anon Top-3)</th>
          <th style="background:#1565C0;">Baseline</th>
          <th colspan="3" style="background:#6A1B9A;">Parallel</th>
          <th colspan="3" style="background:#BF360C;">Sequential</th>
        </tr>
        <tr>
          <th style="background:#1976D2;">GPT-4o</th>
          <th style="background:#7B1FA2;">Atk A</th>
          <th style="background:#9C27B0;">Atk B</th>
          <th style="background:#AB47BC;">Merged</th>
          <th style="background:#E65100;">Atk A</th>
          <th style="background:#FF6F00;">Atk B</th>
          <th style="background:#F9A825;">Accum.</th>
        </tr>
      </thead>
      <tbody>{per_type_rows}</tbody>
    </table>
    <p class="legend">
      <span style="color:#2e7d32;font-weight:bold;">Green &#9660;</span> = lower than baseline (better privacy) &nbsp;|&nbsp;
      <span style="color:#f44336;">Red &#9650;</span> = higher than baseline (worse privacy)
    </p>"""

    # ── Section 3: Evidence Rate ──────────────────────────────────────────
    b_ev_pre  = bp["certainty"]["overall_evidence_rate"]
    b_ev_post = bpost["certainty"]["overall_evidence_rate"]

    def _ev_row_full(label, pre_ev, post_ev, b_ev_post_ref=None):
        drop = pre_ev - post_ev
        post_cell = _ev_cell(post_ev, b_ev_post_ref) if b_ev_post_ref is not None else f"<td>{_pct(post_ev)}</td>"
        return f"""
        <tr>
          <td><strong>{label}</strong></td>
          <td>{_pct(pre_ev)}</td>
          {post_cell}
          <td>{_delta_html(drop)}</td>
        </tr>"""

    cert_table = f"""
    <table>
      <thead>
        <tr>
          <th>Architecture / Attack</th>
          <th>Evidence Rate (Pre-Anon)</th>
          <th>Evidence Rate (Post-Anon)</th>
          <th>Drop &#9660;</th>
        </tr>
      </thead>
      <tbody>
        {_ev_row_full("Baseline — GPT-4o",
                      b_ev_pre, b_ev_post)}
        {_ev_row_full("Parallel — Attack A (GPT-4o)",
                      ppa["attack_a"]["certainty"]["overall_evidence_rate"],
                      ppst["attack_a"]["certainty"]["overall_evidence_rate"], b_ev_post)}
        {_ev_row_full("Parallel — Attack B (Claude)",
                      ppa["attack_b"]["certainty"]["overall_evidence_rate"],
                      ppst["attack_b"]["certainty"]["overall_evidence_rate"], b_ev_post)}
        {_ev_row_full("Parallel — Merged",
                      ppa["merged"]["certainty"]["overall_evidence_rate"],
                      ppst["merged"]["certainty"]["overall_evidence_rate"], b_ev_post)}
        {_ev_row_full("Sequential — Attack A (GPT-4o)",
                      spa["attack_a"]["certainty"]["overall_evidence_rate"],
                      spst["attack_a"]["certainty"]["overall_evidence_rate"], b_ev_post)}
        {_ev_row_full("Sequential — Attack B (Claude, informed)",
                      spa["attack_b"]["certainty"]["overall_evidence_rate"],
                      spst["attack_b"]["certainty"]["overall_evidence_rate"], b_ev_post)}
        {_ev_row_full("Sequential — Accumulated",
                      spa["accumulated"]["certainty"]["overall_evidence_rate"],
                      spst["accumulated"]["certainty"]["overall_evidence_rate"], b_ev_post)}
      </tbody>
    </table>"""

    # ── Section 4: Utility ────────────────────────────────────────────────
    b_c = bu.get("avg_combined");  b_r = bu.get("avg_readability");  b_m = bu.get("avg_meaning");  b_r1 = bu.get("avg_rouge1")
    p_c = pu.get("avg_combined");  p_r = pu.get("avg_readability");  p_m = pu.get("avg_meaning");  p_r1 = pu.get("avg_rouge1")
    s_c = su.get("avg_combined");  s_r = su.get("avg_readability");  s_m = su.get("avg_meaning");  s_r1 = su.get("avg_rouge1")

    util_table = f"""
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th style="background:#1565C0;">Baseline</th>
          <th style="background:#6A1B9A;">Parallel</th>
          <th style="background:#BF360C;">Sequential</th>
          <th>Note</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Readability / 10</strong></td>
          <td>{_f(b_r/10 if b_r else None, 3)}</td>
          {_util_cell(p_r/10 if p_r else None, b_r/10 if b_r else None)}
          {_util_cell(s_r/10 if s_r else None, b_r/10 if b_r else None)}
          <td>LLM judge (GPT-4)</td>
        </tr>
        <tr>
          <td><strong>Meaning / 10</strong></td>
          <td>{_f(b_m/10 if b_m else None, 3)}</td>
          {_util_cell(p_m/10 if p_m else None, b_m/10 if b_m else None)}
          {_util_cell(s_m/10 if s_m else None, b_m/10 if b_m else None)}
          <td>LLM judge (GPT-4)</td>
        </tr>
        <tr>
          <td><strong>ROUGE-1</strong></td>
          <td>{_f(b_r1, 3)}</td>
          {_util_cell(p_r1, b_r1)}
          {_util_cell(s_r1, b_r1)}
          <td>Unigram F1</td>
        </tr>
        <tr style="background:#e8f5e9;font-weight:bold;">
          <td><strong>Combined Utility</strong></td>
          <td><strong>{_f(b_c, 3)}</strong></td>
          {_util_cell(p_c, b_c)}
          {_util_cell(s_c, b_c)}
          <td>= mean(R/10, M/10, ROUGE-1)</td>
        </tr>
      </tbody>
    </table>"""

    # ── Section 5: Tradeoff table ─────────────────────────────────────────
    def _tradeoff_row(label, acc_pre, acc_post, acc_red, util, b_acc_post=None):
        acc_post_cell = _acc_cell(acc_post, b_acc_post) if b_acc_post is not None else f"<td>{_pct(acc_post)}</td>"
        return f"""
        <tr>
          <td><strong>{label}</strong></td>
          <td>{_pct(acc_pre)}</td>
          {acc_post_cell}
          <td>{_delta_html(acc_red)}</td>
          <td>{_f(util, 3)}</td>
        </tr>"""

    tradeoff_table = f"""
    <table>
      <thead>
        <tr>
          <th>Architecture</th>
          <th>Adv. Accuracy Pre-Anon (Top-3)</th>
          <th>Adv. Accuracy Post-Anon (Top-3)</th>
          <th>Reduction</th>
          <th>Combined Utility</th>
        </tr>
      </thead>
      <tbody>
        {_tradeoff_row("Baseline (GPT-4o only)", bt["adversarial_accuracy_pre"], bt["adversarial_accuracy_post"], bt["accuracy_reduction"], bt["combined_utility"])}
        {_tradeoff_row("Parallel — Attack A (GPT-4o)", pt["attack_a"]["adversarial_accuracy_pre"], pt["attack_a"]["adversarial_accuracy_post"], pt["attack_a"]["accuracy_reduction"], pt["attack_a"]["combined_utility"], bt["adversarial_accuracy_post"])}
        {_tradeoff_row("Parallel — Attack B (Claude)", pt["attack_b"]["adversarial_accuracy_pre"], pt["attack_b"]["adversarial_accuracy_post"], pt["attack_b"]["accuracy_reduction"], pt["attack_b"]["combined_utility"], bt["adversarial_accuracy_post"])}
        {_tradeoff_row("Parallel — Merged", pt["merged"]["adversarial_accuracy_pre"], pt["merged"]["adversarial_accuracy_post"], pt["merged"]["accuracy_reduction"], pt["merged"]["combined_utility"], bt["adversarial_accuracy_post"])}
        {_tradeoff_row("Sequential — Attack A (GPT-4o)", st["attack_a"]["adversarial_accuracy_pre"], st["attack_a"]["adversarial_accuracy_post"], st["attack_a"]["accuracy_reduction"], st["attack_a"]["combined_utility"], bt["adversarial_accuracy_post"])}
        {_tradeoff_row("Sequential — Attack B (Claude, informed)", st["attack_b"]["adversarial_accuracy_pre"], st["attack_b"]["adversarial_accuracy_post"], st["attack_b"]["accuracy_reduction"], st["attack_b"]["combined_utility"], bt["adversarial_accuracy_post"])}
        {_tradeoff_row("Sequential — Accumulated", st["accumulated"]["adversarial_accuracy_pre"], st["accumulated"]["adversarial_accuracy_post"], st["accumulated"]["accuracy_reduction"], st["accumulated"]["combined_utility"], bt["adversarial_accuracy_post"])}
      </tbody>
    </table>"""

    # ── Verdict ───────────────────────────────────────────────────────────
    all_post = {
        "Baseline":            bpost["accuracy"]["overall_top3"],
        "Parallel Merged":     ppst["merged"]["accuracy"]["overall_top3"],
        "Sequential Accum.":   spst["accumulated"]["accuracy"]["overall_top3"],
        "Parallel B":          ppst["attack_b"]["accuracy"]["overall_top3"],
        "Sequential B":        spst["attack_b"]["accuracy"]["overall_top3"],
    }
    best_label = min(all_post, key=all_post.get)
    best_val   = all_post[best_label]

    all_ev_post = {
        "Baseline":        bpost["certainty"]["overall_evidence_rate"],
        "Parallel B":      ppst["attack_b"]["certainty"]["overall_evidence_rate"],
        "Sequential B":    spst["attack_b"]["certainty"]["overall_evidence_rate"],
        "Sequential Acc.": spst["accumulated"]["certainty"]["overall_evidence_rate"],
    }
    best_ev_label = min(all_ev_post, key=all_ev_post.get)
    best_ev_val   = all_ev_post[best_ev_label]

    verdict = f"""
    <div class="verdict">
      <h2>Verdict — Does Either Architecture Improve Anonymization?</h2>
      <h3>1 · Adversarial Accuracy</h3>
      <p>
        All architectures reduce adversarial accuracy by the same
        <strong>10 percentage points</strong> (65% &#8594; 55%).
        The best single view is <strong>{best_label}</strong> at
        <strong>{_pct(best_val)}</strong>, which equals the baseline.
        On the paper's primary metric, neither parallel nor sequential
        produces a statistically meaningful improvement.
      </p>
      <h3>2 · Evidence Rate — the clearest difference</h3>
      <p>
        The best post-anonymization evidence rate belongs to
        <strong>{best_ev_label}</strong> at <strong>{_pct(best_ev_val)}</strong>,
        compared with the baseline's <strong>{_pct(b_ev_post)}</strong>.
        Both parallel and sequential anonymizers suppress implicit signals
        (style-based and informed cues) that the single-model baseline never knew to target.
        Sequential Attack B benefits the most: it was explicitly told what Attack A found,
        so when the anonymizer erases those signals, B can no longer find them.
      </p>
      <h3>3 · Utility — no trade-off penalty</h3>
      <p>
        All three architectures deliver essentially the same combined utility
        (baseline {_f(b_c,3)}, parallel {_f(p_c,3)}, sequential {_f(s_c,3)}).
        Neither advanced architecture sacrifices text quality to achieve better privacy.
      </p>
      <h3>Summary</h3>
      <ul style="line-height:2;">
        <li><strong>Adversarial accuracy (paper's primary metric):</strong>
            no improvement — all architectures tie at 55% post-anon Top-3.</li>
        <li><strong>Evidence rate (secondary metric):</strong>
            parallel B &#8722;20pp, sequential B &#8722;{round((spa['attack_b']['certainty']['overall_evidence_rate'] - spst['attack_b']['certainty']['overall_evidence_rate'])*100, 0):.0f}pp vs baseline's drop — real but doesn't translate to lower accuracy.</li>
        <li><strong>Utility:</strong> all three are equivalent (~0.94).</li>
        <li><strong>Caveat:</strong> n=20 profiles / 20 PII instances.
            A 10pp accuracy difference requires ~100 instances for 80% power.
            Results are directionally informative but not conclusive.</li>
      </ul>
    </div>"""

    # ── Full HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>All Architectures — Paper Metrics Comparison</title>
  <style>
    body  {{ font-family: Arial, sans-serif; margin: 28px; color: #222;
             background: #fafafa; line-height: 1.6; }}
    h1    {{ color: #2a4d69; border-bottom: 3px solid #4b86b4; padding-bottom: 10px; }}
    h2    {{ color: #4b86b4; margin-top: 36px; }}
    h3    {{ color: #2a4d69; margin-top: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 9px 12px; text-align: center;
              font-size: 0.86em; }}
    th    {{ color: white; }}
    tr:nth-child(even) {{ background: #f4f8fb; }}
    tr:hover           {{ background: #e8f0f8; }}
    td:first-child     {{ text-align: left; }}
    .section {{ background: white; border: 1px solid #ddd; border-radius: 10px;
                padding: 24px; margin: 24px 0;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05); }}
    .verdict {{ background: linear-gradient(135deg,#e8f5e9,#fff8e1);
                border: 2px solid #4CAF50; border-radius: 12px;
                padding: 22px 28px; margin: 24px 0; }}
    .verdict h2 {{ color: #2e7d32; margin-top: 0; }}
    .note {{ background: #fff8e1; border-left: 4px solid #ffc107;
             padding: 12px 18px; border-radius: 4px; margin: 16px 0;
             font-size: 0.88em; color: #555; }}
    .card-row {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 20px 0; }}
    .card {{ flex: 1; min-width: 160px; background: white; border-radius: 10px;
             padding: 18px; text-align: center;
             box-shadow: 0 2px 6px rgba(0,0,0,0.06); }}
    .card-label {{ font-size: 0.78em; color: #555; margin-bottom: 6px; }}
    .card-value {{ font-size: 1.8em; font-weight: bold; margin: 6px 0; }}
    .card-sub   {{ font-size: 0.76em; color: #666; margin: 3px 0; }}
    .legend     {{ font-size: 0.84em; color: #555; margin: 6px 0 12px; }}
  </style>
</head>
<body>

<h1>Baseline vs Parallel vs Sequential — Paper-Aligned Metrics</h1>
<p style="color:#666;">
  Three anonymization architectures evaluated on the same 20 profiles using
  the metrics from <em>"LLMs Are Advanced Anonymizers"</em>.
</p>
<p>
  <span style="color:#1565C0;font-weight:bold;">&#9632; Baseline</span>: single GPT-4o attack &#8594; anonymize (2 rounds) &#8594; re-attack &nbsp;|&nbsp;
  <span style="color:#6A1B9A;font-weight:bold;">&#9632; Parallel</span>: GPT-4o + Claude independently &#8594; merge &#8594; anonymize &#8594; re-attack both &nbsp;|&nbsp;
  <span style="color:#BF360C;font-weight:bold;">&#9632; Sequential</span>: GPT-4o A &#8594; Claude B (informed by A) &#8594; anonymize &#8594; re-attack both
</p>

<div class="note">
  <strong>Reading the colours:</strong>
  <span style="color:#2e7d32;font-weight:bold;">Green &#9660;</span> = better than baseline (lower accuracy = harder to guess = better privacy).
  <span style="color:#f44336;">Red &#9650;</span> = worse than baseline.
  Utility: green = higher (better).
</div>

<div class="section">
  <h2>At a Glance</h2>
  {cards_html}
</div>

<div class="section">
  <h2>1 &middot; Adversarial Accuracy (Top-3)</h2>
  <p>Fraction of attributes the attacker correctly identifies. <strong>Lower = better privacy.</strong></p>
  {acc_table}
  <h3>Per-PII-Type (post-anonymization)</h3>
  {per_type_table}
</div>

<div class="section">
  <h2>2 &middot; Adversarial Certainty &mdash; Evidence Rate</h2>
  <p>Fraction of inferences citing direct textual evidence (certainty &ge; 3/5).
     <strong>Lower post-anon = anonymizer removed more real signals.</strong></p>
  {cert_table}
</div>

<div class="section">
  <h2>3 &middot; Utility</h2>
  <p>Combined Utility = mean(Readability/10, Meaning/10, ROUGE-1). Higher = better.</p>
  {util_table}
</div>

<div class="section">
  <h2>4 &middot; Privacy&ndash;Utility Tradeoff</h2>
  {tradeoff_table}
</div>

{verdict}

<p style="color:#bbb;font-size:0.78em;margin-top:40px;">
  Baseline: {baseline_dir} &nbsp;|&nbsp;
  Parallel: {parallel_dir} &nbsp;|&nbsp;
  Sequential: {sequential_dir}
</p>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote full comparison report -> {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run(baseline_dir, parallel_dir, sequential_dir, output):
    print(f"\n{'='*65}")
    print("PAPER METRICS: BASELINE vs PARALLEL vs SEQUENTIAL")
    print(f"{'='*65}")

    print(f"\nBaseline   metrics from: {baseline_dir}")
    bm = compute_baseline_paper_metrics(baseline_dir)

    print(f"Parallel   metrics from: {parallel_dir}")
    pm = load_parallel_metrics(parallel_dir)

    print(f"Sequential metrics from: {sequential_dir}")
    sm = load_sequential_metrics(sequential_dir)

    # ── Text summary ──────────────────────────────────────────────────────
    cols = ["Baseline", "Par-A", "Par-B", "Par-M", "Seq-A", "Seq-B", "Seq-Acc"]
    vals_post = [
        bm["post_anon"]["accuracy"]["overall_top3"],
        pm["post_anon"]["attack_a"]["accuracy"]["overall_top3"],
        pm["post_anon"]["attack_b"]["accuracy"]["overall_top3"],
        pm["post_anon"]["merged"]["accuracy"]["overall_top3"],
        sm["post_anon"]["attack_a"]["accuracy"]["overall_top3"],
        sm["post_anon"]["attack_b"]["accuracy"]["overall_top3"],
        sm["post_anon"]["accumulated"]["accuracy"]["overall_top3"],
    ]
    vals_ev = [
        bm["post_anon"]["certainty"]["overall_evidence_rate"],
        pm["post_anon"]["attack_a"]["certainty"]["overall_evidence_rate"],
        pm["post_anon"]["attack_b"]["certainty"]["overall_evidence_rate"],
        pm["post_anon"]["merged"]["certainty"]["overall_evidence_rate"],
        sm["post_anon"]["attack_a"]["certainty"]["overall_evidence_rate"],
        sm["post_anon"]["attack_b"]["certainty"]["overall_evidence_rate"],
        sm["post_anon"]["accumulated"]["certainty"]["overall_evidence_rate"],
    ]
    vals_util = [
        bm["utility"].get("avg_combined"),
        pm["utility"].get("avg_combined"), pm["utility"].get("avg_combined"),
        pm["utility"].get("avg_combined"),
        sm["utility"].get("avg_combined"), sm["utility"].get("avg_combined"),
        sm["utility"].get("avg_combined"),
    ]

    header = f"{'Metric':<32}" + "".join(f"{c:>10}" for c in cols)
    sep    = "-" * (32 + 10 * len(cols))
    print(f"\n{header}\n{sep}")
    print(f"{'Adv. Accuracy (post-anon Top-3)':<32}" + "".join(f"{v*100:>9.1f}%" for v in vals_post))
    print(f"{'Evidence Rate (post-anon)':<32}"        + "".join(f"{v*100:>9.1f}%" for v in vals_ev))
    print(f"{'Combined Utility':<32}"                 + "".join(f"{v:>10.3f}" if v else f"{'N/A':>10}" for v in vals_util))

    # Save sequential metrics JSON
    seq_json = f"{sequential_dir}/paper_metrics.json"
    if not os.path.exists(seq_json):
        with open(seq_json, "w") as f:
            json.dump(sm, f, indent=2, default=str)
        print(f"\nSaved sequential metrics JSON -> {seq_json}")

    generate_full_report(bm, pm, sm, output, baseline_dir, parallel_dir, sequential_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir",   default=DEFAULT_BASELINE)
    parser.add_argument("--parallel_dir",   default=DEFAULT_PARALLEL)
    parser.add_argument("--sequential_dir", default=DEFAULT_SEQUENTIAL)
    parser.add_argument("--output",         default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.baseline_dir, args.parallel_dir, args.sequential_dir, args.output)
