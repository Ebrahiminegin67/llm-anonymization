import json
import os
import sys

LEVELS = {
    "Level 1 (Naive)": "anonymized_results/tab_level1/evaluation.json",
    "Level 2 (Intermediate)": "anonymized_results/tab_level2/evaluation.json",
    "Level 3 (CoT Expert)": "anonymized_results/tab_level3/evaluation.json",
    "Level 3_fix1 (CoT Expert)": "anonymized_results/tab_level3_fix1/evaluation.json"
}

def load_eval(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_html(evals):
    entity_types = set()
    for name, data in evals.items():
        entity_types.update(data.get("per_type_recall", {}).keys())
    entity_types = sorted(entity_types)

    level_names = list(evals.keys())

    # Build rows for main metrics
    def metric(data, *keys):
        v = data
        for k in keys:
            v = v.get(k, {}) if isinstance(v, dict) else 0
        return v

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Prompt Level Comparison</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #f4f6f9; margin: 0; padding: 40px; }}
  h1 {{ color: #2a4d69; text-align: center; }}
  h2 {{ color: #4b86b4; margin-top: 40px; }}
  .subtitle {{ text-align: center; color: #666; margin-bottom: 40px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: white;
           border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  th {{ background: #2a4d69; color: white; padding: 14px 18px; text-align: left; }}
  td {{ padding: 12px 18px; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #f0f4f8; }}
  .best {{ background: #d4edda; font-weight: bold; }}
  .bar-container {{ background: #e9ecef; border-radius: 8px; height: 24px; position: relative; }}
  .bar {{ border-radius: 8px; height: 24px; display: flex; align-items: center;
          padding-left: 8px; color: white; font-size: 0.85em; font-weight: 600;
          min-width: 40px; }}
  .bar.green {{ background: linear-gradient(90deg, #28a745, #5cb85c); }}
  .bar.blue {{ background: linear-gradient(90deg, #4b86b4, #6ca0c8); }}
  .bar.orange {{ background: linear-gradient(90deg, #fd7e14, #f5a623); }}
  .bar.purple {{ background: linear-gradient(90deg, #9b59b6, #b07cce); }}
  .legend {{ display: flex; gap: 24px; justify-content: center; margin: 20px 0; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-dot {{ width: 14px; height: 14px; border-radius: 4px; }}
  .card {{ background: white; border-radius: 10px; padding: 24px; margin: 20px 0;
           box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
</style>
</head>
<body>
<h1>Prompt Level Comparison</h1>
<p class="subtitle">TAB Dataset &mdash; GPT-4o &mdash; {list(evals.values())[0].get('num_documents_evaluated', '?')} documents</p>

<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#28a745"></div> Level 1 (Naive)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#4b86b4"></div> Level 2 (Intermediate)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#fd7e14"></div> Level 3 (CoT Expert)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#9b59b6"></div> Level 3_fix1 (CoT Expert)</div>
</div>

<h2>Overall Metrics</h2>
<table>
<tr><th>Metric</th>"""

    colors = ["green", "blue", "orange", "purple"]
    for name in level_names:
        html += f"<th>{name}</th>"
    html += "</tr>\n"

    rows = [
        ("Overall Recall", "aggregate", "overall_recall"),
        ("Word Retention", "aggregate", "avg_word_retention"),
        ("Structure Similarity", "aggregate", "avg_structure_similarity"),
        ("Entities Masked", "aggregate", "total_masked"),
        ("Entities Missed", "aggregate", "total_missed"),
    ]

    for label, *keys in rows:
        vals = [metric(evals[n], *keys) for n in level_names]
        is_pct = isinstance(vals[0], float) and vals[0] <= 1.0
        best = max(vals) if label != "Entities Missed" else min(vals)
        html += f"<tr><td><strong>{label}</strong></td>"
        for v in vals:
            cls = ' class="best"' if v == best and len(set(vals)) > 1 else ""
            display = f"{v:.1%}" if is_pct else str(v)
            html += f"<td{cls}>{display}</td>"
        html += "</tr>\n"

    html += "</table>\n"

    # Per-entity-type visual bars
    html += "<h2>Recall by Entity Type</h2>\n"
    for etype in entity_types:
        html += f'<div class="card"><h3 style="margin-top:0;color:#2a4d69">{etype}</h3>\n'
        for i, name in enumerate(level_names):
            pt = evals[name].get("per_type_recall", {}).get(etype, {})
            recall = pt.get("recall", 0)
            masked = pt.get("masked", 0)
            total = pt.get("total", 0)
            pct = recall * 100
            html += f"""<div style="margin:8px 0">
  <div style="display:flex;justify-content:space-between;font-size:0.9em;margin-bottom:2px">
    <span>{name}</span><span>{recall:.1%} ({masked}/{total})</span>
  </div>
  <div class="bar-container"><div class="bar {colors[i]}" style="width:{max(pct,5):.0f}%">{recall:.0%}</div></div>
</div>\n"""
        html += "</div>\n"

    html += """
<div style="text-align:center;color:#888;margin-top:40px;padding:20px;border-top:1px solid #ddd">
  Generated by compare_levels.py
</div>
</body></html>"""
    return html

def main():
    evals = {}
    for name, path in LEVELS.items():
        data = load_eval(path)
        if data:
            evals[name] = data
            print(f"Loaded: {name} ({path})")
        else:
            print(f"Skipped: {name} — {path} not found")

    if len(evals) < 2:
        print(f"\nOnly {len(evals)} level(s) found. Run at least 2 levels first.")
        sys.exit(1)

    html = generate_html(evals)
    out = "prompt_level_comparison_TAB.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nComparison report saved to {out}")
    print(f"Open it in your browser: start {out}")

if __name__ == "__main__":
    main()