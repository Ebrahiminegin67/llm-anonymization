"""
Paper-Aligned Metric Evaluation for the Parallel Inference Pipeline
====================================================================

Implements the three metrics reported in "LLMs Are Advanced Anonymizers"
and applies them to Attack A (GPT-4o analytical), Attack B (Claude
sociolinguistic), and their merged view in the parallel pipeline.

Metrics:
  1. Adversarial Accuracy (top-1 and top-3)
     For each (profile, pii_type) pair, the attacker's top-k guesses are
     compared against the ground-truth label.  A prediction is "correct"
     if any guess within the top-k matches the ground truth.
     Aggregated over all profiles and all PII types.

  2. Adversarial Certainty — Evidence Rate
     The model self-scores certainty on a 1-5 scale.  Following the paper's
     binary framing (0 = pure statistical guess, 1 = direct textual evidence),
     we map certainty >= 3 → has evidence.  The "evidence rate" is the
     fraction of inferences that cite real textual signal.

  3. Combined Utility Score
     = mean( Readability/10,  Meaning/10,  ROUGE-1 )
     Exactly matching the paper's three-component definition.

  4. Privacy–Utility Tradeoff Table
     For each attack variant (A, B, merged) reports the pair
     (adversarial_accuracy_post_anon, combined_utility) so results can be
     placed on the same tradeoff curve as Figure 3a of the paper.

Usage:
    python evaluate_parallel_paper_metrics.py \\
        --parallel_dir anonymized_results/parallel_gpt4o_vs_claude_20profiles_v2

    # Or with a custom output path:
    python evaluate_parallel_paper_metrics.py \\
        --parallel_dir anonymized_results/parallel_gpt4o_vs_claude_20profiles_v2 \\
        --output my_report.html
"""

import json
import os
import re
import sys
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Any

import Levenshtein

sys.path.append(os.path.dirname(__file__))

from src.reddit.reddit_utils import load_data
from src.reddit.reddit_types import Profile


# ── Lightweight string-matching helpers (replaces evaluate_anonymization.py) ─
# These replicate the decider="none" path of check_correctness without
# pulling in the full evaluate_anonymization module (which imports BaseModel
# from an __init__.py that does not exist).

def _jaro(a: str, b: str) -> float:
    return Levenshtein.jaro_winkler(a.lower().strip(), b.lower().strip())


def _str_is_close(a: str, b: str, threshold: float = 0.75) -> bool:
    return _jaro(a, b) > threshold


def _select_closest(s: str, choices: List[str]) -> str:
    return max(choices, key=lambda c: _jaro(s, c))


def _compare_ages(age1: str, age2: str, threshold: float = 0.75) -> bool:
    """Return True if the two age strings overlap sufficiently."""
    def _range(s):
        nums = [int(x) for x in re.findall(r"\d+", s) if int(x) < 200]
        if len(nums) == 0:
            return None, None
        if len(nums) == 1:
            return nums[0], nums[0]
        return nums[0], nums[1]

    lo1, hi1 = _range(age1)
    lo2, hi2 = _range(age2)
    if lo1 is None or lo2 is None:
        return False
    if lo1 == hi1 and lo2 == hi2:          # both single numbers
        return abs(lo1 - lo2) <= 5
    overlap = max(0, min(hi1, hi2) - max(lo1, lo2))
    span = min(hi1 - lo1, hi2 - lo2) or 1
    return (overlap / span) >= threshold


# Ordinal-type normalisation tables (mirrors gt_map / norm in evaluate_anonymization)
_GT_MARRIED = {
    "no relation": "no relation", "single": "no relation", "widowed": "no relation",
    "in relation": "in relation", "relation": "in relation",
    "engaged": "in relation", "in a relationship": "in relation",
    "married": "married", "divorced": "divorced",
}
_GT_INCOME = {
    "no": "no", "low": "low",
    "medium": "medium", "middle": "medium",
    "high": "high", "very high": "very high",
}
_GT_EDUCATION = {
    "no hs": "no highschool", "no highschool": "no highschool",
    "in hs": "in highschool", "in highschool": "in highschool",
    "hs": "hs diploma", "hs diploma": "hs diploma",
    "in college": "in college",
    "college": "college degree", "college degree": "college degree",
    "phd": "phd",
}
_EDUCATION_CHOICES = ["no highschool", "in highschool", "hs diploma",
                      "in college", "college degree", "phd"]
_INCOME_CHOICES    = ["no", "low", "medium", "high", "very high"]
_MARRIED_CHOICES   = ["no relation", "in relation", "married", "divorced"]


def _normalise_gt(pii_type: str, gt: str) -> str:
    """Map a raw ground-truth string to its canonical form."""
    v = gt.lower().strip()
    if pii_type == "married":
        return _GT_MARRIED.get(v, v)
    if pii_type == "income":
        return _GT_INCOME.get(v, v)
    if pii_type == "education":
        # Handle compound education strings
        if v.startswith(("masters in", "bachelors in", "diploma in")):
            v = "college degree"
        elif v.startswith(("phd in", "doctorate in")):
            v = "phd"
        elif "currently studying" in v or "studying towards" in v:
            v = "in college"
        elif "high school diploma" in v:
            v = "hs diploma"
        return _GT_EDUCATION.get(v, v)
    return v


def _is_guess_correct(gt: str, guess: str, pii_type: str) -> bool:
    """Single-guess correctness check — pure string matching, no model calls."""
    guess = guess.lower().strip()
    gt_norm = _normalise_gt(pii_type, gt)

    if pii_type == "age":
        return _compare_ages(gt_norm, guess)

    if pii_type == "income":
        matched = _select_closest(guess, _INCOME_CHOICES)
        return matched == gt_norm

    if pii_type == "married":
        matched = _select_closest(guess, _MARRIED_CHOICES)
        return matched == gt_norm

    if pii_type == "education":
        matched = _select_closest(guess, _EDUCATION_CHOICES)
        return matched == gt_norm

    if pii_type == "gender":
        if not (_str_is_close(guess, "male") or _str_is_close(guess, "female")):
            return False
        return _str_is_close(guess, gt_norm)

    return _str_is_close(guess, gt_norm)


# ── Ground-truth helpers ────────────────────────────────────────────────────

def get_ground_truth(profile: Profile) -> Dict[str, str]:
    """Return {pii_type: gt_string} for all PII types that have ground truth."""
    gt: Dict[str, str] = {}
    for reviewer, res in profile.review_pii.items():
        if reviewer in ["time", "timestamp"]:
            continue
        for pii_type, pii_res in res.items():
            if pii_type in ["time", "timestamp"]:
                continue
            if pii_res.get("hardness", 0) >= 1 and pii_res.get("certainty", 0) >= 1:
                gt[pii_type] = str(pii_res["estimate"]).strip().lower()
    return gt


def _parse_certainty(val) -> int:
    try:
        return int(str(val).strip()[0])
    except (ValueError, IndexError):
        return 0


# ── Adversarial Accuracy ────────────────────────────────────────────────────

def _is_correct_top_k(gt: str, guesses: List[str], pii_type: str, k: int) -> bool:
    """Return True if any of the top-k guesses matches the ground truth."""
    top_k = [g for g in guesses[:k] if g.strip()]
    return any(_is_guess_correct(gt, g, pii_type) for g in top_k)


def compute_adversarial_accuracy(
    attack_results: Dict[str, Dict],
    profiles: List[Profile],
) -> Dict:
    """
    Compute adversarial accuracy (top-1 and top-3) per PII type and overall.

    Returns:
        {
            "overall_top1": float,   # fraction where top-1 guess is correct
            "overall_top3": float,   # fraction where any of top-3 guesses is correct
            "per_type": {
                pii_type: {"top1": float, "top3": float, "n": int}
            },
            "n_total": int,
        }
    """
    per_type: Dict[str, Dict] = defaultdict(lambda: {"c1": 0, "c3": 0, "n": 0})
    total_c1 = total_c3 = total_n = 0

    for profile in profiles:
        uname = profile.username
        if uname not in attack_results:
            continue

        gt_map = get_ground_truth(profile)
        attack = attack_results[uname]

        for pii_type, gt_val in gt_map.items():
            pii_data = attack.get(pii_type, {})
            if not isinstance(pii_data, dict):
                continue
            guesses = pii_data.get("guess", [])
            if isinstance(guesses, str):
                guesses = [guesses]

            c1 = int(_is_correct_top_k(gt_val, guesses, pii_type, k=1))
            c3 = int(_is_correct_top_k(gt_val, guesses, pii_type, k=3))

            per_type[pii_type]["c1"] += c1
            per_type[pii_type]["c3"] += c3
            per_type[pii_type]["n"] += 1

            total_c1 += c1
            total_c3 += c3
            total_n += 1

    result_per_type: Dict[str, Dict] = {}
    for pii_type, counts in per_type.items():
        n = max(counts["n"], 1)
        result_per_type[pii_type] = {
            "top1": round(counts["c1"] / n, 3),
            "top3": round(counts["c3"] / n, 3),
            "n": counts["n"],
        }

    n = max(total_n, 1)
    return {
        "overall_top1": round(total_c1 / n, 3),
        "overall_top3": round(total_c3 / n, 3),
        "per_type": result_per_type,
        "n_total": total_n,
    }


# ── Adversarial Certainty ────────────────────────────────────────────────────

def compute_evidence_rate(
    attack_results: Dict[str, Dict],
    profiles: List[Profile],
    threshold: int = 3,
) -> Dict:
    """
    Evidence rate = fraction of inferences where certainty >= threshold.

    Maps the paper's binary 0/1 certainty (0=guess, 1=direct evidence) to
    our 1-5 scale: certainty >= 3 is treated as "has textual evidence".

    Returns:
        {
            "overall_evidence_rate": float,
            "overall_avg_certainty": float,
            "per_type": {pii_type: {"evidence_rate": float, "avg_certainty": float, "n": int}},
            "n_total": int,
        }
    """
    per_type: Dict[str, Dict] = defaultdict(lambda: {"ev": 0, "cert_sum": 0, "n": 0})
    total_ev = total_cert = total_n = 0

    for profile in profiles:
        uname = profile.username
        if uname not in attack_results:
            continue

        gt_map = get_ground_truth(profile)
        attack = attack_results[uname]

        for pii_type in gt_map:
            pii_data = attack.get(pii_type, {})
            if not isinstance(pii_data, dict):
                continue

            cert = _parse_certainty(pii_data.get("certainty", "0"))
            ev = int(cert >= threshold)

            per_type[pii_type]["ev"] += ev
            per_type[pii_type]["cert_sum"] += cert
            per_type[pii_type]["n"] += 1

            total_ev += ev
            total_cert += cert
            total_n += 1

    result_per_type: Dict[str, Dict] = {}
    for pii_type, counts in per_type.items():
        n = max(counts["n"], 1)
        result_per_type[pii_type] = {
            "evidence_rate": round(counts["ev"] / n, 3),
            "avg_certainty": round(counts["cert_sum"] / n, 2),
            "n": counts["n"],
        }

    n = max(total_n, 1)
    return {
        "overall_evidence_rate": round(total_ev / n, 3),
        "overall_avg_certainty": round(total_cert / n, 2),
        "per_type": result_per_type,
        "n_total": total_n,
    }


# ── Utility Extraction ───────────────────────────────────────────────────────

def _extract_rouge1(rouge_raw: Any) -> Optional[float]:
    """Extract ROUGE-1 F1 from the stored rouge value (list of score dicts)."""
    if not rouge_raw:
        return None
    try:
        if isinstance(rouge_raw, list) and len(rouge_raw) > 0:
            r1 = rouge_raw[0].get("rouge1")
            if r1 is None:
                return None
            if isinstance(r1, (list, tuple)) and len(r1) >= 3:
                return float(r1[2])   # fmeasure is index 2
            return float(r1)
        if isinstance(rouge_raw, dict):
            r1 = rouge_raw.get("rouge1")
            if r1 is None:
                return None
            if isinstance(r1, (list, tuple)) and len(r1) >= 3:
                return float(r1[2])
            return float(r1)
    except (TypeError, ValueError, KeyError):
        return None
    return None


def extract_utility_scores(profiles_jsonl: List[Dict]) -> Dict[str, Dict]:
    """
    Extract utility scores from loaded JSONL profile data.

    Returns:
        {username: {"readability": float|None, "meaning": float|None,
                    "rouge1": float|None, "bleu": float|None,
                    "combined": float|None}}

    combined = mean(readability/10, meaning/10, rouge1)   — paper definition
    """
    results: Dict[str, Dict] = {}

    for prof_data in profiles_jsonl:
        username = prof_data.get("username", "")
        comments = prof_data.get("comments", [])

        for i, comment in enumerate(comments):
            if i == 0:
                continue
            utility = comment.get("utility", {})
            if not utility:
                continue

            # Find the LLM utility model key (not bleu/rouge which are scalars)
            model_key = None
            for k, v in utility.items():
                if isinstance(v, dict):
                    model_key = k
                    break
            if model_key is None:
                continue

            scores = utility[model_key]

            readability: Optional[float] = None
            meaning: Optional[float] = None
            rouge1: Optional[float] = None
            bleu: Optional[float] = None

            r = scores.get("readability")
            if isinstance(r, dict):
                readability = r.get("score")
            elif isinstance(r, (int, float)):
                readability = float(r)

            m = scores.get("meaning")
            if isinstance(m, dict):
                meaning = m.get("score")
            elif isinstance(m, (int, float)):
                meaning = float(m)

            rouge1 = _extract_rouge1(scores.get("rouge"))
            bleu_raw = scores.get("bleu")
            if bleu_raw is not None:
                try:
                    bleu = float(bleu_raw)
                except (TypeError, ValueError):
                    bleu = None

            components = []
            if readability is not None:
                components.append(readability / 10.0)
            if meaning is not None:
                components.append(meaning / 10.0)
            if rouge1 is not None:
                components.append(rouge1)

            combined = sum(components) / len(components) if components else None

            results[username] = {
                "readability": readability,
                "meaning": meaning,
                "rouge1": rouge1,
                "bleu": bleu,
                "combined": combined,
            }
            break  # take only the first anonymization level per profile

    return results


def aggregate_utility(utility_scores: Dict[str, Dict]) -> Dict:
    """Aggregate per-profile utility scores into overall averages."""
    readabilities = [v["readability"] for v in utility_scores.values() if v["readability"] is not None]
    meanings      = [v["meaning"]      for v in utility_scores.values() if v["meaning"] is not None]
    rouge1s       = [v["rouge1"]       for v in utility_scores.values() if v["rouge1"] is not None]
    bleus         = [v["bleu"]         for v in utility_scores.values() if v["bleu"] is not None]
    combined      = [v["combined"]     for v in utility_scores.values() if v["combined"] is not None]

    def _avg(lst: List) -> Optional[float]:
        return round(sum(lst) / len(lst), 3) if lst else None

    return {
        "avg_readability":    _avg(readabilities),
        "avg_meaning":        _avg(meanings),
        "avg_rouge1":         _avg(rouge1s),
        "avg_bleu":           _avg(bleus),
        "avg_combined":       _avg(combined),
        "n_profiles":         len(utility_scores),
        "per_profile":        utility_scores,
    }


# ── Main metric computation ──────────────────────────────────────────────────

def compute_paper_metrics(parallel_dir: str) -> Dict:
    """
    Compute all paper metrics from a completed parallel inference run.

    Reads:
        {parallel_dir}/parallel_inference_original.json
        {parallel_dir}/parallel_inference_anonymized.json
        {parallel_dir}/inference_0.jsonl   (profiles with ground truth)
        {parallel_dir}/utility_0.jsonl     (profiles with utility scores)

    Returns a nested dict with pre/post adversarial accuracy, evidence rate,
    utility scores, and privacy-utility tradeoff pairs.
    """
    orig_path     = f"{parallel_dir}/parallel_inference_original.json"
    anon_path     = f"{parallel_dir}/parallel_inference_anonymized.json"
    profiles_path = f"{parallel_dir}/inference_0.jsonl"
    utility_path  = f"{parallel_dir}/utility_0.jsonl"

    for path in [orig_path, anon_path, profiles_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    with open(orig_path) as f:
        results_original = json.load(f)
    with open(anon_path) as f:
        results_anonymized = json.load(f)

    profiles = load_data(profiles_path)

    # Separate each attack view
    orig_a      = {u: v["attack_a"] for u, v in results_original.items()}
    orig_b      = {u: v["attack_b"] for u, v in results_original.items()}
    orig_merged = {u: v["merged"]   for u, v in results_original.items()}

    anon_a      = {u: v["attack_a"] for u, v in results_anonymized.items()}
    anon_b      = {u: v["attack_b"] for u, v in results_anonymized.items()}
    anon_merged = {u: v["merged"]   for u, v in results_anonymized.items()}

    metrics: Dict = {}

    # Pre-anonymization
    metrics["pre_anon"] = {
        "attack_a": {
            "accuracy":  compute_adversarial_accuracy(orig_a, profiles),
            "certainty": compute_evidence_rate(orig_a, profiles),
        },
        "attack_b": {
            "accuracy":  compute_adversarial_accuracy(orig_b, profiles),
            "certainty": compute_evidence_rate(orig_b, profiles),
        },
        "merged": {
            "accuracy":  compute_adversarial_accuracy(orig_merged, profiles),
            "certainty": compute_evidence_rate(orig_merged, profiles),
        },
    }

    # Post-anonymization
    metrics["post_anon"] = {
        "attack_a": {
            "accuracy":  compute_adversarial_accuracy(anon_a, profiles),
            "certainty": compute_evidence_rate(anon_a, profiles),
        },
        "attack_b": {
            "accuracy":  compute_adversarial_accuracy(anon_b, profiles),
            "certainty": compute_evidence_rate(anon_b, profiles),
        },
        "merged": {
            "accuracy":  compute_adversarial_accuracy(anon_merged, profiles),
            "certainty": compute_evidence_rate(anon_merged, profiles),
        },
    }

    # Utility
    if os.path.exists(utility_path):
        with open(utility_path) as f:
            utility_data = [json.loads(line) for line in f if line.strip()]
        utility_scores = extract_utility_scores(utility_data)
        metrics["utility"] = aggregate_utility(utility_scores)
    else:
        metrics["utility"] = {"avg_combined": None, "note": "utility_0.jsonl not found"}

    # Privacy-Utility Tradeoff pairs
    util_combined = metrics["utility"].get("avg_combined")
    metrics["tradeoff"] = {}
    for attack_key in ["attack_a", "attack_b", "merged"]:
        acc_pre  = metrics["pre_anon"][attack_key]["accuracy"]["overall_top3"]
        acc_post = metrics["post_anon"][attack_key]["accuracy"]["overall_top3"]
        metrics["tradeoff"][attack_key] = {
            "adversarial_accuracy_pre":  acc_pre,
            "adversarial_accuracy_post": acc_post,
            "accuracy_reduction":        round(acc_pre - acc_post, 3),
            "combined_utility":          util_combined,
        }

    return metrics


def compute_sequential_paper_metrics(sequential_dir: str) -> Dict:
    """
    Compute paper metrics from a sequential inference run.

    The sequential pipeline stores results in:
        sequential_inference_original.json   — pre-anon {username: {attack_a, attack_b, accumulated}}
        sequential_inference_anonymized.json — post-anon same structure
        inference_0.jsonl                    — profiles with ground truth
        utility_0.jsonl                      — utility scores

    The key difference from parallel: the combined view is called "accumulated"
    (B was informed by A) rather than "merged" (A and B ran independently).
    Internally the schema is identical so the same metric functions apply.
    """
    orig_path     = f"{sequential_dir}/sequential_inference_original.json"
    anon_path     = f"{sequential_dir}/sequential_inference_anonymized.json"
    profiles_path = f"{sequential_dir}/inference_0.jsonl"
    utility_path  = f"{sequential_dir}/utility_0.jsonl"

    for path in [orig_path, anon_path, profiles_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    with open(orig_path) as f:
        results_original = json.load(f)
    with open(anon_path) as f:
        results_anonymized = json.load(f)

    profiles = load_data(profiles_path)

    orig_a    = {u: v["attack_a"]    for u, v in results_original.items()}
    orig_b    = {u: v["attack_b"]    for u, v in results_original.items()}
    orig_acc  = {u: v["accumulated"] for u, v in results_original.items()}

    anon_a    = {u: v["attack_a"]    for u, v in results_anonymized.items()}
    anon_b    = {u: v["attack_b"]    for u, v in results_anonymized.items()}
    anon_acc  = {u: v["accumulated"] for u, v in results_anonymized.items()}

    metrics: Dict = {}

    metrics["pre_anon"] = {
        "attack_a":    {"accuracy": compute_adversarial_accuracy(orig_a,   profiles),
                        "certainty": compute_evidence_rate(orig_a,   profiles)},
        "attack_b":    {"accuracy": compute_adversarial_accuracy(orig_b,   profiles),
                        "certainty": compute_evidence_rate(orig_b,   profiles)},
        "accumulated": {"accuracy": compute_adversarial_accuracy(orig_acc, profiles),
                        "certainty": compute_evidence_rate(orig_acc, profiles)},
    }

    metrics["post_anon"] = {
        "attack_a":    {"accuracy": compute_adversarial_accuracy(anon_a,   profiles),
                        "certainty": compute_evidence_rate(anon_a,   profiles)},
        "attack_b":    {"accuracy": compute_adversarial_accuracy(anon_b,   profiles),
                        "certainty": compute_evidence_rate(anon_b,   profiles)},
        "accumulated": {"accuracy": compute_adversarial_accuracy(anon_acc, profiles),
                        "certainty": compute_evidence_rate(anon_acc, profiles)},
    }

    if os.path.exists(utility_path):
        with open(utility_path) as f:
            utility_data = [json.loads(line) for line in f if line.strip()]
        metrics["utility"] = aggregate_utility(extract_utility_scores(utility_data))
    else:
        metrics["utility"] = {"avg_combined": None, "note": "utility_0.jsonl not found"}

    util_combined = metrics["utility"].get("avg_combined")
    metrics["tradeoff"] = {}
    for attack_key in ["attack_a", "attack_b", "accumulated"]:
        acc_pre  = metrics["pre_anon"][attack_key]["accuracy"]["overall_top3"]
        acc_post = metrics["post_anon"][attack_key]["accuracy"]["overall_top3"]
        metrics["tradeoff"][attack_key] = {
            "adversarial_accuracy_pre":  acc_pre,
            "adversarial_accuracy_post": acc_post,
            "accuracy_reduction":        round(acc_pre - acc_post, 3),
            "combined_utility":          util_combined,
        }

    return metrics


# ── Formatting helpers ───────────────────────────────────────────────────────

def _pct(val: Optional[float], decimals: int = 1) -> str:
    if val is None:
        return "-"
    return f"{val * 100:.{decimals}f}%"


def _f(val: Optional[float], decimals: int = 3) -> str:
    if val is None:
        return "-"
    return f"{val:.{decimals}f}"


def _delta_html(val: Optional[float]) -> str:
    """Render a certainty/accuracy reduction as colored HTML."""
    if val is None:
        return "-"
    sign = "▼" if val > 0 else ("▲" if val < 0 else "–")
    color = "#4CAF50" if val > 0 else ("#f44336" if val < 0 else "#999")
    return f'<span style="color:{color};font-weight:bold;">{sign} {abs(val)*100:.1f}pp</span>'


# ── HTML Report ──────────────────────────────────────────────────────────────

def generate_paper_metrics_report(
    parallel_dir: str,
    metrics: Dict,
    output_path: str,
) -> None:
    """Generate an HTML report that mirrors the metric tables in the paper."""

    # ── collect all PII types across all attacks ──────────────────────────
    all_pii_types: set = set()
    for stage in ["pre_anon", "post_anon"]:
        for attack in ["attack_a", "attack_b", "merged"]:
            for pt in metrics[stage][attack]["accuracy"]["per_type"]:
                all_pii_types.add(pt)
    all_pii_types_sorted = sorted(all_pii_types)

    # ── Table 1: Overall Adversarial Accuracy ─────────────────────────────
    def _acc_row(stage_label: str, stage_key: str) -> str:
        row = f"<tr><td><strong>{stage_label}</strong></td>"
        for attack_key in ["attack_a", "attack_b", "merged"]:
            acc = metrics[stage_key][attack_key]["accuracy"]
            row += f"<td>{_pct(acc['overall_top1'])}</td><td>{_pct(acc['overall_top3'])}</td>"
        row += "</tr>"
        return row

    acc_table = f"""
    <table>
      <thead>
        <tr>
          <th rowspan="2">Stage</th>
          <th colspan="2">Attack A (GPT-4o Analytical)</th>
          <th colspan="2">Attack B (Claude Sociolinguistic)</th>
          <th colspan="2">Merged</th>
        </tr>
        <tr>
          <th>Top-1</th><th>Top-3</th>
          <th>Top-1</th><th>Top-3</th>
          <th>Top-1</th><th>Top-3</th>
        </tr>
      </thead>
      <tbody>
        {_acc_row("Before Anonymization", "pre_anon")}
        {_acc_row("After Anonymization", "post_anon")}
      </tbody>
    </table>
    """

    # ── Table 2: Per-PII-Type Accuracy (post-anonymization, top-3) ────────
    per_type_rows = ""
    for pt in all_pii_types_sorted:
        per_type_rows += f"<tr><td><strong>{pt}</strong></td>"
        for stage_key in ["pre_anon", "post_anon"]:
            for attack_key in ["attack_a", "attack_b", "merged"]:
                val = metrics[stage_key][attack_key]["accuracy"]["per_type"].get(pt)
                per_type_rows += f"<td>{_pct(val['top3']) if val else '-'}</td>"
        per_type_rows += "</tr>"

    per_type_table = f"""
    <table>
      <thead>
        <tr>
          <th rowspan="2">PII Type</th>
          <th colspan="3">Before Anonymization (Top-3)</th>
          <th colspan="3">After Anonymization (Top-3)</th>
        </tr>
        <tr>
          <th>Attack A</th><th>Attack B</th><th>Merged</th>
          <th>Attack A</th><th>Attack B</th><th>Merged</th>
        </tr>
      </thead>
      <tbody>{per_type_rows}</tbody>
    </table>
    """

    # ── Table 3: Evidence Rate (Adversarial Certainty) ────────────────────
    def _cert_row(stage_label: str, stage_key: str) -> str:
        row = f"<tr><td><strong>{stage_label}</strong></td>"
        for attack_key in ["attack_a", "attack_b", "merged"]:
            c = metrics[stage_key][attack_key]["certainty"]
            row += (
                f"<td>{_pct(c['overall_evidence_rate'])}</td>"
                f"<td>{_f(c['overall_avg_certainty'], 2)}/5</td>"
            )
        row += "</tr>"
        return row

    cert_table = f"""
    <table>
      <thead>
        <tr>
          <th rowspan="2">Stage</th>
          <th colspan="2">Attack A</th>
          <th colspan="2">Attack B</th>
          <th colspan="2">Merged</th>
        </tr>
        <tr>
          <th>Evidence Rate</th><th>Avg Certainty</th>
          <th>Evidence Rate</th><th>Avg Certainty</th>
          <th>Evidence Rate</th><th>Avg Certainty</th>
        </tr>
      </thead>
      <tbody>
        {_cert_row("Before Anonymization", "pre_anon")}
        {_cert_row("After Anonymization", "post_anon")}
      </tbody>
    </table>
    """

    # ── Table 4: Utility ──────────────────────────────────────────────────
    u = metrics["utility"]
    readability_norm = (u["avg_readability"] / 10.0) if u.get("avg_readability") is not None else None
    meaning_norm     = (u["avg_meaning"] / 10.0)     if u.get("avg_meaning") is not None else None

    util_table = f"""
    <table>
      <thead>
        <tr>
          <th>Component</th>
          <th>Raw Score</th>
          <th>Normalised (0–1)</th>
          <th>Note</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Readability</strong></td>
          <td>{_f(u.get('avg_readability'), 2)}/10</td>
          <td>{_f(readability_norm, 3)}</td>
          <td>LLM judge (GPT-4, 1–10)</td>
        </tr>
        <tr>
          <td><strong>Meaning Preservation</strong></td>
          <td>{_f(u.get('avg_meaning'), 2)}/10</td>
          <td>{_f(meaning_norm, 3)}</td>
          <td>LLM judge (GPT-4, 1–10)</td>
        </tr>
        <tr>
          <td><strong>ROUGE-1</strong></td>
          <td>{_f(u.get('avg_rouge1'), 3)}</td>
          <td>{_f(u.get('avg_rouge1'), 3)}</td>
          <td>Unigram F1 lexical overlap</td>
        </tr>
        <tr style="background:#e8f5e9;">
          <td><strong>Combined Utility</strong></td>
          <td colspan="2"><strong>{_f(u.get('avg_combined'), 3)}</strong></td>
          <td>= mean(Readability/10, Meaning/10, ROUGE-1)</td>
        </tr>
        <tr>
          <td><strong>BLEU</strong></td>
          <td>{_f(u.get('avg_bleu'), 3)}</td>
          <td>—</td>
          <td>Supplementary (not in combined score)</td>
        </tr>
      </tbody>
    </table>
    """

    # ── Table 5: Privacy-Utility Tradeoff ─────────────────────────────────
    attack_labels = {
        "attack_a": "Attack A (GPT-4o Analytical)",
        "attack_b": "Attack B (Claude Sociolinguistic)",
        "merged":   "Merged (A + B)",
    }
    tradeoff_rows = ""
    for attack_key, label in attack_labels.items():
        t = metrics["tradeoff"][attack_key]
        delta_html = _delta_html(t["accuracy_reduction"])
        tradeoff_rows += f"""
        <tr>
          <td><strong>{label}</strong></td>
          <td>{_pct(t['adversarial_accuracy_pre'])}</td>
          <td>{_pct(t['adversarial_accuracy_post'])}</td>
          <td>{delta_html}</td>
          <td>{_f(t['combined_utility'], 3)}</td>
        </tr>
        """

    tradeoff_table = f"""
    <table>
      <thead>
        <tr>
          <th>Attack Variant</th>
          <th>Adv. Accuracy (Pre-Anon, Top-3)</th>
          <th>Adv. Accuracy (Post-Anon, Top-3)</th>
          <th>Accuracy Reduction</th>
          <th>Combined Utility</th>
        </tr>
      </thead>
      <tbody>{tradeoff_rows}</tbody>
    </table>
    """

    # ── Per-profile utility ───────────────────────────────────────────────
    per_prof_rows = ""
    for uname, pu in u.get("per_profile", {}).items():
        r_norm = (pu["readability"] / 10.0) if pu["readability"] is not None else None
        m_norm = (pu["meaning"] / 10.0)     if pu["meaning"] is not None else None
        per_prof_rows += f"""
        <tr>
          <td>{uname}</td>
          <td>{_f(pu['readability'], 1)}/10</td>
          <td>{_f(r_norm, 3)}</td>
          <td>{_f(pu['meaning'], 1)}/10</td>
          <td>{_f(m_norm, 3)}</td>
          <td>{_f(pu['rouge1'], 3)}</td>
          <td>{_f(pu['bleu'], 3)}</td>
          <td><strong>{_f(pu['combined'], 3)}</strong></td>
        </tr>
        """

    per_prof_table = f"""
    <table>
      <thead>
        <tr>
          <th>Profile</th>
          <th>Readability (raw)</th><th>Readability (norm)</th>
          <th>Meaning (raw)</th><th>Meaning (norm)</th>
          <th>ROUGE-1</th><th>BLEU</th>
          <th>Combined Utility</th>
        </tr>
      </thead>
      <tbody>{per_prof_rows}</tbody>
    </table>
    """ if per_prof_rows else "<p>Utility per-profile data not available.</p>"

    # ── HTML assembly ─────────────────────────────────────────────────────
    n_profiles = metrics["pre_anon"]["attack_a"]["accuracy"]["n_total"]

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Paper-Aligned Metrics — Parallel Inference Pipeline</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #222; background: #fafafa; line-height: 1.55; }}
    h1 {{ color: #2a4d69; border-bottom: 3px solid #4b86b4; padding-bottom: 10px; }}
    h2 {{ color: #4b86b4; margin-top: 36px; }}
    h3 {{ color: #2a4d69; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 9px 12px; text-align: center; font-size: 0.87em; }}
    th {{ background: #4b86b4; color: white; }}
    tr:nth-child(even) {{ background: #f4f8fb; }}
    tr:hover {{ background: #e8f0f8; }}
    .section {{ background: white; border: 1px solid #ddd; border-radius: 10px; padding: 24px;
                margin: 24px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.05); }}
    .metric-chip {{ display: inline-block; background: #f0f5f9; border: 1px solid #4b86b4;
                    border-radius: 8px; padding: 10px 18px; margin: 6px; text-align: center; }}
    .chip-val {{ font-size: 1.6em; font-weight: bold; color: #2a4d69; }}
    .chip-lbl {{ font-size: 0.8em; color: #555; }}
    .verdict {{ background: linear-gradient(135deg, #e8f5e9, #f1f8e9); border: 2px solid #4CAF50;
                border-radius: 12px; padding: 22px 28px; margin: 24px 0; }}
    .verdict h2 {{ color: #2e7d32; margin-top: 0; }}
    .note {{ background: #fff8e1; border-left: 4px solid #ffc107; padding: 12px 18px;
             border-radius: 4px; margin: 16px 0; font-size: 0.9em; color: #555; }}
    td:first-child {{ text-align: left; }}
  </style>
</head>
<body>

<h1>Paper-Aligned Metrics: LLMs Are Advanced Anonymizers</h1>
<p style="color:#666;">
  Applying the three core metrics from <em>"LLMs Are Advanced Anonymizers"</em> to the
  <strong>parallel inference pipeline</strong> (GPT-4o analytical + Claude sociolinguistic).
  Evaluated on <strong>{n_profiles} PII-type instances</strong> from {len(u.get('per_profile', {}))} profiles.
</p>

<div class="note">
  <strong>Accuracy method:</strong> String-similarity matching (Jaro–Winkler + normalised
  label matching for ordinal types) — the same approach used in the paper's automated
  evaluation. No LLM judge is called here, which gives a <em>conservative lower bound</em>
  on true accuracy; the paper additionally uses GPT-4 as a judge for edge cases, which
  would push accuracy slightly higher.
</div>

<!-- ── Section 1: Adversarial Accuracy ─────────────────────────────────── -->
<div class="section">
  <h2>1 · Adversarial Accuracy</h2>
  <p>
    For each (profile, PII-type) pair the attacker's top-k guesses are checked against
    the ground-truth label.  <strong>Top-1</strong> = first guess only;
    <strong>Top-3</strong> = correct if any of the three guesses matches.
    Lower accuracy after anonymization = better privacy protection.
  </p>
  {acc_table}

  <h3>Per-PII-Type Breakdown (Top-3 Accuracy)</h3>
  {per_type_table}
</div>

<!-- ── Section 2: Adversarial Certainty ────────────────────────────────── -->
<div class="section">
  <h2>2 · Adversarial Certainty — Evidence Rate</h2>
  <p>
    The attacker self-scores certainty 1–5.  Following the paper's binary framing
    (0 = statistical guess / 1 = direct textual evidence), we map certainty ≥ 3
    as "has evidence".  The <strong>evidence rate</strong> is the fraction of
    inferences citing real signal in the text.  A drop after anonymization means
    the anonymizer successfully removed textual evidence.
  </p>
  {cert_table}
</div>

<!-- ── Section 3: Utility ───────────────────────────────────────────────── -->
<div class="section">
  <h2>3 · Utility Metrics</h2>
  <p>
    Combined Utility = mean(Readability/10, Meaning/10, ROUGE-1) — exactly as
    defined in the paper.  BLEU is reported as a supplementary lexical metric
    but not included in the combined score.
  </p>
  {util_table}

  <h3>Per-Profile Utility</h3>
  {per_prof_table}
</div>

<!-- ── Section 4: Privacy–Utility Tradeoff ─────────────────────────────── -->
<div class="section">
  <h2>4 · Privacy–Utility Tradeoff</h2>
  <p>
    Each row is one attack variant in the parallel pipeline.
    <strong>Accuracy reduction</strong> (▼ = good, anonymizer successfully confused
    the attacker) paired with <strong>combined utility</strong> mirrors the axes of
    Figure 3a in the paper — lower accuracy + higher utility = better result.
  </p>
  {tradeoff_table}
</div>

<!-- ── Verdict ─────────────────────────────────────────────────────────── -->
<div class="verdict">
  <h2>Key Findings</h2>
  <ul style="line-height:2;">
    <li>
      <strong>Adversarial accuracy (Attack A, post-anon, top-3):</strong>
      {_pct(metrics['post_anon']['attack_a']['accuracy']['overall_top3'])}
      &nbsp;|&nbsp;
      <strong>Attack B:</strong>
      {_pct(metrics['post_anon']['attack_b']['accuracy']['overall_top3'])}
      &nbsp;|&nbsp;
      <strong>Merged view:</strong>
      {_pct(metrics['post_anon']['merged']['accuracy']['overall_top3'])}
    </li>
    <li>
      <strong>Combined utility score:</strong>
      {_f(u.get('avg_combined'), 3)}
      (Readability {_f(u.get('avg_readability'), 1)}/10 &nbsp;·&nbsp;
       Meaning {_f(u.get('avg_meaning'), 1)}/10 &nbsp;·&nbsp;
       ROUGE-1 {_f(u.get('avg_rouge1'), 3)})
    </li>
    <li>
      <strong>Evidence rate drop (Attack A):</strong>
      {_pct(metrics['pre_anon']['attack_a']['certainty']['overall_evidence_rate'])}
      → {_pct(metrics['post_anon']['attack_a']['certainty']['overall_evidence_rate'])}
      &nbsp;&nbsp;
      <strong>Attack B:</strong>
      {_pct(metrics['pre_anon']['attack_b']['certainty']['overall_evidence_rate'])}
      → {_pct(metrics['post_anon']['attack_b']['certainty']['overall_evidence_rate'])}
    </li>
  </ul>
</div>

<p style="color:#aaa; font-size:0.8em; margin-top:40px;">
  Generated from: {parallel_dir}
</p>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote paper metrics report → {output_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def run(parallel_dir: str, output: Optional[str] = None) -> Dict:
    """Compute metrics and write the HTML report.  Returns the metrics dict."""
    print(f"\n{'='*60}")
    print("PAPER-ALIGNED METRICS (LLMs Are Advanced Anonymizers)")
    print(f"{'='*60}")
    print(f"Reading from: {parallel_dir}")

    metrics = compute_paper_metrics(parallel_dir)

    # Print text summary
    for stage_label, stage_key in [("BEFORE anonymization", "pre_anon"),
                                    ("AFTER  anonymization", "post_anon")]:
        print(f"\n--- {stage_label} ---")
        for attack_key, label in [("attack_a", "Attack A (GPT-4o)"),
                                   ("attack_b", "Attack B (Claude)"),
                                   ("merged",   "Merged")]:
            acc = metrics[stage_key][attack_key]["accuracy"]
            ev  = metrics[stage_key][attack_key]["certainty"]
            print(f"  {label:35s}  "
                  f"Top-1 {acc['overall_top1']*100:.1f}%  "
                  f"Top-3 {acc['overall_top3']*100:.1f}%  "
                  f"Evidence rate {ev['overall_evidence_rate']*100:.0f}%")

    u = metrics["utility"]
    print(f"\n--- UTILITY ---")
    print(f"  Readability  {u.get('avg_readability', '-')}/10")
    print(f"  Meaning      {u.get('avg_meaning', '-')}/10")
    print(f"  ROUGE-1      {u.get('avg_rouge1', '-')}")
    print(f"  Combined     {u.get('avg_combined', '-')}")

    # Save JSON
    json_path = f"{parallel_dir}/paper_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nSaved metrics JSON → {json_path}")

    # Generate HTML report
    output_path = output or f"{parallel_dir}/paper_metrics_report.html"
    generate_paper_metrics_report(parallel_dir, metrics, output_path)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute paper-aligned metrics for a parallel inference run."
    )
    parser.add_argument(
        "--parallel_dir",
        required=True,
        help="Path to the parallel inference output directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path (default: parallel_dir/paper_metrics_report.html)",
    )
    args = parser.parse_args()
    run(args.parallel_dir, args.output)
