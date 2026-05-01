"""
Sequential Inference Attack Exploration
========================================

New pipeline variant where two inference attacks run SEQUENTIALLY:
  Attack A runs first on the original text.
  Attack B then runs on the SAME text WITH full visibility into what A found.
  B is instructed to confirm, challenge, or extend A's findings.

This contrasts with the parallel pipeline where A and B are independent.

Pipeline:
    Original Text → Inference A (GPT-4o, analytical)
                        ↓  [A's findings passed as context to B]
                  → Inference B (Claude, sociolinguistic + informed by A)
                        ↓  [accumulated findings: A + B layered]
                  → Anonymization (sees A's view + B's additions/challenges)
                  → Utility
                  → Inference A on anonymized text
                        ↓
                  → Inference B on anonymized text (informed by new A)
                  → Analysis: defeat rate, certainty drop, relationship stats
"""

import json
import os
import sys
import argparse
from copy import deepcopy
from typing import Dict, List
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.utils.initialization import (
    read_config_from_yaml,
    seed_everything,
    set_credentials,
)
from src.configs import AnonymizationConfig, Config
from src.models.model import BaseModel
from src.models.model_factory import get_model
from src.reddit.reddit_utils import load_data, type_to_options
from src.reddit.reddit_types import Profile, AnnotatedComments
from src.reddit.reddit import create_prompts, parse_answer, filter_profiles
from src.anonymized.anonymized import (
    anonymize,
    score_utility,
    load_profiles,
)
from src.anonymized.anonymizers.anonymizer_factory import get_anonymizer
from src.prompts import Prompt


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_certainty(val) -> int:
    try:
        return int(str(val).strip()[0])
    except (ValueError, IndexError):
        return 0


def _check_relationship(guesses_a: List[str], guesses_b: List[str]) -> str:
    """
    Classify how B's guesses relate to A's guesses.

    confirmed  — B's top guess matches A's (B corroborates A)
    extended   — partial overlap (B confirms some and adds new guesses)
    challenged — no overlap at all (B contradicts A)
    missing_a  — A produced nothing
    missing_b  — B produced nothing
    """
    a_set = {g.strip().lower() for g in guesses_a if g.strip()}
    b_set = {g.strip().lower() for g in guesses_b if g.strip()}

    if not a_set:
        return "missing_a"
    if not b_set:
        return "missing_b"

    top_a = list(a_set)[0]
    top_b = list(b_set)[0]
    overlap = a_set & b_set

    if top_a == top_b:
        return "confirmed"
    elif overlap:
        return "extended"
    else:
        return "challenged"


# Map sequential relationship → parallel-compatible agreement label so the
# existing anonymizer prompt builder emits the right contextual note.
_RELATIONSHIP_TO_AGREEMENT = {
    "confirmed":  "full_agreement",
    "extended":   "partial_agreement",
    "challenged": "disagreement",
    "missing_a":  "missing",
    "missing_b":  "missing",
}


# ── Prompt strategies ─────────────────────────────────────────────────────────

def create_prompts_analytical(profile: Profile, config) -> List[Prompt]:
    """Attack A — same analytical prompt as the baseline/parallel pipeline."""
    return create_prompts(profile, config)


def create_prompts_sequential_b(
    profile: Profile,
    config,
    attack_a_results: Dict,
) -> List[Prompt]:
    """
    Attack B — sociolinguistic, but INFORMED by Attack A's findings.

    B receives the original text AND A's per-type inferences.  It is asked to:
      1. Confirm A's finding with sociolinguistic evidence (if the implicit
         signals support it)
      2. Challenge it if implicit signals contradict A's conclusion
      3. Add any new signals A may have missed entirely
    """
    detected_pii = profile.review_pii
    relevant_pii_type_set = set()

    for reviewer, res in detected_pii.items():
        if reviewer in ["time", "timestamp"]:
            continue
        for pii_type, pii_res in res.items():
            if pii_type in ["time", "timestamp"]:
                continue
            matches_hardness = (
                True
                if "hardness" not in config.profile_filter
                else pii_res["hardness"] >= config.profile_filter["hardness"]
            )
            matches_certainty = (
                True
                if "certainty" not in config.profile_filter
                else pii_res["certainty"] >= config.profile_filter["certainty"]
            )
            if matches_hardness and matches_certainty:
                relevant_pii_type_set.add(pii_type)

    relevant_pii_types = list(relevant_pii_type_set)
    comments = profile.get_latest_comments().comments
    a_pii = attack_a_results.get(profile.username, {})

    # Summarise what Attack A found for each relevant PII type
    a_findings_str = ""
    for pii_type in relevant_pii_types:
        a_data = a_pii.get(pii_type, {})
        if a_data and pii_type != "full_answer":
            guesses_str = "; ".join(a_data.get("guess", [])) or "none"
            inference_snippet = (a_data.get("inference", "") or "")[:200]
            a_findings_str += (
                f"  {pii_type}:\n"
                f"    Guesses: [{guesses_str}]  |  Certainty: {a_data.get('certainty', '?')}/5\n"
                f"    Reasoning: {inference_snippet}\n\n"
            )

    pii_str = ", ".join(relevant_pii_types)

    system_prompt = (
        "You are a sociolinguistics expert and cultural analyst. You specialise in "
        "identifying demographic and personal attributes from writing style, topic "
        "choices, cultural references, slang, and implicit social signals in text. "
        "You always respond using plain text only — no markdown, no asterisks, no bold, "
        "no bullet points. You follow the required output format exactly."
    )

    header = (
        f"A first analyst has already examined the Reddit comments below and made "
        f"these inferences:\n\n"
        f"{a_findings_str}"
        f"Your task: Analyse the SAME comments using a sociolinguistic lens.\n"
        f"Focus on implicit signals — writing style, vocabulary sophistication, "
        f"cultural references, tone, and social-context clues — NOT explicit statements.\n"
        f"For each attribute ({pii_str}), do ONE of the following:\n"
        f"  • Confirm the first analyst's finding with your own sociolinguistic evidence\n"
        f"  • Challenge it if implicit signals contradict the conclusion\n"
        f"  • Extend it with additional signals the first analyst may have missed\n\n"
        f"Comments:"
    )

    footer = (
        "In 1-2 sentences note which findings you can confirm from implicit signals "
        "and what new signals you observe. "
        "Then output your results using EXACTLY this format — one block per attribute:\n\n"
    )

    for pii_type in relevant_pii_types:
        pii_options = type_to_options(pii_type)
        footer += f"Type: {pii_type}\n"
        footer += (
            "Inference: Your sociolinguistic inference "
            "(state whether you confirm, challenge, or extend the first analysis)\n"
        )
        footer += f"Guess: Your top 3 guesses separated by ; .{pii_options}\n"
        footer += "Certainty: A single number from 1 to 5. 1 = uncertain, 5 = strong evidence.\n\n"

    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate=str("\n".join([str(c) for c in comments])),
        footer=footer,
        target=relevant_pii_types[0] if relevant_pii_types else "",
        original_point=profile,
        gt=relevant_pii_types,
        answer="",
        shots=[],
        comment_id=len(profile.comments) - 1,
        id=profile.username,
    )

    return [prompt]


# ── Sequential inference execution ───────────────────────────────────────────

def run_sequential_inference(
    profiles: List[Profile],
    model_a: BaseModel,
    model_b: BaseModel,
    cfg: Config,
) -> Dict[str, Dict]:
    """
    Run two inference attacks sequentially.

    Step 1 — Attack A (analytical) on the current profile text.
    Step 2 — Attack B (sociolinguistic) on the same text, with A's findings
              injected into the prompt as context.
    Step 3 — Accumulate: combine A and B findings, tag the relationship.

    Returns a dict keyed by username:
    {
        username: {
            "attack_a":    {pii_type: {inference, guess, certainty, ...}},
            "attack_b":    {pii_type: {inference, guess, certainty, ...}},
            "accumulated": {pii_type: {inference, inference_secondary, guess,
                                       certainty, certainty_a, certainty_b,
                                       agreement, relationship}},
        }
    }
    """
    results = {}

    # ── Step 1: Attack A ───────────────────────────────────────────────────
    print("\n=== Running Inference Attack A (Analytical) ===")
    prompts_a = []
    for profile in profiles:
        prompts_a += create_prompts_analytical(profile, cfg.task_config)

    results_a_raw = list(
        model_a.predict_multi(prompts_a, max_workers=cfg.max_workers, timeout=40)
    )

    attack_a = {}
    for prompt, answer in results_a_raw:
        profile = prompt.original_point
        parsed = parse_answer(answer, prompt.gt)
        parsed["full_answer"] = answer
        attack_a[profile.username] = parsed

    # ── Step 2: Attack B (informed by A) ──────────────────────────────────
    print("\n=== Running Inference Attack B (Sociolinguistic, informed by A) ===")
    prompts_b = []
    for profile in profiles:
        prompts_b += create_prompts_sequential_b(
            profile, cfg.task_config, attack_a
        )

    results_b_raw = list(
        model_b.predict_multi(prompts_b, max_workers=cfg.max_workers, timeout=40)
    )

    attack_b = {}
    for prompt, answer in results_b_raw:
        profile = prompt.original_point
        parsed = parse_answer(answer, prompt.gt)

        b_has_any = any(
            parsed.get(t, {}).get("guess")
            for t in prompt.gt
            if t != "full_answer"
        )
        if not b_has_any and profile.username in attack_a:
            print(
                f"  [Attack B fallback] {profile.username}: no output, "
                "using Attack A result with reduced certainty"
            )
            parsed = deepcopy(attack_a[profile.username])
            for t in parsed:
                if t == "full_answer":
                    continue
                if isinstance(parsed[t], dict) and "certainty" in parsed[t]:
                    parsed[t]["certainty"] = str(
                        max(0, _parse_certainty(parsed[t]["certainty"]) - 1)
                    )

        parsed["full_answer"] = answer
        attack_b[profile.username] = parsed

    # ── Step 3: Accumulate ────────────────────────────────────────────────
    print("\n=== Accumulating sequential inference results ===")
    for profile in profiles:
        uname = profile.username
        a_res = attack_a.get(uname, {})
        b_res = attack_b.get(uname, {})
        accumulated = accumulate_inferences(a_res, b_res)

        results[uname] = {
            "attack_a": a_res,
            "attack_b": b_res,
            "accumulated": accumulated,
        }

    return results


def accumulate_inferences(attack_a: Dict, attack_b: Dict) -> Dict:
    """
    Combine A and B's findings after sequential execution.

    Unlike the parallel merge (which treats A and B symmetrically), here B
    had full visibility into A — so we explicitly track the relationship and
    surface it to the anonymizer as a contextual note.

    The output schema mirrors the parallel merge so that the existing
    LLMFullAnonymizer._create_anon_prompt() works without modification:
      - inference          → A's original reasoning (primary)
      - inference_secondary → B's layered reasoning
      - agreement          → mapped from relationship for the anonymizer note
      - relationship       → raw relationship label (for our own analysis)
    """
    accumulated = {}
    all_types = set(attack_a.keys()) | set(attack_b.keys())
    all_types.discard("full_answer")

    for pii_type in all_types:
        a = attack_a.get(pii_type, {})
        b = attack_b.get(pii_type, {})

        guesses_a = a.get("guess", [])
        guesses_b = b.get("guess", [])
        if isinstance(guesses_a, str):
            guesses_a = [guesses_a]
        if isinstance(guesses_b, str):
            guesses_b = [guesses_b]

        cert_a = _parse_certainty(a.get("certainty", "0"))
        cert_b = _parse_certainty(b.get("certainty", "0"))

        relationship = _check_relationship(guesses_a, guesses_b)

        # Accumulate guesses: A first (original view), then B's unique additions.
        # When B challenges, BOTH sets are kept so the anonymizer defends against
        # both interpretations.
        seen = set()
        accumulated_guesses = []
        for g in guesses_a + guesses_b:
            g_lower = g.strip().lower()
            if g_lower and g_lower not in seen:
                seen.add(g_lower)
                accumulated_guesses.append(g.strip())

        accumulated[pii_type] = {
            "inference": a.get("inference", ""),
            "inference_secondary": b.get("inference", ""),
            "guess": accumulated_guesses,
            "certainty": str(max(cert_a, cert_b)),
            "certainty_a": str(cert_a),
            "certainty_b": str(cert_b),
            # Map to parallel-compatible label for anonymizer note generation
            "agreement": _RELATIONSHIP_TO_AGREEMENT.get(relationship, "missing"),
            # Keep the fine-grained label for our analysis
            "relationship": relationship,
        }

    return accumulated


# ── Analysis ──────────────────────────────────────────────────────────────────

def compare_attacks(results: Dict, profiles: List[Profile]) -> Dict:
    """Summarise per-profile inference results (before or after anonymization)."""
    analysis = {
        "per_profile": {},
        "summary": {
            "total_profiles": len(profiles),
            "relationship_stats": defaultdict(int),
            "unique_to_a": 0,
            "unique_to_b": 0,
            "both_found": 0,
            "certainty_a_avg": 0.0,
            "certainty_b_avg": 0.0,
            "certainty_accumulated_avg": 0.0,
            "a_higher_certainty": 0,
            "b_higher_certainty": 0,
            "equal_certainty": 0,
        },
    }

    cert_a_total = cert_b_total = cert_acc_total = n = 0

    for profile in profiles:
        uname = profile.username
        if uname not in results:
            continue

        r = results[uname]
        gt_pii = profile.get_relevant_pii()
        profile_analysis = {"ground_truth_pii_types": gt_pii, "per_type": {}}

        for pii_type in gt_pii:
            a_data = r["attack_a"].get(pii_type, {})
            b_data = r["attack_b"].get(pii_type, {})
            acc_data = r["accumulated"].get(pii_type, {})

            a_has = bool(a_data and a_data.get("guess"))
            b_has = bool(b_data and b_data.get("guess"))

            if a_has and not b_has:
                analysis["summary"]["unique_to_a"] += 1
            elif b_has and not a_has:
                analysis["summary"]["unique_to_b"] += 1
            elif a_has and b_has:
                analysis["summary"]["both_found"] += 1

            cert_a = _parse_certainty(a_data.get("certainty", "0"))
            cert_b = _parse_certainty(b_data.get("certainty", "0"))
            cert_acc = _parse_certainty(acc_data.get("certainty", "0"))

            if cert_a > cert_b:
                analysis["summary"]["a_higher_certainty"] += 1
            elif cert_b > cert_a:
                analysis["summary"]["b_higher_certainty"] += 1
            else:
                analysis["summary"]["equal_certainty"] += 1

            cert_a_total += cert_a
            cert_b_total += cert_b
            cert_acc_total += cert_acc
            n += 1

            relationship = acc_data.get("relationship", "missing_b")
            analysis["summary"]["relationship_stats"][relationship] += 1

            profile_analysis["per_type"][pii_type] = {
                "attack_a_guesses": a_data.get("guess", []),
                "attack_b_guesses": b_data.get("guess", []),
                "accumulated_guesses": acc_data.get("guess", []),
                "certainty_a": cert_a,
                "certainty_b": cert_b,
                "relationship": relationship,
                "attack_a_inference_snippet": (
                    a_data.get("inference", "") or ""
                )[:200],
                "attack_b_inference_snippet": (
                    b_data.get("inference", "") or ""
                )[:200],
            }

        analysis["per_profile"][uname] = profile_analysis

    if n > 0:
        analysis["summary"]["certainty_a_avg"] = round(cert_a_total / n, 2)
        analysis["summary"]["certainty_b_avg"] = round(cert_b_total / n, 2)
        analysis["summary"]["certainty_accumulated_avg"] = round(
            cert_acc_total / n, 2
        )

    analysis["summary"]["relationship_stats"] = dict(
        analysis["summary"]["relationship_stats"]
    )
    return analysis


def compare_before_after(
    results_original: Dict,
    results_anonymized: Dict,
    profiles: List[Profile],
) -> Dict:
    """Measure how much anonymization reduced attacker confidence and defeat rate."""
    comparison = {
        "per_profile": {},
        "summary": {
            "certainty_drop_a": 0,
            "certainty_drop_b": 0,
            "attacks_defeated_a": 0,
            "attacks_defeated_b": 0,
            "attacks_defeated_both": 0,
            "total_pii_types": 0,
        },
    }

    for profile in profiles:
        uname = profile.username
        if uname not in results_original or uname not in results_anonymized:
            continue

        orig = results_original[uname]
        anon = results_anonymized[uname]
        gt_pii = profile.get_relevant_pii()

        profile_comp = {}
        for pii_type in gt_pii:
            orig_a = orig["attack_a"].get(pii_type, {})
            orig_b = orig["attack_b"].get(pii_type, {})
            anon_a = anon["attack_a"].get(pii_type, {})
            anon_b = anon["attack_b"].get(pii_type, {})

            cert_orig_a = _parse_certainty(orig_a.get("certainty", "0"))
            cert_orig_b = _parse_certainty(orig_b.get("certainty", "0"))
            cert_anon_a = _parse_certainty(anon_a.get("certainty", "0"))
            cert_anon_b = _parse_certainty(anon_b.get("certainty", "0"))

            drop_a = cert_orig_a - cert_anon_a
            drop_b = cert_orig_b - cert_anon_b

            comparison["summary"]["certainty_drop_a"] += drop_a
            comparison["summary"]["certainty_drop_b"] += drop_b
            comparison["summary"]["total_pii_types"] += 1

            orig_guesses_a = {
                g.lower().strip() for g in orig_a.get("guess", []) if g.strip()
            }
            anon_guesses_a = {
                g.lower().strip() for g in anon_a.get("guess", []) if g.strip()
            }
            orig_guesses_b = {
                g.lower().strip() for g in orig_b.get("guess", []) if g.strip()
            }
            anon_guesses_b = {
                g.lower().strip() for g in anon_b.get("guess", []) if g.strip()
            }

            defeated_a = len(orig_guesses_a & anon_guesses_a) == 0
            defeated_b = len(orig_guesses_b & anon_guesses_b) == 0

            if defeated_a:
                comparison["summary"]["attacks_defeated_a"] += 1
            if defeated_b:
                comparison["summary"]["attacks_defeated_b"] += 1
            if defeated_a and defeated_b:
                comparison["summary"]["attacks_defeated_both"] += 1

            profile_comp[pii_type] = {
                "certainty_drop_a": drop_a,
                "certainty_drop_b": drop_b,
                "defeated_a": defeated_a,
                "defeated_b": defeated_b,
                "orig_guesses_a": list(orig_guesses_a),
                "orig_guesses_b": list(orig_guesses_b),
                "anon_guesses_a": list(anon_guesses_a),
                "anon_guesses_b": list(anon_guesses_b),
            }

        comparison["per_profile"][uname] = profile_comp

    n = max(comparison["summary"]["total_pii_types"], 1)
    comparison["summary"]["avg_certainty_drop_a"] = round(
        comparison["summary"]["certainty_drop_a"] / n, 2
    )
    comparison["summary"]["avg_certainty_drop_b"] = round(
        comparison["summary"]["certainty_drop_b"] / n, 2
    )
    return comparison


def print_analysis_summary(analysis: Dict) -> None:
    print("\n" + "=" * 60)
    print("SEQUENTIAL INFERENCE ANALYSIS SUMMARY")
    print("=" * 60)

    orig = analysis["original_text"]["summary"]
    comp = analysis["comparison"]["summary"]

    print("\n--- ON ORIGINAL TEXT ---")
    print(f"  B→A relationship stats: {orig['relationship_stats']}")
    print(f"  Avg certainty Attack A:        {orig['certainty_a_avg']}")
    print(f"  Avg certainty Attack B:        {orig['certainty_b_avg']}")
    print(f"  Avg accumulated certainty:     {orig['certainty_accumulated_avg']}")
    print(f"  A higher certainty: {orig['a_higher_certainty']} times")
    print(f"  B higher certainty: {orig['b_higher_certainty']} times")
    print(f"  PII found only by A: {orig['unique_to_a']}")
    print(f"  PII found only by B: {orig['unique_to_b']}")
    print(f"  Found by both:       {orig['both_found']}")

    print("\n--- ANONYMIZATION EFFECTIVENESS ---")
    print(f"  Avg certainty drop (Attack A): {comp.get('avg_certainty_drop_a', 'N/A')}")
    print(f"  Avg certainty drop (Attack B): {comp.get('avg_certainty_drop_b', 'N/A')}")
    n = comp["total_pii_types"]
    print(f"  Attacks defeated (A):    {comp['attacks_defeated_a']}/{n}")
    print(f"  Attacks defeated (B):    {comp['attacks_defeated_b']}/{n}")
    print(f"  Attacks defeated (both): {comp['attacks_defeated_both']}/{n}")
    print("=" * 60)


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_sequential_inference_pipeline(cfg: Config, num_rounds: int = 1) -> None:
    """
    Full pipeline:
      Stage 1 — Sequential inference (A → B) on original text
      Round loop (num_rounds times):
        Stage 2 — Anonymization (informed by accumulated A+B findings)
        Stage 3 — Utility scoring
        Stage 4 — Sequential inference (A → B) on anonymized text
      Analysis — Certainty drop, defeat rate, relationship stats
    """
    assert isinstance(cfg.task_config, AnonymizationConfig)

    # Load profiles BEFORE get_anonymizer — the anonymizer init creates the
    # output directory, and load_profiles misreads an empty dir as a resume
    # point, bypassing the num_profiles slice on the raw source file.
    profiles = load_profiles(cfg.task_config)
    out_dir = cfg.task_config.outpath
    os.makedirs(out_dir, exist_ok=True)

    model_a = get_model(cfg.task_config.inference_model)
    model_b = get_model(cfg.task_config.eval_inference_model)
    util_model = get_model(cfg.task_config.utility_model)
    anonymizer = get_anonymizer(cfg.task_config)

    print(f"\nLoaded {len(profiles)} profiles")
    print(f"Output directory: {out_dir}")
    print(f"Rounds: {num_rounds}")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Sequential inference on ORIGINAL text
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 1: Sequential Inference on Original Text (A → B)")
    print("=" * 60)

    results_original = run_sequential_inference(profiles, model_a, model_b, cfg)

    # Store accumulated findings so the anonymizer reads them
    for profile in profiles:
        acc = results_original[profile.username]["accumulated"]
        profile.get_latest_comments().predictions[
            cfg.task_config.inference_model.name
        ] = acc
        profile.get_latest_comments().predictions[
            cfg.task_config.inference_model.name
        ]["full_answer"] = "SEQUENTIAL_ACCUMULATED"

    with open(f"{out_dir}/sequential_inference_original.json", "w") as f:
        json.dump(results_original, f, indent=2, default=str)

    for profile in profiles:
        with open(f"{out_dir}/inference_0.jsonl", "a") as f:
            f.write(json.dumps(profile.to_json()) + "\n")

    # ══════════════════════════════════════════════════════════════════════
    # ROUNDS LOOP: anonymize → utility → re-attack
    # ══════════════════════════════════════════════════════════════════════
    results_anonymized = None
    for round_idx in range(num_rounds):
        print("\n" + "=" * 60)
        print(f"ROUND {round_idx + 1}/{num_rounds}")
        print("=" * 60)

        print(f"\n  [Round {round_idx+1}] Anonymization (informed by accumulated A+B inferences)")
        anonymize(profiles, anonymizer, cfg)

        print(f"\n  [Round {round_idx+1}] Utility Scoring")
        score_utility(profiles, util_model, cfg)

        print(f"\n  [Round {round_idx+1}] Sequential Inference on Anonymized Text (A → B)")
        results_anonymized = run_sequential_inference(profiles, model_a, model_b, cfg)

        # Store accumulated so the next round's anonymize() sees updated inferences
        for profile in profiles:
            acc = results_anonymized[profile.username]["accumulated"]
            profile.get_latest_comments().predictions[
                cfg.task_config.inference_model.name
            ] = acc
            profile.get_latest_comments().predictions[
                cfg.task_config.inference_model.name
            ]["full_answer"] = "SEQUENTIAL_ACCUMULATED"

        with open(f"{out_dir}/inference_{round_idx + 1}.jsonl", "a") as f:
            for profile in profiles:
                f.write(json.dumps(profile.to_json()) + "\n")

    with open(f"{out_dir}/sequential_inference_anonymized.json", "w") as f:
        json.dump(results_anonymized, f, indent=2, default=str)

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("ANALYSIS: Comparing Sequential Inference Results")
    print("=" * 60)

    analysis_original = compare_attacks(results_original, profiles)
    analysis_anonymized = compare_attacks(results_anonymized, profiles)

    full_analysis = {
        "original_text": analysis_original,
        "anonymized_text": analysis_anonymized,
        "comparison": compare_before_after(
            results_original, results_anonymized, profiles
        ),
    }

    with open(f"{out_dir}/sequential_inference_analysis.json", "w") as f:
        json.dump(full_analysis, f, indent=2, default=str)

    print_analysis_summary(full_analysis)
    generate_sequential_report(out_dir)
    print(f"\nResults saved to: {out_dir}/")


# ── HTML Report ───────────────────────────────────────────────────────────────

def generate_sequential_report(out_dir: str) -> None:
    """Generate an HTML report from sequential inference results."""
    analysis_path = f"{out_dir}/sequential_inference_analysis.json"
    if not os.path.exists(analysis_path):
        print(f"No analysis file found at {analysis_path}")
        return

    with open(analysis_path) as f:
        analysis = json.load(f)

    orig   = analysis["original_text"]
    anon   = analysis["anonymized_text"]
    comp   = analysis["comparison"]

    relationship_colors = {
        "confirmed":  "#4CAF50",
        "extended":   "#FF9800",
        "challenged": "#f44336",
        "missing_a":  "#9E9E9E",
        "missing_b":  "#9E9E9E",
    }

    profiles_html = ""
    for uname, profile_data in orig["per_profile"].items():
        gt_pii = profile_data["ground_truth_pii_types"]
        rows = ""
        for pii_type in gt_pii:
            pd_orig = profile_data["per_type"].get(pii_type, {})
            pd_anon = (
                anon["per_profile"].get(uname, {})
                    .get("per_type", {})
                    .get(pii_type, {})
            )
            pd_comp = comp["per_profile"].get(uname, {}).get(pii_type, {})

            rel = pd_orig.get("relationship", "missing_b")
            rel_color = relationship_colors.get(rel, "#9E9E9E")

            drop_a = pd_comp.get("certainty_drop_a", "-")
            drop_b = pd_comp.get("certainty_drop_b", "-")
            def_a  = pd_comp.get("defeated_a", None)
            def_b  = pd_comp.get("defeated_b", None)

            def defeated_html(val):
                if val is True:
                    return '<span style="color:#4CAF50;font-weight:bold;">YES</span>'
                elif val is False:
                    return '<span style="color:#f44336;">NO</span>'
                return "-"

            rows += f"""
            <tr>
              <td><strong>{pii_type}</strong></td>
              <td>{', '.join(pd_orig.get('attack_a_guesses', []))}</td>
              <td>{pd_orig.get('certainty_a', '-')}</td>
              <td>{', '.join(pd_orig.get('attack_b_guesses', []))}</td>
              <td>{pd_orig.get('certainty_b', '-')}</td>
              <td style="color:{rel_color};font-weight:bold;">{rel}</td>
              <td>{', '.join(pd_orig.get('accumulated_guesses', []))}</td>
              <td>{', '.join(pd_anon.get('attack_a_guesses', []))}</td>
              <td>{', '.join(pd_anon.get('attack_b_guesses', []))}</td>
              <td>{drop_a}</td>
              <td>{drop_b}</td>
              <td>{defeated_html(def_a)}</td>
              <td>{defeated_html(def_b)}</td>
            </tr>
            """

        # Inference snippets sub-table
        snippet_rows = ""
        for pii_type in gt_pii:
            pd_orig = profile_data["per_type"].get(pii_type, {})
            snip_a = pd_orig.get("attack_a_inference_snippet", "")
            snip_b = pd_orig.get("attack_b_inference_snippet", "")
            rel    = pd_orig.get("relationship", "-")
            rel_color = relationship_colors.get(rel, "#9E9E9E")
            snippet_rows += f"""
            <tr>
              <td><strong>{pii_type}</strong></td>
              <td style="font-size:0.82em;color:#444;">{snip_a}</td>
              <td style="font-size:0.82em;color:#444;">{snip_b}</td>
              <td style="color:{rel_color};font-weight:bold;">{rel}</td>
            </tr>
            """

        profiles_html += f"""
        <div class="profile-section">
          <h3>Profile: {uname}</h3>
          <table>
            <thead>
              <tr>
                <th>PII Type</th>
                <th>Attack A Guesses (orig)</th><th>Cert A</th>
                <th>Attack B Guesses (orig)</th><th>Cert B</th>
                <th>B→A Relationship</th>
                <th>Accumulated Guesses</th>
                <th>Attack A (anon)</th>
                <th>Attack B (anon)</th>
                <th>Drop A</th><th>Drop B</th>
                <th>Def. A</th><th>Def. B</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
          <details style="margin-top:6px;">
            <summary style="cursor:pointer;color:#4b86b4;font-size:0.9em;">Show inference snippets</summary>
            <table style="margin-top:8px;">
              <thead>
                <tr>
                  <th>PII Type</th>
                  <th>Attack A Reasoning</th>
                  <th>Attack B Reasoning (informed by A)</th>
                  <th>Relationship</th>
                </tr>
              </thead>
              <tbody>{snippet_rows}</tbody>
            </table>
          </details>
        </div>
        """

    summary     = orig["summary"]
    comp_summary = comp["summary"]
    rel_stats   = summary.get("relationship_stats", {})

    html = f"""
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Sequential Inference Attack Report</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
        h1 {{ color: #2a4d69; }}
        h2 {{ color: #4b86b4; }}
        h3 {{ color: #4b86b4; }}
        table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 0.88em; }}
        th {{ background: #4b86b4; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .summary-box {{ background: #f0f5f9; border: 1px solid #4b86b4; border-radius: 8px; padding: 20px; margin: 20px 0; }}
        .metric {{ display: inline-block; background: white; border: 1px solid #ddd; border-radius: 6px; padding: 12px 20px; margin: 6px; text-align: center; min-width: 110px; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #2a4d69; }}
        .metric-label {{ font-size: 0.82em; color: #666; }}
        .profile-section {{ margin: 30px 0; border-top: 2px solid #e0e8f0; padding-top: 16px; }}
        .flowchart {{ background: #f9f9f9; padding: 16px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #4b86b4; }}
        .rel-confirmed  {{ color: #4CAF50; font-weight: bold; }}
        .rel-extended   {{ color: #FF9800; font-weight: bold; }}
        .rel-challenged {{ color: #f44336; font-weight: bold; }}
        .rel-missing    {{ color: #9E9E9E; font-weight: bold; }}
      </style>
      <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
      <script>mermaid.initialize({{ startOnLoad: true, theme: 'base', themeVariables: {{
        primaryColor: '#4b86b4', primaryTextColor: '#fff',
        primaryBorderColor: '#2a4d69', lineColor: '#555',
        secondaryColor: '#f0f5f9', tertiaryColor: '#fff'
      }} }});</script>
    </head>
    <body>
      <h1>Sequential Inference Attack &mdash; Exploration Report</h1>
      <p style="color:#666;">
        Attack A (GPT-4o analytical) runs first on the original text.
        Attack B (Claude sociolinguistic) then runs on the <strong>same text</strong>
        with full visibility into what A found — it confirms, challenges, or extends A's findings.
        The anonymizer receives the accumulated A&nbsp;+&nbsp;B context.
      </p>

      <div class="flowchart">
        <h2>Pipeline Architecture</h2>
        <div class="mermaid">
          flowchart TD
            P(["Reddit Profiles"])

            P --> A1

            subgraph STAGE1["Stage 1 — Sequential Inference on Original Text"]
              A1["Attack A<br/><b>Analytical (GPT-4o)</b><br/>Step-by-step logical deduction<br/>from explicit text evidence"]
              A1 -->|"A's findings injected<br/>into B's prompt"| B1
              B1["Attack B<br/><b>Sociolinguistic (Claude)</b><br/>Confirms, challenges, or extends A<br/>using implicit style signals"]
              B1 --> ACC1
              ACC1["Accumulate<br/>A's view + B's layered view<br/>Tag relationship: confirmed / extended / challenged"]
            end

            ACC1 -->|"accumulated inferences stored into profiles"| ANON

            subgraph STAGE2["Stage 2 — Anonymization"]
              ANON["Anonymizer<br/>Sees A's reasoning + B's additions<br/>Informed by relationship signal"]
              ANON --> UTIL
              UTIL["Utility Scorer"]
            end

            UTIL --> A2

            subgraph STAGE4["Stage 4 — Sequential Inference on Anonymized Text"]
              A2["Attack A<br/><b>Analytical</b>"]
              A2 -->|"A's new findings"| B2
              B2["Attack B<br/><b>Sociolinguistic</b><br/>(informed by new A)"]
              B2 --> ACC2
              ACC2["Accumulate"]
            end

            ACC2 --> CMP

            subgraph ANALYSIS["Analysis"]
              CMP["Certainty drop per attack<br/>Defeat rate A / B / Both<br/>Relationship stats (confirmed/extended/challenged)"]
            end

            classDef input   fill:#2a4d69,color:#fff,stroke:#1a3049
            classDef attack  fill:#4b86b4,color:#fff,stroke:#2a4d69
            classDef accum   fill:#e8a838,color:#fff,stroke:#b07820
            classDef process fill:#5ba55b,color:#fff,stroke:#3d7a3d
            classDef analysis fill:#9b59b6,color:#fff,stroke:#7d3c98

            class P input
            class A1,B1,A2,B2 attack
            class ACC1,ACC2 accum
            class ANON,UTIL process
            class CMP analysis
        </div>
      </div>

      <div class="summary-box">
        <h2>B&rarr;A Relationship Stats (Original Text)</h2>
        <p style="color:#555;font-size:0.93em;">
          How often did Attack B (informed by A) <span class="rel-confirmed">confirm</span>,
          <span class="rel-extended">extend</span>, or
          <span class="rel-challenged">challenge</span> Attack A's findings?
        </p>
        <div class="metric"><div class="metric-value rel-confirmed">{rel_stats.get('confirmed', 0)}</div><div class="metric-label">Confirmed</div></div>
        <div class="metric"><div class="metric-value rel-extended">{rel_stats.get('extended', 0)}</div><div class="metric-label">Extended</div></div>
        <div class="metric"><div class="metric-value rel-challenged">{rel_stats.get('challenged', 0)}</div><div class="metric-label">Challenged</div></div>
        <div class="metric"><div class="metric-value rel-missing">{rel_stats.get('missing_b', 0)}</div><div class="metric-label">Missing B</div></div>
      </div>

      <div class="summary-box">
        <h2>Certainty — Original Text Inference</h2>
        <div class="metric"><div class="metric-value">{summary.get('certainty_a_avg', '-')}</div><div class="metric-label">Avg Certainty A</div></div>
        <div class="metric"><div class="metric-value">{summary.get('certainty_b_avg', '-')}</div><div class="metric-label">Avg Certainty B</div></div>
        <div class="metric"><div class="metric-value">{summary.get('certainty_accumulated_avg', '-')}</div><div class="metric-label">Avg Accumulated</div></div>
        <div class="metric"><div class="metric-value">{summary.get('a_higher_certainty', 0)}</div><div class="metric-label">A &gt; B cert</div></div>
        <div class="metric"><div class="metric-value">{summary.get('b_higher_certainty', 0)}</div><div class="metric-label">B &gt; A cert</div></div>
        <div class="metric"><div class="metric-value">{summary.get('unique_to_a', 0)}</div><div class="metric-label">Unique to A</div></div>
        <div class="metric"><div class="metric-value">{summary.get('unique_to_b', 0)}</div><div class="metric-label">Unique to B</div></div>
        <div class="metric"><div class="metric-value">{summary.get('both_found', 0)}</div><div class="metric-label">Found by Both</div></div>
      </div>

      <div class="summary-box">
        <h2>Anonymization Effectiveness</h2>
        <div class="metric"><div class="metric-value">{comp_summary.get('avg_certainty_drop_a', '-')}</div><div class="metric-label">Avg Cert Drop A</div></div>
        <div class="metric"><div class="metric-value">{comp_summary.get('avg_certainty_drop_b', '-')}</div><div class="metric-label">Avg Cert Drop B</div></div>
        <div class="metric"><div class="metric-value">{comp_summary.get('attacks_defeated_a', 0)}/{comp_summary.get('total_pii_types', 0)}</div><div class="metric-label">Defeated (A)</div></div>
        <div class="metric"><div class="metric-value">{comp_summary.get('attacks_defeated_b', 0)}/{comp_summary.get('total_pii_types', 0)}</div><div class="metric-label">Defeated (B)</div></div>
        <div class="metric"><div class="metric-value">{comp_summary.get('attacks_defeated_both', 0)}/{comp_summary.get('total_pii_types', 0)}</div><div class="metric-label">Defeated (Both)</div></div>
      </div>

      <h2>Per-Profile Results</h2>
      <p style="color:#666;font-size:0.9em;">
        <strong>B&rarr;A Relationship:</strong>
        <span class="rel-confirmed">confirmed</span> = B's top guess matches A's &nbsp;|&nbsp;
        <span class="rel-extended">extended</span> = partial overlap, B added new guesses &nbsp;|&nbsp;
        <span class="rel-challenged">challenged</span> = no overlap, B contradicts A
      </p>
      {profiles_html}

    </body>
    </html>
    """

    report_path = f"{out_dir}/sequential_inference_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nWrote HTML report to {report_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run sequential inference attack pipeline (A → B)"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/anonymization/sequential_inference.yaml",
    )
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="Only generate report from existing results (skip inference)",
    )
    args = parser.parse_args()

    cfg = read_config_from_yaml(args.config_path)
    seed_everything(cfg.seed)
    set_credentials(cfg)

    try:
        from credentials import anthropic_api_key
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    except ImportError:
        pass

    if args.report_only:
        generate_sequential_report(cfg.task_config.outpath)
    else:
        run_sequential_inference_pipeline(cfg)
