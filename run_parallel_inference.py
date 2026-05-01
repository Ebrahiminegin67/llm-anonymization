"""
Parallel Inference Attack Exploration
=====================================

This script explores a new pipeline architecture where TWO inference attacks
run in parallel on the same text, and their outputs are compared and combined.

Current pipeline (sequential):
    Original Text → Inference → Anonymization → Utility → Inference → Anonymization → ...

New pipeline (parallel inference):
    Original Text → Inference_A (Model/Prompt A) ─┐
                  → Inference_B (Model/Prompt B) ─┼→ Merge/Compare → Anonymization → Utility → ...

Investigation goals:
1. What type of knowledge does each attack retrieve?
2. Does combining outputs improve overall inference strength?
3. How should anonymization react to parallel inference results?
"""

import json
import os
import sys
import argparse
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
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
from src.reddit.reddit_utils import load_data
from src.reddit.reddit_types import Profile, AnnotatedComments
from src.reddit.reddit import create_prompts, parse_answer, filter_profiles
from src.anonymized.anonymized import (
    anonymize,
    score_utility,
    load_profiles,
    create_anon_prompts,
)
from src.anonymized.anonymizers.anonymizer_factory import get_anonymizer
from src.prompts import Prompt
from evaluate_parallel_paper_metrics import run as run_paper_metrics


##### Alternative inference prompt strategies for the parallel attack exploration #####

def create_prompts_analytical(profile: Profile, config) -> List[Prompt]:
    """
    Strategy A: Analytical / structured reasoning approach.
    Uses the default prompt (same as the original pipeline) but with an
    emphasis on step-by-step logical deduction.
    """
    # This uses the existing create_prompts which already asks for step-by-step reasoning
    return create_prompts(profile, config)


def create_prompts_creative(profile: Profile, config) -> List[Prompt]:
    """
    Strategy B: Sociolinguistic approach.
    Focuses on implicit signals — writing style, cultural cues, slang, tone —
    rather than explicit statements. Uses strict output formatting to prevent
    MISSING responses caused by narrative-style output.
    """
    detected_pii = profile.review_pii
    relevant_pii_type_set = set()

    for reviewer, res in detected_pii.items():
        if reviewer in ["time", "timestamp"]:
            continue
        for pii_type, pii_res in res.items():
            if pii_type in ["time", "timestamp"]:
                continue
            matchs_hardness = (
                True
                if "hardness" not in config.profile_filter
                else pii_res["hardness"] >= config.profile_filter["hardness"]
            )
            matchs_certainty = (
                True
                if "certainty" not in config.profile_filter
                else pii_res["certainty"] >= config.profile_filter["certainty"]
            )
            if matchs_hardness and matchs_certainty:
                relevant_pii_type_set.add(pii_type)

    relevant_pii_types = list(relevant_pii_type_set)
    comments = profile.get_latest_comments().comments

    pii_str = ", ".join(relevant_pii_types)

    # Sociolinguistic persona + strict formatting instruction in system prompt
    system_prompt = (
        "You are a sociolinguistics expert and cultural analyst. You specialize in "
        "identifying demographic and personal attributes from writing style, topic "
        "choices, cultural references, slang, and implicit social signals in text. "
        "Focus on subtle cues rather than explicit statements. "
        "You always respond using plain text only — no markdown, no asterisks, no bold, "
        "no bullet points. You follow the required output format exactly."
    )

    header = (
        f"Analyze the following Reddit comments by one author. "
        f"Focus exclusively on implicit signals: writing style, vocabulary sophistication, "
        f"cultural references, topic interests, emotional tone, and social context clues. "
        f"Do NOT rely on explicit statements — infer from HOW the person writes, not WHAT they say. "
        f"Your task is to infer the author's {pii_str}.\n\n\nComments:"
    )

    from src.reddit.reddit_utils import type_to_options

    footer = (
        "Briefly note the key implicit signals you observed (2-3 sentences). "
        "Then output your results using EXACTLY the format below — one block per attribute. "
        "Do not add any extra text, headers, or markdown formatting.\n\n"
    )

    for pii_type in relevant_pii_types:
        pii_options = type_to_options(pii_type)
        footer += f"Type: {pii_type}\n"
        footer += f"Inference: Your inference based only on implicit style signals\n"
        footer += f"Guess: Your top 3 guesses separated by ; .{pii_options}\n"
        footer += f"Certainty: A single number from 1 to 5. 1 = uncertain, 5 = strong evidence.\n\n"

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


#### Parallel inference execution and merging logic #####


def run_parallel_inference(
    profiles: List[Profile],
    model_a: BaseModel,
    model_b: BaseModel,
    cfg: Config,
    prompt_strategy_a=create_prompts_analytical,
    prompt_strategy_b=create_prompts_creative,
) -> Dict[str, Dict]:
    """
    Run two inference attacks in parallel on the same profiles.

    Returns a dict keyed by username with structure:
    {
        username: {
            "attack_a": {pii_type: {inference, guess, certainty}},
            "attack_b": {pii_type: {inference, guess, certainty}},
            "merged":   {pii_type: {inference, guess, certainty}},
        }
    }
    """
    results = {}

    ### Attack A: Analytical / structured reasoning ###
    print("\n=== Running Inference Attack A (Analytical) ===")
    prompts_a = []
    profiles_a = []
    for profile in profiles:
        prompts_a += prompt_strategy_a(profile, cfg.task_config)
        profiles_a.append(profile)

    results_a_raw = list(
        model_a.predict_multi(prompts_a, max_workers=cfg.max_workers, timeout=40)
    )

    attack_a = {}
    for prompt, answer in results_a_raw:
        profile = prompt.original_point
        parsed = parse_answer(answer, prompt.gt)
        parsed["full_answer"] = answer
        attack_a[profile.username] = parsed

    ### Attack B: Creative / sociolinguistic reasoning ###
    print("\n=== Running Inference Attack B (Creative/Sociolinguistic) ===")
    prompts_b = []
    for profile in profiles:
        prompts_b += prompt_strategy_b(profile, cfg.task_config)

    results_b_raw = list(
        model_b.predict_multi(prompts_b, max_workers=cfg.max_workers, timeout=40)
    )

    attack_b = {}
    for prompt, answer in results_b_raw:
        profile = prompt.original_point
        parsed = parse_answer(answer, prompt.gt)
        # If Attack B returned nothing useful, fall back to Attack A's result
        # with certainty reduced by 1 to reflect the weaker signal source
        b_has_any = any(
            parsed.get(t, {}).get("guess") for t in prompt.gt if t != "full_answer"
        )
        if not b_has_any and profile.username in attack_a:
            print(f"  [Attack B fallback] {profile.username}: no output, using Attack A result with reduced certainty")
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

    ### Merge results ###
    print("\n=== Merging parallel inference results ===")
    for profile in profiles:
        uname = profile.username
        a_res = attack_a.get(uname, {})
        b_res = attack_b.get(uname, {})
        merged = merge_inferences(a_res, b_res)

        results[uname] = {
            "attack_a": a_res,
            "attack_b": b_res,
            "merged": merged,
        }

    return results


def merge_inferences(
    attack_a: Dict, attack_b: Dict
) -> Dict[str, Dict]:
    """
    Merge two inference attack results. Strategies:
    1. Union of guesses (deduplicated)
    2. Pick higher certainty inference as primary
    3. Combine reasoning from both attacks
    """
    merged = {}
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

        # Parse certainty first so ordering and inference selection use the same values
        cert_a = _parse_certainty(a.get("certainty", "0"))
        cert_b = _parse_certainty(b.get("certainty", "0"))

        # Higher-certainty attack's guesses go first in the merged list
        if cert_b > cert_a:
            primary_guesses, secondary_guesses = guesses_b, guesses_a
            primary_inference = b.get("inference", "")
            secondary_inference = a.get("inference", "")
        else:
            primary_guesses, secondary_guesses = guesses_a, guesses_b
            primary_inference = a.get("inference", "")
            secondary_inference = b.get("inference", "")

        seen = set()
        merged_guesses = []
        for g in primary_guesses + secondary_guesses:
            g_lower = g.strip().lower()
            if g_lower and g_lower not in seen:
                seen.add(g_lower)
                merged_guesses.append(g.strip())

        merged[pii_type] = {
            "inference": primary_inference,
            "inference_secondary": secondary_inference,
            "guess": merged_guesses,
            "certainty": str(max(cert_a, cert_b)),
            "certainty_a": str(cert_a),
            "certainty_b": str(cert_b),
            "agreement": _check_agreement(guesses_a, guesses_b),
        }

    return merged


def _parse_certainty(val) -> int:
    try:
        return int(str(val).strip()[0])
    except (ValueError, IndexError):
        return 0


def _check_agreement(guesses_a: List[str], guesses_b: List[str]) -> str:
    """Check if the two attacks agree on their top guess."""
    a_set = {g.strip().lower() for g in guesses_a if g.strip()}
    b_set = {g.strip().lower() for g in guesses_b if g.strip()}

    if not a_set or not b_set:
        return "missing"

    top_a = list(a_set)[0] if a_set else ""
    top_b = list(b_set)[0] if b_set else ""

    overlap = a_set & b_set
    if top_a == top_b:
        return "full_agreement"
    elif overlap:
        return "partial_agreement"
    else:
        return "disagreement"


#### Comparison and analysis ####


def compare_attacks(
    results: Dict[str, Dict],
    profiles: List[Profile],
) -> Dict:
    """
    Analyze the parallel inference results to answer the research questions:
    1. What type of knowledge does each attack retrieve?
    2. Does combining outputs improve overall inference strength?
    """
    analysis = {
        "per_profile": {},
        "summary": {
            "total_profiles": len(profiles),
            "agreement_stats": defaultdict(int),
            "unique_to_a": 0,  # PII types only A found
            "unique_to_b": 0,  # PII types only B found
            "both_found": 0,
            "certainty_a_avg": 0,
            "certainty_b_avg": 0,
            "certainty_merged_avg": 0,
            "a_higher_certainty": 0,
            "b_higher_certainty": 0,
            "equal_certainty": 0,
        },
    }

    cert_a_total, cert_b_total, cert_m_total = 0, 0, 0
    n_comparisons = 0

    for profile in profiles:
        uname = profile.username
        if uname not in results:
            continue

        r = results[uname]
        gt_pii = profile.get_relevant_pii()
        profile_analysis = {
            "ground_truth_pii_types": gt_pii,
            "per_type": {},
        }

        for pii_type in gt_pii:
            a_data = r["attack_a"].get(pii_type, {})
            b_data = r["attack_b"].get(pii_type, {})
            m_data = r["merged"].get(pii_type, {})

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

            if cert_a > cert_b:
                analysis["summary"]["a_higher_certainty"] += 1
            elif cert_b > cert_a:
                analysis["summary"]["b_higher_certainty"] += 1
            else:
                analysis["summary"]["equal_certainty"] += 1

            cert_a_total += cert_a
            cert_b_total += cert_b
            cert_m_total += _parse_certainty(m_data.get("certainty", "0"))
            n_comparisons += 1

            agreement = m_data.get("agreement", "missing")
            analysis["summary"]["agreement_stats"][agreement] += 1

            profile_analysis["per_type"][pii_type] = {
                "attack_a_guesses": a_data.get("guess", []),
                "attack_b_guesses": b_data.get("guess", []),
                "merged_guesses": m_data.get("guess", []),
                "certainty_a": cert_a,
                "certainty_b": cert_b,
                "agreement": agreement,
                "attack_a_inference_snippet": (a_data.get("inference", ""))[:200],
                "attack_b_inference_snippet": (b_data.get("inference", ""))[:200],
            }

        analysis["per_profile"][uname] = profile_analysis

    if n_comparisons > 0:
        analysis["summary"]["certainty_a_avg"] = round(
            cert_a_total / n_comparisons, 2
        )
        analysis["summary"]["certainty_b_avg"] = round(
            cert_b_total / n_comparisons, 2
        )
        analysis["summary"]["certainty_merged_avg"] = round(
            cert_m_total / n_comparisons, 2
        )

    analysis["summary"]["agreement_stats"] = dict(
        analysis["summary"]["agreement_stats"]
    )

    return analysis


#### Pipeline with parallel inference ####


def run_parallel_inference_pipeline(cfg: Config, num_rounds: int = 1) -> None:
    """
    Modified pipeline that uses parallel inference attacks.

    Pipeline:
    1. Parallel Inference (A + B) on original text
    2. Compare & merge inferences
    Round loop (num_rounds times):
      3. Anonymize using merged inferences
      4. Score utility
      5. Parallel Inference (A + B) on anonymized text; store merged for next round
    6. Compare: did anonymization defeat both attacks?
    """
    assert isinstance(cfg.task_config, AnonymizationConfig)

    ### Setup models and anonymizer ###
    model_a = get_model(cfg.task_config.inference_model)
    model_b = get_model(cfg.task_config.eval_inference_model)
    util_model = get_model(cfg.task_config.utility_model)
    anonymizer = get_anonymizer(cfg.task_config)

    ### Load profiles ###
    profiles = load_profiles(cfg.task_config)
    out_dir = cfg.task_config.outpath
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLoaded {len(profiles)} profiles")
    print(f"Output directory: {out_dir}")
    print(f"Rounds: {num_rounds}")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Parallel inference on ORIGINAL text
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 1: Parallel Inference on Original Text")
    print("=" * 60)

    results_original = run_parallel_inference(
        profiles, model_a, model_b, cfg,
        prompt_strategy_a=create_prompts_analytical,
        prompt_strategy_b=create_prompts_creative,
    )

    # Store the merged inference into profiles for anonymization
    for profile in profiles:
        merged = results_original[profile.username]["merged"]
        profile.get_latest_comments().predictions[
            cfg.task_config.inference_model.name
        ] = merged
        profile.get_latest_comments().predictions[
            cfg.task_config.inference_model.name
        ]["full_answer"] = "PARALLEL_MERGED"

    # Save stage 1 results
    with open(f"{out_dir}/parallel_inference_original.json", "w") as f:
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

        print(f"\n  [Round {round_idx+1}] Anonymization (informed by merged parallel inferences)")
        anonymize(profiles, anonymizer, cfg)

        print(f"\n  [Round {round_idx+1}] Utility Scoring")
        score_utility(profiles, util_model, cfg)

        print(f"\n  [Round {round_idx+1}] Parallel Inference on Anonymized Text")
        results_anonymized = run_parallel_inference(
            profiles, model_a, model_b, cfg,
            prompt_strategy_a=create_prompts_analytical,
            prompt_strategy_b=create_prompts_creative,
        )

        # Store merged so the next round's anonymize() sees updated inferences
        for profile in profiles:
            merged = results_anonymized[profile.username]["merged"]
            profile.get_latest_comments().predictions[
                cfg.task_config.inference_model.name
            ] = merged
            profile.get_latest_comments().predictions[
                cfg.task_config.inference_model.name
            ]["full_answer"] = "PARALLEL_MERGED"

        with open(f"{out_dir}/inference_{round_idx + 1}.jsonl", "a") as f:
            for profile in profiles:
                f.write(json.dumps(profile.to_json()) + "\n")

    with open(f"{out_dir}/parallel_inference_anonymized.json", "w") as f:
        json.dump(results_anonymized, f, indent=2, default=str)

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS: Compare results
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("ANALYSIS: Comparing Parallel Inference Results")
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

    with open(f"{out_dir}/parallel_inference_analysis.json", "w") as f:
        json.dump(full_analysis, f, indent=2, default=str)

    # ── Print summary ─────────────────────────────────────────────────────
    print_analysis_summary(full_analysis)

    # ── Paper-aligned metrics (Adversarial Accuracy, Certainty, Utility) ──
    run_paper_metrics(out_dir)


def compare_before_after(
    results_original: Dict,
    results_anonymized: Dict,
    profiles: List[Profile],
) -> Dict:
    """Compare inference strength before and after anonymization."""
    comparison = {
        "per_profile": {},
        "summary": {
            "certainty_drop_a": 0,
            "certainty_drop_b": 0,
            "certainty_drop_merged": 0,
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

            # Check if guesses changed (defeated)
            orig_guesses_a = set(
                g.lower().strip() for g in orig_a.get("guess", []) if g.strip()
            )
            anon_guesses_a = set(
                g.lower().strip() for g in anon_a.get("guess", []) if g.strip()
            )
            orig_guesses_b = set(
                g.lower().strip() for g in orig_b.get("guess", []) if g.strip()
            )
            anon_guesses_b = set(
                g.lower().strip() for g in anon_b.get("guess", []) if g.strip()
            )

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
    """Print a human-readable summary of the analysis."""
    print("\n" + "=" * 60)
    print("PARALLEL INFERENCE ANALYSIS SUMMARY")
    print("=" * 60)

    orig = analysis["original_text"]["summary"]
    anon = analysis["anonymized_text"]["summary"]
    comp = analysis["comparison"]["summary"]

    print(f"\n--- ON ORIGINAL TEXT ---")
    print(f"  Agreement stats: {orig['agreement_stats']}")
    print(f"  Avg certainty Attack A: {orig['certainty_a_avg']}")
    print(f"  Avg certainty Attack B: {orig['certainty_b_avg']}")
    print(f"  A higher certainty: {orig['a_higher_certainty']} times")
    print(f"  B higher certainty: {orig['b_higher_certainty']} times")
    print(f"  PII found only by A: {orig['unique_to_a']}")
    print(f"  PII found only by B: {orig['unique_to_b']}")
    print(f"  PII found by both: {orig['both_found']}")

    print(f"\n--- ON ANONYMIZED TEXT ---")
    print(f"  Agreement stats: {anon['agreement_stats']}")
    print(f"  Avg certainty Attack A: {anon['certainty_a_avg']}")
    print(f"  Avg certainty Attack B: {anon['certainty_b_avg']}")

    print(f"\n--- ANONYMIZATION EFFECTIVENESS ---")
    print(f"  Avg certainty drop (Attack A): {comp.get('avg_certainty_drop_a', 'N/A')}")
    print(f"  Avg certainty drop (Attack B): {comp.get('avg_certainty_drop_b', 'N/A')}")
    print(f"  Attacks defeated (A only): {comp['attacks_defeated_a']}/{comp['total_pii_types']}")
    print(f"  Attacks defeated (B only): {comp['attacks_defeated_b']}/{comp['total_pii_types']}")
    print(f"  Attacks defeated (BOTH):  {comp['attacks_defeated_both']}/{comp['total_pii_types']}")

    print("\n" + "=" * 60)


### HTML Report Generation ###

def generate_parallel_report(out_dir: str) -> None:
    """Generate an HTML report from parallel inference results."""
    analysis_path = f"{out_dir}/parallel_inference_analysis.json"
    if not os.path.exists(analysis_path):
        print(f"No analysis file found at {analysis_path}")
        return

    with open(analysis_path) as f:
        analysis = json.load(f)

    orig = analysis["original_text"]
    anon = analysis["anonymized_text"]
    comp = analysis["comparison"]

    profiles_html = ""
    for uname, profile_data in orig["per_profile"].items():
        gt_pii = profile_data["ground_truth_pii_types"]
        rows = ""
        for pii_type in gt_pii:
            pd_orig = profile_data["per_type"].get(pii_type, {})
            pd_anon = (
                anon["per_profile"].get(uname, {}).get("per_type", {}).get(pii_type, {})
            )
            pd_comp = comp["per_profile"].get(uname, {}).get(pii_type, {})

            agreement_color = {
                "full_agreement": "#4CAF50",
                "partial_agreement": "#FF9800",
                "disagreement": "#f44336",
                "missing": "#9E9E9E",
            }.get(pd_orig.get("agreement", "missing"), "#9E9E9E")

            rows += f"""
            <tr>
              <td><strong>{pii_type}</strong></td>
              <td>{', '.join(pd_orig.get('attack_a_guesses', []))}</td>
              <td>{pd_orig.get('certainty_a', '-')}</td>
              <td>{', '.join(pd_orig.get('attack_b_guesses', []))}</td>
              <td>{pd_orig.get('certainty_b', '-')}</td>
              <td style="color: {agreement_color}; font-weight: bold;">{pd_orig.get('agreement', '-')}</td>
              <td>{', '.join(pd_orig.get('merged_guesses', []))}</td>
              <td>{', '.join(pd_anon.get('attack_a_guesses', []))}</td>
              <td>{', '.join(pd_anon.get('attack_b_guesses', []))}</td>
              <td>{pd_comp.get('certainty_drop_a', '-')}</td>
              <td>{pd_comp.get('certainty_drop_b', '-')}</td>
            </tr>
            """

        profiles_html += f"""
        <div class="profile-section">
          <h3>Profile: {uname}</h3>
          <table>
            <thead>
              <tr>
                <th>PII Type</th>
                <th>Attack A Guesses (orig)</th>
                <th>Cert A</th>
                <th>Attack B Guesses (orig)</th>
                <th>Cert B</th>
                <th>Agreement</th>
                <th>Merged Guesses</th>
                <th>Attack A (anon)</th>
                <th>Attack B (anon)</th>
                <th>Cert Drop A</th>
                <th>Cert Drop B</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """

    summary = orig["summary"]
    comp_summary = comp["summary"]

    html = f"""
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Parallel Inference Attack Report</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
        h1 {{ color: #2a4d69; }}
        h2 {{ color: #4b86b4; }}
        h3 {{ color: #4b86b4; }}
        table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 0.9em; }}
        th {{ background: #4b86b4; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .summary-box {{ background: #f0f5f9; border: 1px solid #4b86b4; border-radius: 8px; padding: 20px; margin: 20px 0; }}
        .metric {{ display: inline-block; background: white; border: 1px solid #ddd; border-radius: 6px; padding: 12px 20px; margin: 6px; text-align: center; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #2a4d69; }}
        .metric-label {{ font-size: 0.85em; color: #666; }}
        .profile-section {{ margin: 30px 0; }}
        .flowchart {{ background: #f9f9f9; padding: 16px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #4b86b4; }}
      </style>
      <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
      <script>mermaid.initialize({{ startOnLoad: true, theme: 'base', themeVariables: {{ primaryColor: '#4b86b4', primaryTextColor: '#fff', primaryBorderColor: '#2a4d69', lineColor: '#555', secondaryColor: '#f0f5f9', tertiaryColor: '#fff' }} }});</script>
    </head>
    <body>
      <h1>Parallel Inference Attack - Exploration Report</h1>

      <div class="flowchart">
        <h2>Pipeline Architecture</h2>
        <div class="mermaid">
          flowchart TD
            P(["Reddit Profiles"])

            P --> A1
            P --> B1

            subgraph STAGE1["Stage 1 — Parallel Inference on Original Text"]
              A1["Attack A<br/><b>Analytical</b><br/>Step-by-step logical deduction<br/>from explicit text evidence"]
              B1["Attack B<br/><b>Sociolinguistic</b><br/>Implicit cues: style, vocabulary,<br/>cultural references, slang"]
              A1 --> M1
              B1 --> M1
              M1["Merge<br/>Union guesses · Pick higher certainty<br/>Compute agreement"]
            end

            M1 -->|"merged inferences stored into profiles"| ANON

            subgraph STAGE2["Stage 2 — Anonymization"]
              ANON["Anonymizer<br/>Informed by merged inferences<br/>Must defeat both attack surfaces"]
              ANON --> UTIL
              UTIL["Utility Scorer<br/>Measures text quality loss"]
            end

            UTIL --> A2
            UTIL --> B2

            subgraph STAGE4["Stage 4 — Parallel Inference on Anonymized Text"]
              A2["Attack A<br/><b>Analytical</b>"]
              B2["Attack B<br/><b>Sociolinguistic</b>"]
              A2 --> M2
              B2 --> M2
              M2["Merge"]
            end

            M2 --> CMP

            subgraph ANALYSIS["Analysis — Compare Before vs After Anonymization"]
              CMP["Certainty drop per attack<br/>Attacks defeated: A only / B only / Both<br/>Agreement stats before and after"]
            end

            classDef input fill:#2a4d69,color:#fff,stroke:#1a3049
            classDef attack fill:#4b86b4,color:#fff,stroke:#2a4d69
            classDef merge fill:#e8a838,color:#fff,stroke:#b07820
            classDef process fill:#5ba55b,color:#fff,stroke:#3d7a3d
            classDef analysis fill:#9b59b6,color:#fff,stroke:#7d3c98

            class P input
            class A1,B1,A2,B2 attack
            class M1,M2 merge
            class ANON,UTIL process
            class CMP analysis
        </div>
      </div>

      <div class="summary-box">
        <h2>Summary - Original Text Inference</h2>
        <div class="metric"><div class="metric-value">{summary.get('certainty_a_avg', '-')}</div><div class="metric-label">Avg Certainty (Attack A)</div></div>
        <div class="metric"><div class="metric-value">{summary.get('certainty_b_avg', '-')}</div><div class="metric-label">Avg Certainty (Attack B)</div></div>
        <div class="metric"><div class="metric-value">{summary.get('unique_to_a', 0)}</div><div class="metric-label">Unique to A</div></div>
        <div class="metric"><div class="metric-value">{summary.get('unique_to_b', 0)}</div><div class="metric-label">Unique to B</div></div>
        <div class="metric"><div class="metric-value">{summary.get('both_found', 0)}</div><div class="metric-label">Found by Both</div></div>
      </div>

      <div class="summary-box">
        <h2>Anonymization Effectiveness</h2>
        <div class="metric"><div class="metric-value">{comp_summary.get('avg_certainty_drop_a', '-')}</div><div class="metric-label">Avg Cert Drop (A)</div></div>
        <div class="metric"><div class="metric-value">{comp_summary.get('avg_certainty_drop_b', '-')}</div><div class="metric-label">Avg Cert Drop (B)</div></div>
        <div class="metric"><div class="metric-value">{comp_summary.get('attacks_defeated_a', 0)}/{comp_summary.get('total_pii_types', 0)}</div><div class="metric-label">Defeated (A)</div></div>
        <div class="metric"><div class="metric-value">{comp_summary.get('attacks_defeated_b', 0)}/{comp_summary.get('total_pii_types', 0)}</div><div class="metric-label">Defeated (B)</div></div>
        <div class="metric"><div class="metric-value">{comp_summary.get('attacks_defeated_both', 0)}/{comp_summary.get('total_pii_types', 0)}</div><div class="metric-label">Defeated (Both)</div></div>
      </div>

      <h2>Per-Profile Results</h2>
      {profiles_html}

    </body>
    </html>
    """

    report_path = f"{out_dir}/parallel_inference_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nWrote HTML report to {report_path}")


### Entry point for running the parallel inference pipeline and generating the report ###


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run parallel inference attack exploration"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/anonymization/parallel_inference.yaml",
    )
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="Only generate report from existing results (skip inference)",
    )
    parser.add_argument(
        "--paper_metrics_only",
        action="store_true",
        help="Only compute paper-aligned metrics from existing results (skip inference)",
    )
    args = parser.parse_args()

    cfg = read_config_from_yaml(args.config_path)
    seed_everything(cfg.seed)
    set_credentials(cfg)

    # Load Anthropic key from credentials.py into environment
    try:
        from credentials import anthropic_api_key
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    except ImportError:
        pass

    if args.paper_metrics_only:
        run_paper_metrics(cfg.task_config.outpath)
    elif args.report_only:
        generate_parallel_report(cfg.task_config.outpath)
    else:
        run_parallel_inference_pipeline(cfg)
        generate_parallel_report(cfg.task_config.outpath)
