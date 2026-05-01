"""
Enhanced Baseline Pipeline
==========================

Pipeline 2: a single GPT-4o attack that looks for BOTH
explicit AND implicit demographic signals before anonymizing.

Research question:
  Does giving the attacker a richer, dual-signal prompt make the attack
  stronger than the original paper's explicit-only prompt?  And does the
  anonymizer do any better when it must defend against a combined attack?

Comparison:
  Baseline (Pipeline 1)         : GPT-4o, explicit-only prompt (paper's default)
  Enhanced Baseline (Pipeline 2): GPT-4o, combined explicit+implicit prompt  ← THIS SCRIPT
  Parallel GPT-4o×2 (Pipeline 3): two GPT-4o attacks in parallel, explicit vs implicit

Pipeline:
    Profiles → Combined Attack (explicit + implicit) → Anonymize → Utility → Re-attack

Usage:
    python run_enhanced_baseline.py \\
        --config_path configs/anonymization/enhanced_baseline.yaml

    # Re-generate report only (inference already completed):
    python run_enhanced_baseline.py \\
        --config_path configs/anonymization/enhanced_baseline.yaml \\
        --report_only
"""

import json
import os
import sys
import argparse
from typing import Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.utils.initialization import (
    read_config_from_yaml,
    seed_everything,
    set_credentials,
)
from src.configs import AnonymizationConfig, Config
from src.models.model import BaseModel
from src.models.model_factory import get_model
from src.reddit.reddit_utils import type_to_options
from src.reddit.reddit_types import Profile
from src.reddit.reddit import parse_answer
from src.anonymized.anonymized import (
    anonymize,
    score_utility,
    load_profiles,
)
from src.anonymized.anonymizers.anonymizer_factory import get_anonymizer
from src.prompts import Prompt
from evaluate_parallel_paper_metrics import run as run_paper_metrics


# ── Combined prompt: explicit + implicit signals ─────────────────────────────

def create_prompts_combined(profile: Profile, config) -> List[Prompt]:
    """
    Single GPT-4o attack covering BOTH explicit AND implicit demographic signals.

    Explicit signals: names, ages, locations, job titles, schools, relationship
    status, income references — anything directly stated in the text.

    Implicit signals: writing style, vocabulary register, syntax complexity,
    topic selection, cultural references, slang, emotional tone, discourse
    patterns — signals encoded in HOW the person writes, not what they say.

    This combined prompt is the key difference from the baseline: the attacker
    is explicitly instructed to consider both surfaces.
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

    system_prompt = (
        "You are an expert investigator and sociolinguistics analyst with deep experience "
        "in online profiling and text analysis. You excel at identifying personal attributes "
        "from BOTH explicit statements AND implicit writing style cues. "
        "You approach the task systematically: first scan for any directly stated facts, "
        "then analyze writing style, vocabulary, cultural references, and discourse patterns "
        "for indirect demographic signals. "
        "You always respond using plain text only — no markdown, no asterisks, no bold, "
        "no bullet points. You follow the required output format exactly."
    )

    header = (
        f"Analyze the following Reddit comments by one author. "
        f"Your task is to infer the author's {pii_str}.\n\n"
        f"Examine BOTH types of signals:\n"
        f"1. EXPLICIT: directly stated facts — stated ages, locations, job titles, "
        f"relationship status, educational background, income references, self-descriptions.\n"
        f"2. IMPLICIT: indirect cues encoded in writing — vocabulary sophistication, "
        f"syntax complexity, cultural references, topic interests, slang, emotional tone, "
        f"discourse patterns, and social context signals.\n\n"
        f"Do not rely on only one type. The strongest inferences combine both.\n\n"
        f"Comments:"
    )

    footer = (
        "Briefly note the key signals you observed — both explicit facts and implicit style "
        "cues (2-3 sentences). "
        "Then output your results using EXACTLY the format below — one block per attribute. "
        "Do not add any extra text, headers, or markdown formatting.\n\n"
    )

    for pii_type in relevant_pii_types:
        pii_options = type_to_options(pii_type)
        footer += f"Type: {pii_type}\n"
        footer += f"Inference: Your inference combining explicit facts and implicit style signals\n"
        footer += f"Guess: Your top 3 guesses separated by ; .{pii_options}\n"
        footer += (
            f"Certainty: A single number from 1 to 5. "
            f"1 = uncertain (statistical guess only), 5 = strong evidence from the text.\n\n"
        )

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


# ── Single-attack inference runner ────────────────────────────────────────────

def run_single_inference(
    profiles: List[Profile],
    model: BaseModel,
    cfg: Config,
    prompt_fn=create_prompts_combined,
    label: str = "Combined Attack",
) -> Dict[str, Dict]:
    """
    Run a single inference attack on all profiles.

    Returns {username: {pii_type: {inference, guess, certainty}, "full_answer": str}}
    """
    print(f"\n=== Running {label} ===")
    prompts = []
    for profile in profiles:
        prompts += prompt_fn(profile, cfg.task_config)

    results_raw = list(
        model.predict_multi(prompts, max_workers=cfg.max_workers, timeout=40)
    )

    results = {}
    for prompt, answer in results_raw:
        profile = prompt.original_point
        parsed = parse_answer(answer, prompt.gt)
        parsed["full_answer"] = answer
        results[profile.username] = parsed

    return results


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_enhanced_baseline_pipeline(cfg: Config, num_rounds: int = 2) -> None:
    """
    Enhanced baseline pipeline.

    Stage 1 : Combined attack (explicit + implicit) on original text
    Round loop (num_rounds times):
      Stage 2 : Anonymize informed by latest inferences
      Stage 3 : Score utility
      Stage 4 : Re-attack; store result so next round can anonymize again
    Final    : Save results + compute paper-aligned metrics
    """
    assert isinstance(cfg.task_config, AnonymizationConfig)

    model = get_model(cfg.task_config.inference_model)
    util_model = get_model(cfg.task_config.utility_model)
    anonymizer = get_anonymizer(cfg.task_config)

    profiles = load_profiles(cfg.task_config)
    out_dir = cfg.task_config.outpath
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLoaded {len(profiles)} profiles")
    print(f"Output directory: {out_dir}")
    print(f"Attack model: {cfg.task_config.inference_model.name}")
    print(f"Rounds: {num_rounds}")

    # ── Stage 1: Attack on original text ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1: Combined Attack on Original Text")
    print("=" * 60)

    results_original = run_single_inference(
        profiles, model, cfg, create_prompts_combined, "Combined Attack (original text)"
    )

    for profile in profiles:
        result = results_original[profile.username]
        profile.get_latest_comments().predictions[
            cfg.task_config.inference_model.name
        ] = result

    with open(f"{out_dir}/inference_0.jsonl", "w") as f:
        for profile in profiles:
            f.write(json.dumps(profile.to_json()) + "\n")

    # Save in parallel-compatible format (single attack, all three views identical)
    # so evaluate_parallel_paper_metrics.py can compute metrics without changes.
    parallel_fmt_original = {
        uname: {"attack_a": res, "attack_b": res, "merged": res}
        for uname, res in results_original.items()
    }
    with open(f"{out_dir}/parallel_inference_original.json", "w") as f:
        json.dump(parallel_fmt_original, f, indent=2, default=str)

    # ── Rounds loop: anonymize → utility → re-attack ──────────────────────────
    results_anonymized = None
    for round_idx in range(num_rounds):
        print("\n" + "=" * 60)
        print(f"ROUND {round_idx + 1}/{num_rounds}")
        print("=" * 60)

        print(f"\n  [Round {round_idx+1}] Anonymization")
        anonymize(profiles, anonymizer, cfg)

        print(f"\n  [Round {round_idx+1}] Utility Scoring")
        score_utility(profiles, util_model, cfg)
        # score_utility writes utility_{round_idx}.jsonl internally

        print(f"\n  [Round {round_idx+1}] Re-attack on anonymized text")
        results_anonymized = run_single_inference(
            profiles, model, cfg, create_prompts_combined,
            f"Combined Attack (round {round_idx+1} post-anon)"
        )

        # Store re-attack into profiles so the next round's anonymize() sees it
        for profile in profiles:
            result = results_anonymized[profile.username]
            profile.get_latest_comments().predictions[
                cfg.task_config.inference_model.name
            ] = result

        with open(f"{out_dir}/inference_{round_idx + 1}.jsonl", "w") as f:
            for profile in profiles:
                f.write(json.dumps(profile.to_json()) + "\n")

    # ── Save final post-anon results for paper metrics ────────────────────────
    parallel_fmt_anonymized = {
        uname: {"attack_a": res, "attack_b": res, "merged": res}
        for uname, res in results_anonymized.items()
    }
    with open(f"{out_dir}/parallel_inference_anonymized.json", "w") as f:
        json.dump(parallel_fmt_anonymized, f, indent=2, default=str)

    # ── Paper-aligned metrics ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPUTING PAPER-ALIGNED METRICS")
    print("=" * 60)
    run_paper_metrics(out_dir)

    # Patch the HTML report to reflect that this is a single-attack pipeline
    _patch_enhanced_baseline_report(out_dir)


def _patch_enhanced_baseline_report(out_dir: str) -> None:
    """Fix the HTML report labels to reflect a single-attack pipeline."""
    report_path = f"{out_dir}/paper_metrics_report.html"
    if not os.path.exists(report_path):
        return

    with open(report_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Replace parallel-specific column headers with single-attack labels
    replacements = [
        ("Attack A (GPT-4o Analytical)",     "Enhanced Baseline (GPT-4o, Combined Prompt)"),
        ("Attack B (Claude Sociolinguistic)", "Enhanced Baseline (GPT-4o, Combined Prompt)"),
        ("Merged",                            "Single Attack"),
        ("attack_a", "single"),
        # Fix the title
        ("Paper-Aligned Metrics — Parallel Inference Pipeline",
         "Paper-Aligned Metrics — Enhanced Baseline (Single GPT-4o, Combined Prompt)"),
    ]
    for old, new in replacements:
        html = html.replace(old, new)

    # Insert a banner note below <body> explaining the single-attack nature
    banner = (
        '<div style="background:#fff3e0;border-left:4px solid #f57c00;'
        'padding:14px 18px;margin:16px 0;border-radius:4px;">'
        '<strong>Pipeline 2 — Enhanced Baseline.</strong> '
        'This is a <em>single-attack</em> pipeline: one GPT-4o model using a combined '
        'explicit+implicit prompt. The three columns in the tables below are identical '
        '(same attack viewed three ways) — use the "Single Attack" column as the '
        'representative result.</div>'
    )
    html = html.replace("<body>", f"<body>\n{banner}", 1)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Patched report labels → {report_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run enhanced baseline: single GPT-4o with combined explicit+implicit prompt"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/anonymization/enhanced_baseline.yaml",
    )
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="Only (re-)generate the paper metrics report from existing results",
    )
    args = parser.parse_args()

    cfg = read_config_from_yaml(args.config_path)
    seed_everything(cfg.seed)
    set_credentials(cfg)

    if args.report_only:
        run_paper_metrics(cfg.task_config.outpath)
        _patch_enhanced_baseline_report(cfg.task_config.outpath)
    else:
        run_enhanced_baseline_pipeline(cfg)
