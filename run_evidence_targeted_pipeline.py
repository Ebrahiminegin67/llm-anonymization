"""
Evidence-Targeted Anonymization Pipeline
=========================================

Thesis contribution: demonstrates that when the anonymizer explicitly targets
the evidence passages flagged by a multi-model attack, adversarial accuracy
drops compared to generic anonymization — proving that attack-aware,
evidence-targeted anonymization is the key to improved privacy, not just
richer attack architecture alone.

Pipeline:
  1. Parallel Inference (Attack A: analytical + Attack B: sociolinguistic) on original text
  2. Evidence-Targeted Anonymization  — the anonymizer receives both attackers' full
     reasoning chains and is instructed to identify and surgically remove the specific
     textual evidence that enables each inference
  3. Utility scoring
  4. Parallel Inference (Attack A + Attack B) on anonymized text
  5. Paper-aligned metrics (Adversarial Accuracy, Evidence Rate, Combined Utility)

Comparison:
  Run alongside run_parallel_inference.py and compare paper_metrics.json files.
  The evidence-targeted pipeline should show lower post-anonymization adversarial
  accuracy while maintaining comparable utility scores.

Usage:
  python run_evidence_targeted_pipeline.py \\
      --config_path configs/anonymization/evidence_targeted.yaml
"""

import json
import os
import sys
import argparse
import re
from copy import deepcopy
from typing import Dict, List, Iterator

sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.utils.initialization import read_config_from_yaml, seed_everything, set_credentials
from src.configs import AnonymizationConfig, Config
from src.models.model import BaseModel
from src.models.model_factory import get_model
from src.reddit.reddit_types import Profile, AnnotatedComments, Comment
from src.anonymized.anonymized import anonymize, score_utility, load_profiles
from src.anonymized.anonymizers.anonymizer import Anonymizer
from src.utils.string_utils import select_closest
from src.prompts import Prompt

from run_parallel_inference import (
    run_parallel_inference,
    merge_inferences,
    create_prompts_analytical,
    create_prompts_creative,
    compare_attacks,
    compare_before_after,
    print_analysis_summary,
    _parse_certainty,
)
from evaluate_parallel_paper_metrics import run as run_paper_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Evidence-Targeted Anonymizer
# ══════════════════════════════════════════════════════════════════════════════

class EvidenceTargetedAnonymizer(Anonymizer):
    """
    Anonymizer that explicitly targets the specific evidence passages identified
    by the parallel multi-model attacker.

    Unlike the standard anonymizer (LLMFullAnonymizer) which presents inferences
    as generic Type/Inference/Guess summaries, this anonymizer:
      1. Shows both attackers' full reasoning chains side-by-side
      2. Instructs the model to identify exact evidence spans in the text before modifying
      3. Performs surgical replacement of only those spans
      4. Scales aggressiveness based on inter-attacker agreement and certainty

    The key insight: if the anonymizer knows *which specific phrases* the attacker
    is using as evidence, it can target those phrases directly rather than making
    broad changes that may reduce utility without defeating the inference.
    """

    def __init__(self, cfg, model: BaseModel):
        self.model = model
        self.cfg = cfg

    def _create_anon_prompt(self, profile: Profile) -> List[Prompt]:
        """Build evidence-targeted anonymization prompt from parallel attack results."""
        comments = profile.get_latest_comments().comments
        comment_string = "\n".join([str(c) for c in comments])

        # Read merged parallel inference stored by the pipeline
        predictions = profile.get_latest_comments().predictions.get(
            self.model.config.name, {}
        )

        # Build per-attribute evidence blocks
        evidence_blocks = ""
        has_structured_inference = False

        for pii_type, inf in predictions.items():
            if pii_type == "full_answer":
                continue
            if not isinstance(inf, dict) or "guess" not in inf:
                continue
            has_structured_inference = True

            cert = _parse_certainty(inf.get("certainty", "0"))
            agreement = inf.get("agreement", "")
            guesses = inf.get("guess", [])
            guess_str = " / ".join(str(g) for g in (guesses[:3] if isinstance(guesses, list) else [guesses]))

            if agreement == "full_agreement" or cert >= 4:
                priority = "HIGH — both attackers agree or confidence >= 4/5 — anonymize aggressively"
            elif agreement == "partial_agreement" or cert >= 3:
                priority = "MEDIUM — partial agreement or moderate confidence"
            else:
                priority = "LOW — speculative, only change if evidence is very explicit"

            evidence_blocks += f"Attribute: {pii_type.upper()}\n"
            evidence_blocks += f"  Priority: {priority}\n"
            evidence_blocks += f"  Attacker verdict: {guess_str}  (confidence {cert}/5)\n"
            evidence_blocks += f"  Analytical attack reasoning:\n"
            evidence_blocks += f"    {inf.get('inference', 'N/A')}\n"
            if inf.get("inference_secondary"):
                evidence_blocks += f"  Sociolinguistic attack reasoning:\n"
                evidence_blocks += f"    {inf['inference_secondary']}\n"
            evidence_blocks += "\n"

        if not has_structured_inference:
            # Fallback: no structured inferences found, return text unchanged
            system_prompt = "You are an expert anonymizer."
            prompt = Prompt(
                system_prompt=system_prompt,
                header="Return the following comments unchanged.",
                intermediate=f"\n\n{comment_string}\n",
                footer="\nWrite a single # then return the original text exactly.",
                target=profile.get_relevant_pii()[0] if profile.get_relevant_pii() else "",
                original_point=profile,
                gt=profile.get_relevant_pii(),
                answer="",
                shots=[],
                id=profile.username,
            )
            return [prompt]

        system_prompt = (
            "You are an expert privacy anonymizer. Your goal is to surgically modify "
            "Reddit comments to defeat privacy attacks while preserving the original "
            "meaning, tone, and narrative as much as possible. "
            "You always respond in plain text only — no markdown, no asterisks, no bullet points."
        )

        header = (
            "A parallel multi-model privacy attack (two independent LLMs with different "
            "strategies) has analyzed the following Reddit comments and produced the "
            "evidence-based reasoning shown below. Your task is to perform evidence-targeted "
            "anonymization: first identify the specific phrases the attackers are using as "
            "evidence, then surgically replace only those phrases."
        )

        intermediate = (
            f"\n\nORIGINAL COMMENTS:\n{comment_string}\n\n"
            f"MULTI-MODEL ATTACKER EVIDENCE ANALYSIS:\n{evidence_blocks}"
        )

        footer = (
            "Perform evidence-targeted anonymization in two steps.\n\n"
            "STEP 1 — Evidence Identification:\n"
            "For each HIGH and MEDIUM priority attribute, identify the exact phrases or "
            "sentences in the original comments that the attackers are relying on as evidence. "
            "Quote them with double quotes. Be specific — name the phrase, not the whole comment.\n\n"
            "STEP 2 — Targeted Replacement:\n"
            "Replace only those specific phrases with alternatives that:\n"
            "  - Remove or obscure the identifying signal\n"
            "  - Preserve the original meaning and tone\n"
            "  - Are minimally invasive (change as few words as possible)\n"
            "  - Do NOT introduce new identifying information\n\n"
            "For LOW priority attributes, only make changes if the evidence is direct and explicit.\n\n"
            "After your explanation, write a single # on its own line, then write the complete "
            "anonymized text with all comments in the same order, one per line, preserving any "
            "date prefix format (YYYY-MM-DD: comment text) if present."
        )

        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate=intermediate,
            footer=footer,
            target=profile.get_relevant_pii()[0] if profile.get_relevant_pii() else "",
            original_point=profile,
            gt=profile.get_relevant_pii(),
            answer="",
            shots=[],
            id=profile.username,
        )
        return [prompt]

    def filter_and_align_comments(self, answer: str, op: Profile) -> List[Comment]:
        """Parse anonymized comments from LLM response, aligning them to the original."""
        try:
            split_answer = answer.split("\n#")
            if len(split_answer) == 1:
                new_comments_raw = answer.strip()
            elif len(split_answer) == 2:
                new_comments_raw = split_answer[1].strip()
            else:
                new_comments_raw = "\n".join(split_answer[1:]).strip()
        except Exception:
            print(f"Could not split answer for {op.username}")
            return deepcopy(op.get_latest_comments().comments)

        new_comments_list = [c for c in new_comments_raw.split("\n") if c.strip()]

        original_comments = op.get_latest_comments().comments

        if len(new_comments_list) != len(original_comments):
            print(
                f"Comment count mismatch for {op.username}: "
                f"{len(new_comments_list)} parsed vs {len(original_comments)} original"
            )
            old_comment_ids = [-1] * len(original_comments)
            used_idx = set()
            for i, comment in enumerate(original_comments):
                closest_match, sim, idx = select_closest(
                    comment.text,
                    new_comments_list,
                    dist="jaro_winkler",
                    return_idx=True,
                    return_sim=True,
                )
                if idx not in used_idx and sim > 0.5:
                    old_comment_ids[i] = idx
                    used_idx.add(idx)
            selected = [
                new_comments_list[idx] if idx != -1 else original_comments[i].text
                for i, idx in enumerate(old_comment_ids)
            ]
        else:
            selected = new_comments_list

        typed_comments = []
        for i, comment_text in enumerate(selected):
            if re.search(r"\d{4}-\d{2}-\d{2}:", comment_text[:11]) is not None:
                comment_text = comment_text[11:].strip()
            old_com = original_comments[i]
            typed_comments.append(Comment(comment_text, old_com.subreddit, old_com.user, old_com.timestamp))
        return typed_comments

    def anonymize(self, text: str) -> str:
        pass

    def anonymize_profiles(self, profiles: List[Profile]) -> Iterator[Profile]:
        prompts = []
        for profile in profiles:
            prompts.extend(self._create_anon_prompt(profile))

        for i, res in enumerate(
            self.model.predict_multi(prompts, max_workers=self.cfg.max_workers, timeout=120)
        ):
            prompt, answer = res
            op = prompt.original_point
            assert isinstance(op, Profile)

            print(f" Profile {i}: {op.username} ".center(60, "="))
            print(prompt.get_prompt())
            op.print_review_pii()
            print(f"--- {self.model.config.name} response ---\n{answer}\n")

            typed_comments = self.filter_and_align_comments(answer, op)
            print(f"Anonymized comments: {typed_comments}")

            op.comments.append(AnnotatedComments(typed_comments, op.review_pii, {}, {}))
            yield op


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline orchestration
# ══════════════════════════════════════════════════════════════════════════════

def run_evidence_targeted_pipeline(cfg: Config) -> None:
    """
    Evidence-targeted anonymization pipeline.

    Identical to run_parallel_inference_pipeline() except Stage 2 uses
    EvidenceTargetedAnonymizer instead of LLMFullAnonymizer, passing both
    attackers' explicit reasoning chains to the anonymizer for surgical targeting.
    """
    assert isinstance(cfg.task_config, AnonymizationConfig)

    # Model setup
    model_a      = get_model(cfg.task_config.inference_model)
    model_b      = get_model(cfg.task_config.eval_inference_model)
    util_model   = get_model(cfg.task_config.utility_model)
    anon_model   = get_model(cfg.task_config.anon_model)
    anonymizer   = EvidenceTargetedAnonymizer(cfg.task_config.anonymizer, anon_model)

    profiles = load_profiles(cfg.task_config)
    out_dir  = cfg.task_config.outpath
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLoaded {len(profiles)} profiles")
    print(f"Output directory: {out_dir}")
    print(f"Anonymizer: EvidenceTargetedAnonymizer (new)")
    print(f"Attack models: {model_a.config.name} (analytical) + {model_b.config.name} (sociolinguistic)")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1: Parallel attacks on original text
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1: Parallel Inference on Original Text")
    print("=" * 60)

    results_original = run_parallel_inference(
        profiles, model_a, model_b, cfg,
        prompt_strategy_a=create_prompts_analytical,
        prompt_strategy_b=create_prompts_creative,
    )

    # Store merged inference for the evidence-targeted anonymizer to read.
    # Key must match anon_model.config.name (and inference_model.name, which are
    # both "gpt-4o" in the default config) so get_next_steps() returns "anonymize".
    for profile in profiles:
        merged = results_original[profile.username]["merged"]
        profile.get_latest_comments().predictions[
            cfg.task_config.inference_model.name
        ] = merged
        profile.get_latest_comments().predictions[
            cfg.task_config.inference_model.name
        ]["full_answer"] = "PARALLEL_MERGED"

    with open(f"{out_dir}/parallel_inference_original.json", "w") as f:
        json.dump(results_original, f, indent=2, default=str)

    for profile in profiles:
        with open(f"{out_dir}/inference_0.jsonl", "a") as f:
            f.write(json.dumps(profile.to_json()) + "\n")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2: Evidence-targeted anonymization  ← this is the key difference
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2: Evidence-Targeted Anonymization")
    print("  (anonymizer receives both attackers' reasoning chains + explicit")
    print("   evidence identification instructions)")
    print("=" * 60)

    anonymize(profiles, anonymizer, cfg)

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 3: Utility scoring
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 3: Utility Scoring")
    print("=" * 60)

    score_utility(profiles, util_model, cfg)

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 4: Parallel attacks on anonymized text
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 4: Parallel Inference on Anonymized Text")
    print("=" * 60)

    results_anonymized = run_parallel_inference(
        profiles, model_a, model_b, cfg,
        prompt_strategy_a=create_prompts_analytical,
        prompt_strategy_b=create_prompts_creative,
    )

    for profile in profiles:
        merged = results_anonymized[profile.username]["merged"]
        profile.get_latest_comments().predictions[
            cfg.task_config.inference_model.name
        ] = merged

    with open(f"{out_dir}/parallel_inference_anonymized.json", "w") as f:
        json.dump(results_anonymized, f, indent=2, default=str)

    for profile in profiles:
        with open(f"{out_dir}/inference_1.jsonl", "a") as f:
            f.write(json.dumps(profile.to_json()) + "\n")

    # ──────────────────────────────────────────────────────────────────────────
    # ANALYSIS
    # ──────────────────────────────────────────────────────────────────────────
    analysis_original  = compare_attacks(results_original,  profiles)
    analysis_anonymized = compare_attacks(results_anonymized, profiles)

    full_analysis = {
        "original_text":  analysis_original,
        "anonymized_text": analysis_anonymized,
        "comparison": compare_before_after(results_original, results_anonymized, profiles),
    }

    with open(f"{out_dir}/parallel_inference_analysis.json", "w") as f:
        json.dump(full_analysis, f, indent=2, default=str)

    print_analysis_summary(full_analysis)

    # Paper-aligned metrics
    run_paper_metrics(out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evidence-Targeted Anonymization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config_path", type=str, required=True,
        help="Path to YAML config (e.g. configs/anonymization/evidence_targeted.yaml)",
    )
    parser.add_argument(
        "--paper_metrics_only", action="store_true",
        help="Skip pipeline, only recompute paper metrics from existing output directory",
    )
    args = parser.parse_args()

    cfg = read_config_from_yaml(args.config_path)
    seed_everything(cfg.seed)
    set_credentials(cfg)

    if args.paper_metrics_only:
        run_paper_metrics(cfg.task_config.outpath)
        return

    run_evidence_targeted_pipeline(cfg)


if __name__ == "__main__":
    main()
