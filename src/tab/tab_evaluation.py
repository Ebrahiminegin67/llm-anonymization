"""
Evaluation metrics for TAB anonymization.

Computes entity-level precision, recall, F1 for anonymization quality,
plus text utility metrics (BLEU, ROUGE) for preservation quality.
"""

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src.tab.tab_loader import TABDocument, EntityMention, MASK_TYPES


def extract_replaced_spans(
    original: str, anonymized: str, entity_pattern: str = r"\[([A-Z_]+(?:_\d+)?)\]"
) -> List[Dict[str, str]]:
    """Find spans in the original text that were replaced by placeholders in the anonymized text.

    Uses a heuristic alignment approach since exact offset mapping isn't available
    after LLM rewriting.

    Returns list of dicts with 'placeholder' and approximate 'original_text'.
    """
    placeholders = re.findall(entity_pattern, anonymized)
    return [{"placeholder": p} for p in placeholders]


def count_entity_types_in_text(
    text: str, entity_pattern: str = r"\[([A-Z_]+?)(?:_\d+)?\]"
) -> Dict[str, int]:
    """Count entity type placeholders in anonymized text."""
    matches = re.findall(entity_pattern, text)
    counts: Dict[str, int] = {}
    for m in matches:
        counts[m] = counts.get(m, 0) + 1
    return counts


def evaluate_entity_detection(
    doc: TABDocument,
    anonymized_text: str,
) -> Dict[str, Any]:
    """Evaluate whether entities that should be masked were actually replaced.

    Checks if the original entity span text still appears in the anonymized text.
    If it does, the entity was NOT properly anonymized.

    Returns per-entity-type and overall metrics.
    """
    entities_to_mask = doc.entities_to_mask

    # Track unique span texts to avoid double-counting overlapping annotations
    seen_spans: Set[str] = set()
    results_by_type: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "masked": 0, "missed": 0}
    )
    overall = {"total": 0, "masked": 0, "missed": 0}

    for entity in entities_to_mask:
        span = entity.span_text.strip()
        if not span or span in seen_spans:
            continue
        seen_spans.add(span)

        entity_type = entity.entity_type
        results_by_type[entity_type]["total"] += 1
        overall["total"] += 1

        # Check if the original span text still appears in the anonymized text
        # Use case-sensitive match for names, case-insensitive for others
        if entity_type == "PERSON":
            found = span in anonymized_text
        else:
            found = span.lower() in anonymized_text.lower()

        if found:
            results_by_type[entity_type]["missed"] += 1
            overall["missed"] += 1
        else:
            results_by_type[entity_type]["masked"] += 1
            overall["masked"] += 1

    # Compute recall (% of entities that should be masked that were actually masked)
    recall = overall["masked"] / max(overall["total"], 1)

    # Count placeholder types inserted by the LLM
    placeholder_counts = count_entity_types_in_text(anonymized_text)
    total_placeholders = sum(placeholder_counts.values())

    # Precision estimate: what fraction of inserted placeholders correspond
    # to real entities (rough estimate based on count comparison)
    precision = min(overall["masked"] / max(total_placeholders, 1), 1.0)
    f1 = (2 * precision * recall / max(precision + recall, 1e-8)) if (precision + recall) > 0 else 0.0

    per_type_metrics = {}
    for etype, counts in results_by_type.items():
        r = counts["masked"] / max(counts["total"], 1)
        per_type_metrics[etype] = {
            "total": counts["total"],
            "masked": counts["masked"],
            "missed": counts["missed"],
            "recall": round(r, 3),
        }

    return {
        "overall": {
            "total_entities": overall["total"],
            "masked": overall["masked"],
            "missed": overall["missed"],
            "recall": round(recall, 3),
            "precision_estimate": round(precision, 3),
            "f1_estimate": round(f1, 3),
            "total_placeholders_inserted": total_placeholders,
        },
        "per_type": per_type_metrics,
        "placeholder_distribution": placeholder_counts,
    }


def evaluate_text_preservation(
    original: str, anonymized: str
) -> Dict[str, float]:
    """Evaluate how much of the original text structure is preserved.

    Uses simple overlap metrics (not BLEU/ROUGE which need tokenizers),
    measuring what fraction of original words appear unchanged.
    """
    # Tokenize simply by whitespace
    orig_words = set(original.lower().split())
    anon_words = set(anonymized.lower().split())

    if not orig_words:
        return {"word_overlap": 0.0, "word_retention": 0.0}

    common = orig_words & anon_words
    word_retention = len(common) / len(orig_words)

    # Check structure preservation (paragraph count similarity)
    orig_paras = len([p for p in original.split("\n") if p.strip()])
    anon_paras = len([p for p in anonymized.split("\n") if p.strip()])
    structure_sim = min(orig_paras, anon_paras) / max(orig_paras, 1)

    return {
        "word_retention": round(word_retention, 3),
        "structure_similarity": round(structure_sim, 3),
    }


def evaluate_single_document(
    doc: TABDocument, anonymized_text: str
) -> Dict[str, Any]:
    """Full evaluation of a single anonymized document."""
    entity_eval = evaluate_entity_detection(doc, anonymized_text)
    preservation_eval = evaluate_text_preservation(doc.text, anonymized_text)

    return {
        "doc_id": doc.doc_id,
        "entity_evaluation": entity_eval,
        "text_preservation": preservation_eval,
    }


def evaluate_batch(
    docs: List[TABDocument],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate a batch of anonymization results.

    Args:
        docs: Original TAB documents.
        results: Anonymization results (must have 'doc_id' and 'anonymized_text').

    Returns:
        Aggregated evaluation metrics.
    """
    # Build lookup
    doc_map = {d.doc_id: d for d in docs}
    result_map = {r["doc_id"]: r for r in results if "anonymized_text" in r}

    evaluations = []
    for doc_id, result in result_map.items():
        if doc_id not in doc_map:
            continue
        doc = doc_map[doc_id]
        eval_result = evaluate_single_document(doc, result["anonymized_text"])
        evaluations.append(eval_result)

    if not evaluations:
        return {"error": "No valid evaluation pairs found"}

    # Aggregate metrics
    total_entities = sum(e["entity_evaluation"]["overall"]["total_entities"] for e in evaluations)
    total_masked = sum(e["entity_evaluation"]["overall"]["masked"] for e in evaluations)
    total_missed = sum(e["entity_evaluation"]["overall"]["missed"] for e in evaluations)

    avg_recall = total_masked / max(total_entities, 1)
    avg_word_retention = sum(e["text_preservation"]["word_retention"] for e in evaluations) / len(evaluations)
    avg_structure_sim = sum(e["text_preservation"]["structure_similarity"] for e in evaluations) / len(evaluations)

    # Per-type aggregation
    per_type_agg: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "masked": 0, "missed": 0})
    for e in evaluations:
        for etype, counts in e["entity_evaluation"]["per_type"].items():
            per_type_agg[etype]["total"] += counts["total"]
            per_type_agg[etype]["masked"] += counts["masked"]
            per_type_agg[etype]["missed"] += counts["missed"]

    per_type_recall = {}
    for etype, counts in per_type_agg.items():
        per_type_recall[etype] = {
            "total": counts["total"],
            "masked": counts["masked"],
            "recall": round(counts["masked"] / max(counts["total"], 1), 3),
        }

    return {
        "num_documents_evaluated": len(evaluations),
        "aggregate": {
            "total_entities": total_entities,
            "total_masked": total_masked,
            "total_missed": total_missed,
            "overall_recall": round(avg_recall, 3),
            "avg_word_retention": round(avg_word_retention, 3),
            "avg_structure_similarity": round(avg_structure_sim, 3),
        },
        "per_type_recall": per_type_recall,
        "per_document": evaluations,
    }


def print_evaluation_summary(eval_results: Dict[str, Any]) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("TAB ANONYMIZATION EVALUATION SUMMARY")
    print("=" * 60)

    agg = eval_results.get("aggregate", {})
    print(f"\nDocuments evaluated: {eval_results.get('num_documents_evaluated', 0)}")
    print(f"Total entities to mask: {agg.get('total_entities', 0)}")
    print(f"Successfully masked: {agg.get('total_masked', 0)}")
    print(f"Missed: {agg.get('total_missed', 0)}")
    print(f"\nOverall recall: {agg.get('overall_recall', 0):.1%}")
    print(f"Word retention: {agg.get('avg_word_retention', 0):.1%}")
    print(f"Structure similarity: {agg.get('avg_structure_similarity', 0):.1%}")

    print("\nPer-entity-type recall:")
    for etype, metrics in sorted(eval_results.get("per_type_recall", {}).items()):
        print(f"  {etype:12s}: {metrics['recall']:.1%} ({metrics['masked']}/{metrics['total']})")

    print("=" * 60)
