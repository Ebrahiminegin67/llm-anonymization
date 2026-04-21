"""
Self-contained runner for TAB (Text Anonymization Benchmark) anonymization.

Adapts the LLM anonymization approach from "Large Language Models are Advanced
Anonymizers" (ICLR 2025) to work on the TAB dataset of ECHR court documents.

Usage:
    python run_tab.py --stats_only --split test
    python run_tab.py --model gpt-4o --split test --max_docs 10
    python run_tab.py --evaluate --results_path anonymized_results/tab/results.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ──────────────────────────────────────────────────────
#  TAB Data Loader
# ──────────────────────────────────────────────────────

TAB_ENTITY_TYPES = ["PERSON", "CODE", "LOC", "ORG", "DEM", "DATETIME", "QUANTITY", "MISC"]
MASK_TYPES = {"DIRECT", "QUASI"}

BASE_URL = "https://raw.githubusercontent.com/NorskRegnesentral/text-anonymization-benchmark/master"
TAB_FILES = {"train": "echr_train.json", "dev": "echr_dev.json", "test": "echr_test.json"}

MAX_CHUNK_CHARS = 3500


@dataclass
class EntityMention:
    entity_mention_id: str
    entity_type: str
    start_offset: int
    end_offset: int
    span_text: str
    identifier_type: str
    entity_id: str
    edit_type: str = ""
    confidential_status: str = ""

    @property
    def should_mask(self) -> bool:
        return self.identifier_type in MASK_TYPES


@dataclass
class TABDocument:
    doc_id: str
    text: str
    annotations: List[EntityMention]
    dataset_type: str
    task: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def entities_to_mask(self) -> List[EntityMention]:
        return [e for e in self.annotations if e.should_mask]

    @property
    def entity_types_present(self) -> List[str]:
        return list(set(e.entity_type for e in self.entities_to_mask))

    def get_masked_text(self) -> str:
        sorted_entities = sorted(self.entities_to_mask, key=lambda e: e.start_offset, reverse=True)
        masked = self.text
        for entity in sorted_entities:
            replacement = f"[{entity.entity_type}]"
            masked = masked[:entity.start_offset] + replacement + masked[entity.end_offset:]
        return masked

    def get_entity_summary(self) -> str:
        type_counts: Dict[str, int] = {}
        for entity in self.entities_to_mask:
            type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1
        return ", ".join(f"{count} {etype}" for etype, count in sorted(type_counts.items()))

    def get_annotations_in_range(self, start: int, end: int) -> List[EntityMention]:
        return [e for e in self.annotations if e.start_offset >= start and e.end_offset <= end]


def parse_document(doc_json: Dict[str, Any]) -> TABDocument:
    annotations = []
    for ann_key, ann_data in doc_json.get("annotations", {}).items():
        # Structure: annotations -> annotatorN -> entity_mentions -> [list]
        entity_mentions = ann_data.get("entity_mentions", [])
        for mention_data in entity_mentions:
            if not isinstance(mention_data, dict):
                continue
            annotations.append(EntityMention(
                entity_mention_id=mention_data.get("entity_mention_id", ""),
                entity_type=mention_data.get("entity_type", "MISC"),
                start_offset=mention_data.get("start_offset", 0),
                end_offset=mention_data.get("end_offset", 0),
                span_text=mention_data.get("span_text", ""),
                identifier_type=mention_data.get("identifier_type", "NO_MASK"),
                entity_id=mention_data.get("entity_id", ""),
                edit_type=mention_data.get("edit_type", ""),
                confidential_status=mention_data.get("confidential_status", ""),
            ))
    return TABDocument(
        doc_id=doc_json.get("doc_id", ""),
        text=doc_json.get("text", ""),
        annotations=annotations,
        dataset_type=doc_json.get("dataset_type", ""),
        task=doc_json.get("task", ""),
        meta=doc_json.get("meta", {}),
    )


def download_tab_data(output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for split, filename in TAB_FILES.items():
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            url = f"{BASE_URL}/{filename}"
            print(f"Downloading {split} split from {url}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"  Saved to {filepath}")
        else:
            print(f"  {split} split already exists at {filepath}")
        paths[split] = filepath
    return paths


def load_tab_split(filepath: str) -> List[TABDocument]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [parse_document(doc) for doc in data]


def load_tab_dataset(data_dir: str, splits: Optional[List[str]] = None, download: bool = True):
    if splits is None:
        splits = ["train", "dev", "test"]
    if download:
        download_tab_data(data_dir)
    dataset = {}
    for split in splits:
        filepath = os.path.join(data_dir, TAB_FILES[split])
        if os.path.exists(filepath):
            docs = load_tab_split(filepath)
            dataset[split] = docs
            print(f"Loaded {len(docs)} documents from {split} split")
        else:
            print(f"Warning: {filepath} not found, skipping {split} split")
    return dataset


def get_document_stats(docs: List[TABDocument]) -> Dict[str, Any]:
    total_entities = sum(len(d.annotations) for d in docs)
    mask_entities = sum(len(d.entities_to_mask) for d in docs)
    avg_text_len = sum(len(d.text) for d in docs) / max(len(docs), 1)
    type_counts: Dict[str, int] = {}
    for doc in docs:
        for entity in doc.entities_to_mask:
            type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1
    return {
        "num_documents": len(docs),
        "total_entities": total_entities,
        "entities_to_mask": mask_entities,
        "avg_text_length": int(avg_text_len),
        "entity_type_distribution": type_counts,
    }


# ──────────────────────────────────────────────────────
#  LLM Anonymization
# ──────────────────────────────────────────────────────

def create_tab_prompt(doc, text_chunk, entities_in_chunk, prompt_level=3):
    entities_to_mask = [e for e in entities_in_chunk if e.should_mask]
    entity_info = ""
    for e in entities_to_mask:
        entity_info += f'  - "{e.span_text}" ({e.entity_type}, {e.identifier_type})\n'

    if prompt_level == 1:
        system_prompt = (
            "You are a helpful assistant that anonymizes legal documents by replacing "
            "personal identifiers with appropriate placeholders."
        )
        header = (
            "Below is an excerpt from a court case document. Replace all personal "
            "identifiers (names, locations, organizations, dates, etc.) with "
            "placeholders like [PERSON], [LOC], [ORG], [DATETIME], etc."
        )
        footer = "Write a single # on a new line, then return the anonymized text."
    elif prompt_level == 2:
        system_prompt = (
            "You are an expert legal document anonymizer. You carefully replace "
            "personal identifiers with category placeholders while preserving the "
            "document's structure, meaning, and legal reasoning. You only anonymize "
            "information that could identify specific individuals."
        )
        header = (
            "Below is an excerpt from a European Court of Human Rights case. "
            "Your task is to anonymize it by replacing personal identifiers with "
            "category placeholders (e.g. [PERSON], [LOC], [ORG], [DATETIME], [CODE], "
            "[DEM], [QUANTITY], [MISC]). Keep the rest of the text exactly as is."
        )
        footer = (
            "Write a single # on a new line, then return the anonymized text. "
            "Replace only identifying information, keep everything else intact."
        )
    else:  # level 3
        system_prompt = (
            "You are an expert legal document anonymizer with deep experience in "
            "privacy law and the European Court of Human Rights. You carefully "
            "identify and replace personal identifiers with category placeholders "
            "while preserving the document's structure, meaning, and legal reasoning.\n\n"
            "Replacement categories:\n"
            "- [PERSON]: Names of individuals (applicants, judges, lawyers, witnesses)\n"
            "- [LOC]: Locations (cities, countries, addresses, regions)\n"
            "- [ORG]: Organizations (courts, companies, government bodies)\n"
            "- [DATETIME]: Dates, times, time periods\n"
            "- [CODE]: Case numbers, reference codes, ID numbers\n"
            "- [DEM]: Demographic info (nationality, ethnicity, profession when identifying)\n"
            "- [QUANTITY]: Amounts, measurements, specific numbers\n"
            "- [MISC]: Other identifying information\n\n"
            "Rules:\n"
            "- Replace DIRECT identifiers (uniquely identify someone) always\n"
            "- Replace QUASI identifiers (could identify in combination) when they pose risk\n"
            "- Always replace [CODE] and [ORG] unconditionally — case numbers, reference codes, "
            "court names, and all organization names must always be masked, even if they appear generic\n"
            "- Keep generic legal terms, article references, and non-identifying text intact\n"
            "- Use the same placeholder for repeated mentions of the same entity "
            "(e.g. [PERSON_1], [PERSON_2] for different people)"
        )
        header = (
            "Below is an excerpt from a European Court of Human Rights case, "
            "followed by a list of identified personal entities that should be masked.\n"
            "Your task is to anonymize the text by replacing these entities with "
            "appropriate category placeholders."
        )
        footer = (
            "First, briefly note which entities you will replace and why. "
            "Then write a single # on a new line, followed by the anonymized text. "
            "Replace only the identified entities, keep everything else exactly as is."
        )

    intermediate = f"Document excerpt:\n\n{text_chunk}\n"
    if entities_to_mask:
        intermediate += f"\nIdentified entities to anonymize:\n{entity_info}"
    else:
        intermediate += (
            "\nNo specific entities were pre-identified. "
            "Please identify and replace any personal identifiers you find."
        )

    return {
        "system_prompt": system_prompt,
        "user_prompt": f"{header}\n\n{intermediate}\n\n{footer}",
    }


def chunk_document(doc, max_chars=MAX_CHUNK_CHARS):
    text = doc.text
    if len(text) <= max_chars:
        return [(text, doc.annotations)]

    chunks = []
    paragraphs = text.split("\n")
    current_chunk = ""
    current_start = 0

    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 > max_chars and current_chunk:
            chunk_end = current_start + len(current_chunk)
            entities = doc.get_annotations_in_range(current_start, chunk_end)
            chunks.append((current_chunk, entities))
            current_start = chunk_end
            current_chunk = para + "\n"
        else:
            current_chunk += para + "\n"

    if current_chunk.strip():
        chunk_end = current_start + len(current_chunk)
        entities = doc.get_annotations_in_range(current_start, chunk_end)
        chunks.append((current_chunk, entities))

    return chunks


def parse_anonymized_response(answer: str) -> str:
    parts = answer.split("\n#")
    if len(parts) >= 2:
        return parts[-1].strip()

    patterns = [
        r"(?:anonymized|masked|redacted)\s*(?:text|version|document)\s*[:]\s*\n(.*)",
        r"^#\s*\n(.*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

    return answer.strip()


def call_openai(system_prompt: str, user_prompt: str, model: str, temperature: float = 0.1) -> str:
    import openai

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=4000,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def anonymize_document(doc, model_name, prompt_level=3, temperature=0.1, max_chunk_chars=MAX_CHUNK_CHARS):
    chunks = chunk_document(doc, max_chunk_chars)
    anonymized_chunks = []

    for text_chunk, entities in chunks:
        prompt = create_tab_prompt(doc, text_chunk, entities, prompt_level)
        answer = call_openai(prompt["system_prompt"], prompt["user_prompt"], model_name, temperature)
        anonymized_text = parse_anonymized_response(answer)
        anonymized_chunks.append(anonymized_text)

    anonymized_full = "\n".join(anonymized_chunks)

    return {
        "doc_id": doc.doc_id,
        "original_text": doc.text,
        "anonymized_text": anonymized_full,
        "ground_truth_masked": doc.get_masked_text(),
        "num_entities_to_mask": len(doc.entities_to_mask),
        "entity_summary": doc.get_entity_summary(),
        "num_chunks": len(chunks),
        "model_name": model_name,
        "prompt_level": prompt_level,
    }


def anonymize_documents(docs, model_name, prompt_level=3, temperature=0.1,
                        max_chunk_chars=MAX_CHUNK_CHARS, output_path=None, max_docs=None):
    if max_docs is not None:
        docs = docs[:max_docs]

    results = []
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Resume support
    existing_ids = set()
    if output_path and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing = json.loads(line)
                    existing_ids.add(existing["doc_id"])
                    results.append(existing)
                except (json.JSONDecodeError, KeyError):
                    pass
        if existing_ids:
            print(f"Found {len(existing_ids)} existing results, resuming...")

    for i, doc in enumerate(docs):
        if doc.doc_id in existing_ids:
            continue

        print(f"[{i+1}/{len(docs)}] Anonymizing {doc.doc_id} "
              f"({len(doc.text)} chars, {len(doc.entities_to_mask)} entities)")

        try:
            result = anonymize_document(doc, model_name, prompt_level, temperature, max_chunk_chars)
            results.append(result)

            if output_path:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"  Error processing {doc.doc_id}: {e}")
            error_result = {"doc_id": doc.doc_id, "error": str(e)}
            results.append(error_result)

    return results


# ──────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────

def count_entity_types_in_text(text, pattern=r"\[([A-Z_]+?)(?:_\d+)?\]"):
    matches = re.findall(pattern, text)
    counts: Dict[str, int] = {}
    for m in matches:
        counts[m] = counts.get(m, 0) + 1
    return counts


def evaluate_entity_detection(doc, anonymized_text):
    entities_to_mask = doc.entities_to_mask
    seen_spans: Set[str] = set()
    results_by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "masked": 0, "missed": 0})
    overall = {"total": 0, "masked": 0, "missed": 0}

    for entity in entities_to_mask:
        span = entity.span_text.strip()
        if not span or span in seen_spans:
            continue
        seen_spans.add(span)

        entity_type = entity.entity_type
        results_by_type[entity_type]["total"] += 1
        overall["total"] += 1

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

    recall = overall["masked"] / max(overall["total"], 1)
    placeholder_counts = count_entity_types_in_text(anonymized_text)
    total_placeholders = sum(placeholder_counts.values())
    precision = min(overall["masked"] / max(total_placeholders, 1), 1.0)
    f1 = (2 * precision * recall / max(precision + recall, 1e-8)) if (precision + recall) > 0 else 0.0

    per_type_metrics = {}
    for etype, counts in results_by_type.items():
        r = counts["masked"] / max(counts["total"], 1)
        per_type_metrics[etype] = {
            "total": counts["total"], "masked": counts["masked"],
            "missed": counts["missed"], "recall": round(r, 3),
        }

    return {
        "overall": {
            "total_entities": overall["total"], "masked": overall["masked"],
            "missed": overall["missed"], "recall": round(recall, 3),
            "precision_estimate": round(precision, 3), "f1_estimate": round(f1, 3),
            "total_placeholders_inserted": total_placeholders,
        },
        "per_type": per_type_metrics,
        "placeholder_distribution": placeholder_counts,
    }


def evaluate_text_preservation(original, anonymized):
    orig_words = set(original.lower().split())
    anon_words = set(anonymized.lower().split())
    if not orig_words:
        return {"word_retention": 0.0, "structure_similarity": 0.0}
    common = orig_words & anon_words
    word_retention = len(common) / len(orig_words)
    orig_paras = len([p for p in original.split("\n") if p.strip()])
    anon_paras = len([p for p in anonymized.split("\n") if p.strip()])
    structure_sim = min(orig_paras, anon_paras) / max(orig_paras, 1)
    return {
        "word_retention": round(word_retention, 3),
        "structure_similarity": round(structure_sim, 3),
    }


def evaluate_batch(docs, results):
    doc_map = {d.doc_id: d for d in docs}
    result_map = {r["doc_id"]: r for r in results if "anonymized_text" in r}

    evaluations = []
    for doc_id, result in result_map.items():
        if doc_id not in doc_map:
            continue
        doc = doc_map[doc_id]
        entity_eval = evaluate_entity_detection(doc, result["anonymized_text"])
        preservation_eval = evaluate_text_preservation(doc.text, result["anonymized_text"])
        evaluations.append({
            "doc_id": doc_id,
            "entity_evaluation": entity_eval,
            "text_preservation": preservation_eval,
        })

    if not evaluations:
        return {"error": "No valid evaluation pairs found"}

    total_entities = sum(e["entity_evaluation"]["overall"]["total_entities"] for e in evaluations)
    total_masked = sum(e["entity_evaluation"]["overall"]["masked"] for e in evaluations)
    total_missed = sum(e["entity_evaluation"]["overall"]["missed"] for e in evaluations)
    avg_recall = total_masked / max(total_entities, 1)
    avg_word_retention = sum(e["text_preservation"]["word_retention"] for e in evaluations) / len(evaluations)
    avg_structure_sim = sum(e["text_preservation"]["structure_similarity"] for e in evaluations) / len(evaluations)

    per_type_agg: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "masked": 0, "missed": 0})
    for e in evaluations:
        for etype, counts in e["entity_evaluation"]["per_type"].items():
            per_type_agg[etype]["total"] += counts["total"]
            per_type_agg[etype]["masked"] += counts["masked"]
            per_type_agg[etype]["missed"] += counts["missed"]

    per_type_recall = {}
    for etype, counts in per_type_agg.items():
        per_type_recall[etype] = {
            "total": counts["total"], "masked": counts["masked"],
            "recall": round(counts["masked"] / max(counts["total"], 1), 3),
        }

    return {
        "num_documents_evaluated": len(evaluations),
        "aggregate": {
            "total_entities": total_entities, "total_masked": total_masked,
            "total_missed": total_missed, "overall_recall": round(avg_recall, 3),
            "avg_word_retention": round(avg_word_retention, 3),
            "avg_structure_similarity": round(avg_structure_sim, 3),
        },
        "per_type_recall": per_type_recall,
        "per_document": evaluations,
    }


def print_evaluation_summary(eval_results):
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


# ──────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM anonymization on the TAB dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--provider", type=str, default="openai", help="Model provider")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--max_docs", type=int, default=5, help="Max documents to process")
    parser.add_argument("--prompt_level", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--max_chunk_chars", type=int, default=3500)
    parser.add_argument("--data_dir", type=str, default="data/tab")
    parser.add_argument("--output_dir", type=str, default="anonymized_results/tab")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--evaluate", action="store_true", help="Only evaluate existing results")
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--stats_only", action="store_true", help="Only print dataset stats")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Load config from YAML if provided
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        args.model = cfg.get("model", {}).get("name", args.model)
        args.provider = cfg.get("model", {}).get("provider", args.provider)
        args.temperature = cfg.get("model", {}).get("temperature", args.temperature)
        args.split = cfg.get("split", args.split)
        args.max_docs = cfg.get("max_docs", args.max_docs)
        args.prompt_level = cfg.get("prompt_level", args.prompt_level)
        args.data_dir = cfg.get("data_dir", args.data_dir)
        args.output_dir = cfg.get("output_dir", args.output_dir)

    print("=" * 60)
    print("TAB Anonymization Pipeline")
    print("=" * 60)
    print(f"  Model: {args.model} ({args.provider})")
    print(f"  Split: {args.split}")
    print(f"  Max docs: {args.max_docs}")
    print(f"  Prompt level: {args.prompt_level}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print()

    # Load TAB dataset
    dataset = load_tab_dataset(data_dir=args.data_dir, splits=[args.split], download=not args.no_download)
    docs = dataset.get(args.split, [])
    if not docs:
        print(f"Error: No documents found for split '{args.split}'")
        sys.exit(1)

    # Print stats
    stats = get_document_stats(docs)
    print(f"\nDataset statistics ({args.split}):")
    print(f"  Documents: {stats['num_documents']}")
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Entities to mask: {stats['entities_to_mask']}")
    print(f"  Avg text length: {stats['avg_text_length']} chars")
    print(f"  Entity distribution:")
    for etype, count in sorted(stats['entity_type_distribution'].items()):
        print(f"    {etype:12s}: {count}")
    print()

    if args.stats_only:
        return

    # Set up API credentials
    try:
        sys.path.insert(0, ".")
        import credentials
        import openai
        openai.api_key = credentials.openai_api_key
        if credentials.openai_org:
            openai.organization = credentials.openai_org
    except (ImportError, AttributeError):
        import openai
        if not openai.api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                print("Error: Set OPENAI_API_KEY or configure credentials.py")
                sys.exit(1)
            openai.api_key = api_key

    # Evaluation-only mode
    if args.evaluate:
        results_path = args.results_path or os.path.join(args.output_dir, "results.jsonl")
        if not os.path.exists(results_path):
            print(f"Error: Results file not found: {results_path}")
            sys.exit(1)
        results = []
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
        eval_results = evaluate_batch(docs, results)
        print_evaluation_summary(eval_results)
        eval_path = os.path.join(args.output_dir, "evaluation.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to {eval_path}")
        return

    # Run anonymization
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.jsonl")

    results = anonymize_documents(
        docs=docs,
        model_name=args.model,
        prompt_level=args.prompt_level,
        temperature=args.temperature,
        max_chunk_chars=args.max_chunk_chars,
        output_path=results_path,
        max_docs=args.max_docs,
    )

    print(f"\nAnonymization complete. {len(results)} documents processed.")
    print(f"Results saved to {results_path}")

    # Run evaluation
    eval_results = evaluate_batch(docs, results)
    print_evaluation_summary(eval_results)

    eval_path = os.path.join(args.output_dir, "evaluation.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"Detailed evaluation saved to {eval_path}")


if __name__ == "__main__":
    main()
