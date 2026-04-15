"""
TAB (Text Anonymization Benchmark) data loader.

Loads ECHR court case documents from the TAB dataset and converts them
into a format compatible with the LLM anonymization pipeline.

Dataset: https://github.com/NorskRegnesentral/text-anonymization-benchmark
Paper: Pilán et al., "The Text Anonymization Benchmark (TAB)", 2022
"""

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# TAB entity categories
TAB_ENTITY_TYPES = [
    "PERSON",
    "CODE",
    "LOC",
    "ORG",
    "DEM",
    "DATETIME",
    "QUANTITY",
    "MISC",
]

# Identifier types that should be masked
MASK_TYPES = {"DIRECT", "QUASI"}

BASE_URL = "https://raw.githubusercontent.com/NorskRegnesentral/text-anonymization-benchmark/master"
TAB_FILES = {
    "train": "echr_train.json",
    "dev": "echr_dev.json",
    "test": "echr_test.json",
}


@dataclass
class EntityMention:
    entity_mention_id: str
    entity_type: str
    start_offset: int
    end_offset: int
    span_text: str
    identifier_type: str  # DIRECT, QUASI, NO_MASK
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
        """Generate the ground truth masked version of the text."""
        # Sort by start offset descending so replacements don't shift offsets
        sorted_entities = sorted(
            self.entities_to_mask, key=lambda e: e.start_offset, reverse=True
        )
        masked = self.text
        for entity in sorted_entities:
            replacement = f"[{entity.entity_type}]"
            masked = masked[: entity.start_offset] + replacement + masked[entity.end_offset :]
        return masked

    def get_entity_summary(self) -> str:
        """Get a summary of entities that need to be masked."""
        type_counts: Dict[str, int] = {}
        for entity in self.entities_to_mask:
            type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1
        return ", ".join(f"{count} {etype}" for etype, count in sorted(type_counts.items()))

    def get_text_snippet(self, max_chars: int = 4000) -> str:
        """Get a truncated version of the text for LLM context windows."""
        if len(self.text) <= max_chars:
            return self.text
        return self.text[:max_chars] + "\n[... text truncated ...]"

    def get_annotations_in_range(self, start: int, end: int) -> List[EntityMention]:
        """Get annotations that fall within a character range."""
        return [
            e
            for e in self.annotations
            if e.start_offset >= start and e.end_offset <= end
        ]


def parse_document(doc_json: Dict[str, Any]) -> TABDocument:
    """Parse a single TAB document from JSON."""
    annotations = []
    for ann_key, ann_data in doc_json.get("annotations", {}).items():
        for mention_key, mention_data in ann_data.items():
            if not isinstance(mention_data, dict):
                continue
            annotations.append(
                EntityMention(
                    entity_mention_id=mention_data.get("entity_mention_id", mention_key),
                    entity_type=mention_data.get("entity_type", "MISC"),
                    start_offset=mention_data.get("start_offset", 0),
                    end_offset=mention_data.get("end_offset", 0),
                    span_text=mention_data.get("span_text", ""),
                    identifier_type=mention_data.get("identifier_type", "NO_MASK"),
                    entity_id=mention_data.get("entity_id", ""),
                    edit_type=mention_data.get("edit_type", ""),
                    confidential_status=mention_data.get("confidential_status", ""),
                )
            )

    return TABDocument(
        doc_id=doc_json.get("doc_id", ""),
        text=doc_json.get("text", ""),
        annotations=annotations,
        dataset_type=doc_json.get("dataset_type", ""),
        task=doc_json.get("task", ""),
        meta=doc_json.get("meta", {}),
    )


def download_tab_data(output_dir: str) -> Dict[str, str]:
    """Download TAB dataset files if not already present."""
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
    """Load a single TAB dataset split."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [parse_document(doc) for doc in data]


def load_tab_dataset(
    data_dir: str, splits: Optional[List[str]] = None, download: bool = True
) -> Dict[str, List[TABDocument]]:
    """Load the TAB dataset, optionally downloading it first.

    Args:
        data_dir: Directory to store/find the TAB data files.
        splits: Which splits to load (default: all).
        download: Whether to download files if missing.

    Returns:
        Dict mapping split name to list of TABDocument objects.
    """
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
    """Get summary statistics for a set of TAB documents."""
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
