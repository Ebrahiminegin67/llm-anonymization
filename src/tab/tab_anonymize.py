"""
TAB anonymization pipeline using LLMs.

Adapts the LLM anonymization approach from the paper
"Large Language Models are Advanced Anonymizers" to work
on the TAB (Text Anonymization Benchmark) dataset.

Instead of attribute-level anonymization (preventing inference of age, gender, etc.),
this performs entity-level anonymization (replacing PERSON, LOC, ORG, etc.)
on court case documents.
"""

import json
import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from src.configs import ModelConfig
from src.models.model import BaseModel
from src.models.model_factory import get_model
from src.prompts import Prompt
from src.tab.tab_loader import TABDocument, EntityMention, MASK_TYPES


# Max characters per chunk for long documents
MAX_CHUNK_CHARS = 3500


def create_tab_anonymization_prompt(
    doc: TABDocument,
    text_chunk: str,
    entities_in_chunk: List[EntityMention],
    prompt_level: int = 3,
) -> Prompt:
    """Create a prompt for anonymizing a TAB document chunk.

    Args:
        doc: The full TAB document (for context).
        text_chunk: The text chunk to anonymize.
        entities_in_chunk: Entities that appear in this chunk.
        prompt_level: Prompt sophistication level (1=naive, 2=intermediate, 3=advanced).

    Returns:
        A Prompt object ready for the model.
    """
    entities_to_mask = [e for e in entities_in_chunk if e.should_mask]

    # Build entity info string
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
        footer = (
            "Write a single # on a new line, then return the anonymized text."
        )
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
    elif prompt_level == 3:
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

    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate=intermediate,
        footer=footer,
        target=doc.doc_id,
        original_point={"doc_id": doc.doc_id, "text_chunk": text_chunk},
        gt=None,
        answer="",
        shots=[],
        id=doc.doc_id,
    )

    return prompt


def chunk_document(
    doc: TABDocument, max_chars: int = MAX_CHUNK_CHARS
) -> List[Tuple[str, List[EntityMention]]]:
    """Split a document into chunks that fit in the LLM context window.

    Tries to split on paragraph boundaries. Returns list of (text_chunk, entities_in_chunk).
    """
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
    """Extract the anonymized text from the model's response."""
    # Look for text after the # separator
    parts = answer.split("\n#")
    if len(parts) >= 2:
        return parts[-1].strip()

    # Fallback: try to find text after "Anonymized text:" or similar
    patterns = [
        r"(?:anonymized|masked|redacted)\s*(?:text|version|document)\s*[:]\s*\n(.*)",
        r"^#\s*\n(.*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

    # Last resort: return the whole answer
    return answer.strip()


def anonymize_tab_document(
    doc: TABDocument,
    model: BaseModel,
    prompt_level: int = 3,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> Dict[str, Any]:
    """Anonymize a single TAB document using an LLM.

    Args:
        doc: TAB document to anonymize.
        model: LLM model to use.
        prompt_level: Prompt sophistication level.
        max_chunk_chars: Maximum characters per chunk.

    Returns:
        Dict with original text, anonymized text, ground truth, and metadata.
    """
    chunks = chunk_document(doc, max_chunk_chars)
    anonymized_chunks = []

    for text_chunk, entities in chunks:
        prompt = create_tab_anonymization_prompt(
            doc, text_chunk, entities, prompt_level
        )
        answer = model.predict(prompt)
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
        "model_name": model.config.name,
        "prompt_level": prompt_level,
    }


def anonymize_tab_documents(
    docs: List[TABDocument],
    model: BaseModel,
    prompt_level: int = 3,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    output_path: Optional[str] = None,
    max_docs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Anonymize multiple TAB documents.

    Args:
        docs: List of TAB documents to anonymize.
        model: LLM model to use.
        prompt_level: Prompt sophistication level.
        max_chunk_chars: Maximum characters per chunk.
        output_path: Path to write results as JSONL (incremental).
        max_docs: Maximum number of documents to process.

    Returns:
        List of result dicts for each document.
    """
    if max_docs is not None:
        docs = docs[:max_docs]

    results = []
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check for existing results to enable resume
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
        print(f"Found {len(existing_ids)} existing results, resuming...")

    for i, doc in enumerate(docs):
        if doc.doc_id in existing_ids:
            continue

        print(f"[{i+1}/{len(docs)}] Anonymizing document {doc.doc_id} "
              f"({len(doc.text)} chars, {len(doc.entities_to_mask)} entities)")

        try:
            result = anonymize_tab_document(doc, model, prompt_level, max_chunk_chars)
            results.append(result)

            if output_path:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"  Error processing {doc.doc_id}: {e}")
            error_result = {
                "doc_id": doc.doc_id,
                "error": str(e),
                "original_text": doc.text[:200],
            }
            results.append(error_result)

    return results
