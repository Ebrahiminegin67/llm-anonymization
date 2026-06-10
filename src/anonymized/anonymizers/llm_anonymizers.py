from abc import ABC, abstractmethod
from typing import List, Iterator, Dict, Any
from copy import deepcopy
from .anonymizer import Anonymizer

from src.utils.string_utils import select_closest
from src.configs import Config, AnonymizerConfig
from src.prompts import Prompt
from src.reddit.reddit_utils import type_to_str
from src.reddit.reddit_types import Profile
from src.models.model import BaseModel
from src.reddit.reddit_types import Comment, AnnotatedComments
import re


class LLMFullAnonymizer(Anonymizer):
    def __init__(
        self,
        cfg: AnonymizerConfig,
        model: BaseModel,
    ):
        self.model = model
        self.cfg = cfg

    def _create_anon_prompt(self, profile: Profile) -> List[Prompt]:

        if self.cfg.prompt_level == 1:  # Naive
            system_prompt = "You are an helpful assistant that helps users to anonymize their comments."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments."
            footer = "In a new line write first a single # and then return the anonymized text."
        elif self.cfg.prompt_level == 2:  # Better prompt and explanation
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact."
            footer = "In a new line write first a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information."
        elif self.cfg.prompt_level == 3:  # CoT
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact."
            footer = "In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information."
        elif self.cfg.prompt_level == 4:  # Implicit-aware CoT
            system_prompt = (
                "You are an expert anonymizer. You carefully adapt text to prevent personal "
                "identity inference from BOTH explicit facts AND implicit writing style cues. "
                "You follow instructions precisely and change as little as possible, but when "
                "inferences are style-based you adjust vocabulary or phrasing as needed."
            )
            header = (
                "Below are Reddit comments by one user and personal inferences made from them. "
                "Your task is to anonymize the comments to prevent these inferences.\n\n"
                "Address BOTH types of signals:\n"
                "1. EXPLICIT signals: Remove or generalize directly stated facts — "
                "locations (e.g. city names), currency (e.g. CHF, £), income statements, "
                "job titles, ages, relationship status, self-descriptions.\n"
                "2. IMPLICIT signals: When an inference is based on HOW the person writes, "
                "adjust the writing style. Examples: replace region-specific slang with neutral "
                "alternatives (e.g. 'mate' → 'friend'), generalize strong cultural or language "
                "references (e.g. non-English words that reveal national background), neutralize "
                "vocabulary that strongly signals education level or professional field.\n\n"
                "Only change what is necessary. Do not invent new information — generalize instead. "
                "Keep the original meaning as much as possible, but style adjustments are acceptable when needed to remove implicit signals."
            )
            footer = (
                "First, briefly list the changes you will make — separating explicit signal "
                "removals from implicit style adjustments. Then write a single # on a new line "
                "followed by the anonymized text. Only change relevant parts."
            )
        elif self.cfg.prompt_level == 5:  # Aggressive sentence rewrite for implicit signals
            system_prompt = (
                "You are an expert anonymizer. You apply two distinct strategies based on signal type:\n"
                "- EXPLICIT signals (stated facts, named entities): substitute or generalize only the specific entity.\n"
                "- IMPLICIT signals (writing style, slang, vocabulary register, cultural references, "
                "syntax patterns): rewrite the ENTIRE sentence in plain, neutral English, "
                "preserving only the factual content. Do NOT preserve tone, humor, slang, or "
                "distinctive vocabulary in those sentences."
            )
            header = (
                "Below are Reddit comments and personal inferences made from them. "
                "Each inference is either fact-based (EXPLICIT) or style-based (IMPLICIT).\n\n"
                "Apply these rules:\n"
                "- EXPLICIT inference → locate the specific entity or stated fact → "
                "substitute or generalize it, leave the rest of the sentence unchanged.\n"
                "- IMPLICIT/STYLE inference → locate the sentence(s) that produce that signal → "
                "rewrite those sentences entirely in neutral, formal English. "
                "Remove slang, regional expressions, humor, technical jargon, non-English words, "
                "and any phrasing that reveals background, age, or education.\n"
                "- Sentences not linked to any inference: leave unchanged.\n\n"
                "Do not invent new facts. Preserve the factual meaning of each sentence."
            )
            footer = (
                "For each inference, state: EXPLICIT or IMPLICIT, which sentence it comes from, "
                "and what you will do. Then write a single # on a new line followed by the "
                "fully anonymized text."
            )

        comments = profile.get_latest_comments().comments
        comment_string = "\n".join([str(c) for c in comments])
        previous_inferences = profile.get_latest_comments().predictions[
            self.model.config.name
        ]
        inference_string = ""
        try:
            for key, inf in previous_inferences.items():
                if key == "full_answer":
                    continue
                if "guess" not in inf:
                    continue
                inference_string += f"Type: {key}\n"
                inference_string += f"Inference: {inf['inference']}\n"
                if inf.get("inference_secondary"):
                    inference_string += f"Supporting inference: {inf['inference_secondary']}\n"
                inference_string += f"Guess: {inf['guess']}\n"
                agreement = inf.get("agreement", "")
                if agreement == "full_agreement":
                    inference_string += "Note: Two independent attack strategies fully agree on this — anonymize aggressively.\n"
                elif agreement == "partial_agreement":
                    inference_string += "Note: Two independent attack strategies partially agree on this.\n"
                cert = inf.get("certainty", "")
                if cert:
                    inference_string += f"Combined certainty: {cert}/5\n"
                inference_string += "\n"
        except Exception as e:
            # Fall back to full answer
            inference_string = previous_inferences["full_answer"]

        intermediate = f"\n\n {comment_string}\n\nInferences:\n\n{inference_string}"

        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate=intermediate,
            footer=footer,
            target=(
                profile.get_relevant_pii()[0]
                if len(profile.get_relevant_pii()) > 0
                else ""
            ),
            original_point=profile,  # type: ignore
            gt=profile.get_relevant_pii(),  # type: ignore
            answer="",
            shots=[],
            id=profile.username,  # type: ignore
        )

        return [prompt]

    def filter_and_align_comments(self, answer: str, op: Profile) -> List[str]:

        try:
            split_answer = answer.split("\n#")

            if len(split_answer) == 1:
                new_comments = answer.strip()
            elif len(split_answer) == 2:
                new_comments = split_answer[1].strip()
            else:
                new_comments = ("\n").join(split_answer)

        except Exception:
            print("Could not split answer", answer)
            new_comments = deepcopy([c.text for c in op.get_latest_comments().comments])
            return new_comments

        new_comments = new_comments.split("\n")

        # Remove all lines that are empty
        new_comments = [c for c in new_comments if len(c) > 0]

        if len(new_comments) != len(op.get_latest_comments().comments):
            print(
                f"Number of comments does not match: {len(new_comments)} vs {len(op.get_latest_comments().comments)}"
            )

            old_comment_ids = [
                -1 for _ in range(len(op.get_latest_comments().comments))
            ]

            used_idx = set({})

            for i, comment in enumerate(op.get_latest_comments().comments):
                closest_match, sim, idx = select_closest(
                    comment.text,
                    new_comments,
                    dist="jaro_winkler",
                    return_idx=True,
                    return_sim=True,
                )

                if idx not in used_idx and sim > 0.5:
                    old_comment_ids[i] = idx
                    used_idx.add(idx)

            selected_comments = []
            for i, idx in enumerate(old_comment_ids):
                if idx == -1:
                    selected_comments.append(op.get_latest_comments().comments[i].text)
                else:
                    selected_comments.append(new_comments[idx])
        else:
            selected_comments = new_comments

        typed_comments = []
        i = 0

        for comment in selected_comments:
            if re.search(r"\d{4}-\d{2}-\d{2}:", comment[:11]) is not None:
                comment = comment[11:].strip()

            old_com = op.get_latest_comments().comments[i]
            new_com = Comment(
                comment, old_com.subreddit, old_com.user, old_com.timestamp
            )
            typed_comments.append(new_com)
            i += 1

        return typed_comments

    def anonymize(self, text: str) -> str:
        pass

    def anonymize_profiles(self, profiles: List[Profile]) -> Iterator[Profile]:

        prompts = []
        for profile in profiles:
            prompts.extend(self._create_anon_prompt(profile))

        for i, res in enumerate(
            self.model.predict_multi(
                prompts, max_workers=self.cfg.max_workers, timeout=120
            )
        ):
            prompt, answer = res
            # for i, prompt in enumerate(prompts):
            #     answer = self.model.predict(prompt)

            op = prompt.original_point
            assert isinstance(op, Profile)
            print(f"{i}".center(50, "="))
            print(prompt.get_prompt())
            op.print_review_pii()
            print(f"{self.model.config.name}\n" + answer)

            typed_comments = self.filter_and_align_comments(answer, op)

            print(typed_comments)

            op.comments.append(AnnotatedComments(typed_comments, op.review_pii, {}, {}))

            yield op


class LLMBaselineAnonymizer(LLMFullAnonymizer):

    def _create_anon_prompt(self, profile: Profile) -> List[Prompt]:

        type_str = ""
        for pii in profile.get_relevant_pii():
            type_str += f"{type_to_str(pii)} "

        system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
        header = f"Below I give you a set of comments from a user. Your task is to anonymize the comments to prevent the inferences of the users {type_str} , while keeping the rest of the text intact."
        footer = "In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information."

        comments = profile.get_latest_comments().comments
        comment_string = "\n".join([str(c) for c in comments])
        previous_infereces = profile.get_latest_comments().predictions[
            self.model.config.name
        ]

        intermediate = f"\n\n {comment_string}\n"

        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate=intermediate,
            footer=footer,
            target=profile.get_relevant_pii()[0],
            original_point=profile,  # type: ignore
            gt=profile.get_relevant_pii(),  # type: ignore
            answer="",
            shots=[],
            id=profile.username,  # type: ignore
        )

        return [prompt]
