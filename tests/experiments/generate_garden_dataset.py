#!/usr/bin/env python3
"""
Generate a garden NP/Z v-trans (mod) dataset with clean/corrupted pairs.
Clean uses a transitive verb; corrupted uses an intransitive verb.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate garden_npz_v-trans_mod dataset.")
    p.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    p.add_argument("--n", type=int, default=300, help="Number of samples.")
    p.add_argument("--seed", type=int, default=1, help="Random seed.")
    p.add_argument("--model-name", type=str, default="gpt2", help="Tokenizer model name.")
    p.add_argument("--model-path", type=str, default=None, help="Local tokenizer path.")
    return p.parse_args()


def is_single_token(tokenizer: AutoTokenizer, word: str) -> bool:
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    return len(ids) == 1


def filter_single_token(tokenizer: AutoTokenizer, words: List[str]) -> List[str]:
    return [w for w in words if is_single_token(tokenizer, w)]


def build_samples(
    subjects: List[str],
    objects: List[str],
    rel_clauses: List[str],
    trans_verbs: List[str],
    intrans_verbs: List[str],
    n: int,
    seed: int,
) -> List[Tuple[str, str, str, str]]:
    rng = random.Random(seed)
    samples: List[Tuple[str, str, str, str]] = []
    attempts = 0
    max_attempts = n * 20

    templates = [
        "As the {subj} {verb} the {obj} who {rel}",
        "While the {subj} {verb} the {obj} who {rel}",
        "After the {subj} {verb} the {obj} who {rel}",
        "When the {subj} {verb} the {obj} who {rel}",
    ]

    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        subj = rng.choice(subjects)
        obj = rng.choice(objects)
        rel = rng.choice(rel_clauses)
        v_t = rng.choice(trans_verbs)
        v_i = rng.choice(intrans_verbs)
        tmpl = rng.choice(templates)
        clean = tmpl.format(subj=subj, verb=v_t, obj=obj, rel=rel)
        corrupted = tmpl.format(subj=subj, verb=v_i, obj=obj, rel=rel)
        row = (clean, corrupted, v_t, v_i, tmpl)
        if row in samples:
            continue
        samples.append(row)

    if len(samples) < n:
        raise RuntimeError(
            f"Only generated {len(samples)} samples; increase vocab or reduce n."
        )
    return samples


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    model_name = args.model_path or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    subjects = [
        "criminal",
        "detective",
        "tourist",
        "driver",
        "chef",
        "student",
        "teacher",
        "sailor",
        "doctor",
        "soldier",
        "pilot",
        "farmer",
        "lawyer",
        "artist",
        "scientist",
    ]
    objects = [
        "woman",
        "man",
        "teacher",
        "student",
        "neighbor",
        "tourist",
        "doctor",
        "chef",
        "singer",
        "child",
        "clerk",
        "driver",
        "writer",
        "gardener",
    ]
    rel_clauses = [
        "told bad jokes",
        "won the prize",
        "left early",
        "forgot the keys",
        "lost the ticket",
        "missed the train",
        "wrote a letter",
        "made a mistake",
        "broke the rule",
        "shouted too loud",
        "paid the bill",
        "answered late",
        "called the office",
        "spoke too softly",
        "waited outside",
    ]

    trans_verbs = [
        "shot",
        "killed",
        "attacked",
        "chased",
        "punished",
        "arrested",
        "blamed",
        "photographed",
        "mocked",
        "insulted",
        "questioned",
        "hired",
        "fired",
        "rescued",
        "escorted",
        "greeted",
        "surprised",
        "frightened",
        "warned",
        "watched",
        "followed",
        "grabbed",
        "pushed",
        "pulled",
        "helped",
        "visited",
        "called",
        "ignored",
        "betrayed",
        "supported",
    ]
    intrans_verbs = [
        "slept",
        "laughed",
        "sneezed",
        "coughed",
        "smiled",
        "stumbled",
        "yawned",
        "shivered",
        "waited",
        "lingered",
        "panicked",
        "hesitated",
        "fainted",
        "nodded",
        "relaxed",
        "dozed",
        "vanished",
        "blinked",
        "paused",
        "sighed",
        "wandered",
        "arrived",
        "departed",
        "swam",
        "jogged",
        "danced",
        "knelt",
        "tripped",
        "collapsed",
        "shouted",
    ]

    trans_verbs = filter_single_token(tokenizer, trans_verbs)
    intrans_verbs = filter_single_token(tokenizer, intrans_verbs)

    if not trans_verbs or not intrans_verbs:
        raise RuntimeError("Filtered verb list is empty after single-token constraint.")

    samples = build_samples(
        subjects,
        objects,
        rel_clauses,
        trans_verbs,
        intrans_verbs,
        args.n,
        args.seed,
    )
    rng.shuffle(samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "clean",
                "corrupted",
                "correct_token",
                "incorrect_token",
                "transitive_verb",
                "intransitive_verb",
                "template",
                "template_text",
            ]
        )
        for clean, corrupted, v_t, v_i, tmpl in samples:
            # Original (clean=transitive -> correct=for)
            writer.writerow(
                [clean, corrupted, "for", "was", v_t, v_i, "garden_npz_v-trans_mod", tmpl]
            )
            # Flipped (clean=intransitive -> correct=was)
            writer.writerow(
                [corrupted, clean, "was", "for", v_t, v_i, "garden_npz_v-trans_mod", tmpl]
            )

    print(f"Wrote {len(samples)} samples to {args.output}")
    print(f"Transitive verbs (single-token): {len(trans_verbs)}")
    print(f"Intransitive verbs (single-token): {len(intrans_verbs)}")


if __name__ == "__main__":
    main()
