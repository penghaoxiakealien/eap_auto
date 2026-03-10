#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick sanity checks for token parsing used by auto_gender_terminal_token.py.

Run:
  python tests/experiments/quick_test_gender_token_parsing.py
"""

import sys
from pathlib import Path

# Allow running from repo root without installing as a package.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.experiments.auto_gender_terminal_token import _parse_and_suffix_tokens, normalize_token  # noqa: E402


def check(sentence: str, marked: str, exp_inc: list[str], exp_dec: list[str]) -> None:
    inc, dec = _parse_and_suffix_tokens(sentence, marked)
    inc_n = [normalize_token(x) for x in inc]
    dec_n = [normalize_token(x) for x in dec]
    assert inc_n == [normalize_token(x) for x in exp_inc], (inc, exp_inc)
    assert dec_n == [normalize_token(x) for x in exp_dec], (dec, exp_dec)
    print("OK")
    print("  sentence:", sentence)
    print("  marked:  ", marked)
    print("  inc:", inc, "->", inc_n)
    print("  dec:", dec, "->", dec_n)


def main() -> None:
    # From your recent debug example
    check(
        sentence="Paul and Kelly missed the train. Afterward, Paul insisted because",
        marked="Paul and Kelly missed the train. Afterward, [[Paul]] insisted <<because>>",
        exp_inc=["because"],
        exp_dec=["Paul"],
    )

    # Punctuation as a marked token
    check(
        sentence="Paul and Kelly missed the train. Afterward, Paul insisted because",
        marked="Paul and Kelly missed the train. Afterward[[,]] Paul insisted <<because>>",
        exp_inc=["because"],
        exp_dec=[","],
    )

    # Duplicate name should suffix the first occurrence consistently (IOI-style)
    check(
        sentence="Paul and Kelly talked. Later, Paul complained because",
        marked="[[Paul]] and Kelly talked. Later, <<Paul>> complained because",
        exp_inc=["Paul"],  # will become Paul_1 or Paul_2 internally; normalize_token ignores suffix
        exp_dec=["Paul"],
    )


if __name__ == "__main__":
    main()
