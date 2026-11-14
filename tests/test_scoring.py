"""Checks for the answer scoring, so a wrong score can't sneak into the results."""

from compare import cost_usd, is_correct, version_of
from pathlib import Path


def test_exact_and_inside_a_sentence():
    assert is_correct("Canberra", ["canberra"])
    assert is_correct("The answer is 391.", ["391"])


def test_whole_words_only():
    assert not is_correct("0.05", ["5"])        # "05" is not "5"
    assert not is_correct("3911", ["391"])


def test_wrong_answer():
    assert not is_correct("Sydney", ["canberra"])
    assert not is_correct("ERROR: TimeoutException", ["thursday"])


def test_local_models_cost_nothing():
    assert cost_usd("ollama", 1000, 1000) == 0.0


def test_version_from_file_name():
    assert version_of(Path("data/questions_v1.jsonl")) == "v1"
