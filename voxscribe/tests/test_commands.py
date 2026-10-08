import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from commands import match_command  # noqa: E402


def test_exact_command():
    assert match_command("Save note.")["action"] == "save_note"


def test_fuzzy_command():
    assert match_command("save notes")["action"] == "save_note"


def test_search_query():
    result = match_command("Search for speech recognition")
    assert result["action"] == "search"
    assert result["query"] == "speech recognition"


def test_plain_dictation_is_not_a_command():
    assert match_command("the weather is nice today") is None


def test_empty_text():
    assert match_command("   ") is None
