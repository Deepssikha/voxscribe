"""Voice-command parser using fuzzy matching, so small recognition errors
(for example "save notes" instead of "save note") still trigger the command."""
import difflib
import re

COMMANDS = {
    "stop listening": "stop",
    "save note": "save_note",
    "clear text": "clear",
    "copy text": "copy",
    "download text": "download",
    "open settings": "settings",
}
SEARCH_PREFIX = "search for"


def normalise(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def match_command(text: str, threshold: float = 0.8) -> dict | None:
    """Return {'action', 'phrase', 'score', ...} or None for plain dictation."""
    clean = normalise(text)
    if not clean:
        return None
    words = clean.split()

    if len(words) >= 3:
        score = _ratio(" ".join(words[:2]), SEARCH_PREFIX)
        if score >= threshold:
            return {"action": "search", "phrase": SEARCH_PREFIX,
                    "query": " ".join(words[2:]), "score": round(score, 2)}

    best = None
    for phrase, action in COMMANDS.items():
        score = _ratio(clean, phrase)
        if score >= threshold and (best is None or score > best["score"]):
            best = {"action": action, "phrase": phrase, "score": round(score, 2)}
    return best
