from __future__ import annotations

import re

from .settings import MAX_FRAGMENT_CHARS

_WHITESPACE_RE = re.compile(r"[ \t]+")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for raw_line in text.split("\n"):
        cleaned_line = _WHITESPACE_RE.sub(" ", raw_line).strip()
        if cleaned_line:
            lines.append(cleaned_line)
    return "\n".join(lines)


def split_text_for_tts(text: str, max_fragment_chars: int = MAX_FRAGMENT_CHARS) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    sentence_candidates = re.split(r"(?<=[\.\?\!…])\s+|\n+", normalized)
    fragments: list[str] = []
    for candidate in sentence_candidates:
        stripped = candidate.strip()
        if not stripped:
            continue
        fragments.extend(_split_overlong_sentence(stripped, max_fragment_chars))

    return fragments


def _split_overlong_sentence(sentence: str, max_fragment_chars: int) -> list[str]:
    if len(sentence) <= max_fragment_chars:
        return [sentence]

    comma_parts = re.split(r"(?<=[,;:])\s+", sentence)
    if len(comma_parts) > 1:
        fragments: list[str] = []
        current = ""
        for part in comma_parts:
            if not current:
                current = part
                continue
            prospective = f"{current} {part}"
            if len(prospective) <= max_fragment_chars:
                current = prospective
            else:
                fragments.append(current)
                current = part
        if current:
            fragments.append(current)
        return fragments

    words = sentence.split(" ")
    fragments = []
    current_words: list[str] = []
    current_length = 0

    for word in words:
        separator = 1 if current_words else 0
        if current_length + len(word) + separator <= max_fragment_chars:
            current_words.append(word)
            current_length += len(word) + separator
            continue

        if current_words:
            fragments.append(" ".join(current_words))
            current_words = []
            current_length = 0

        if len(word) > max_fragment_chars:
            fragments.extend(_slice_long_token(word, max_fragment_chars))
        else:
            current_words = [word]
            current_length = len(word)

    if current_words:
        fragments.append(" ".join(current_words))

    return fragments


def _slice_long_token(token: str, max_fragment_chars: int) -> list[str]:
    return [token[i : i + max_fragment_chars] for i in range(0, len(token), max_fragment_chars)]
