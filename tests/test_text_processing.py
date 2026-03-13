from __future__ import annotations

import unittest

from xtts_spanish_app.text_processing import normalize_text, split_text_for_tts


class TextProcessingTests(unittest.TestCase):
    def test_normalize_text_collapses_spaces_and_blank_lines(self) -> None:
        raw = "  Hola,   mundo. \n\n Esta   es   una prueba.  "
        self.assertEqual(normalize_text(raw), "Hola, mundo.\nEsta es una prueba.")

    def test_split_text_for_tts_splits_by_sentence(self) -> None:
        text = "Hola. Esto es una prueba. Otra frase mas."
        self.assertEqual(
            split_text_for_tts(text, max_fragment_chars=24),
            ["Hola.", "Esto es una prueba.", "Otra frase mas."],
        )

    def test_split_text_for_tts_wraps_very_long_text(self) -> None:
        text = "uno dos tres cuatro cinco seis siete ocho nueve diez"
        fragments = split_text_for_tts(text, max_fragment_chars=12)
        self.assertTrue(all(len(fragment) <= 12 for fragment in fragments))
        self.assertGreater(len(fragments), 1)


if __name__ == "__main__":
    unittest.main()
