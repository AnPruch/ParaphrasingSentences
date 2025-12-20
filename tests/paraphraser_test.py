"""
Test for the paraphraser class
"""

import json
import unittest
from pathlib import Path
import shutil
import logging

import pytest

from src.paraphraser import ParaphrasingTransformer, MODEL_PATH

from transformers import PegasusTokenizer


logger = logging.getLogger("../src/paraphrazer")

logging.basicConfig(level=logging.INFO)


class ParaphrasingTransformerTest(unittest.TestCase):
    """
    Tests model saving functionality.
    """

    def setUp(self) -> None:
        """
        Setup for saving model
        """
        self.model = ParaphrasingTransformer(MODEL_PATH, PegasusTokenizer)

        self.test_path = Path(__file__).parent.parent / "test_tmp" / "paraphrased_sentences.json"
        self.test_path.parent.mkdir(parents=True, exist_ok=True)

    def test_paraphrase_sentences_ideal(self) -> None:
        """
        Ideal scenario for the paraphrasing method.
        """
        sentences = ["This is an example sentence", "Each sentence is converted"]
        number_of_paraphrases = 5
        result = self.model.paraphrase_sentences(sentences, number_of_paraphrases)

        expected = [
            "This is an example sentence.",
            "This is an example sentence",
            "This sentence is an example.",
            "This is an example of a sentence.",
            "An example sentence is this one.",
            "The sentence is converted.",
            "The sentence is converted into something else.",
            "Each sentence is converted into something else.",
            "Each sentence is converted.",
            "The sentence is converted",
        ]

        self.assertListEqual(result, expected)

        def test_paraphrase_sentences_invalid_input(self) -> None:
            """
            Output check for the paraphrasing method.
            """
            bad_inputs = [0, True, {}, (), "string"]
            for bad_input in bad_inputs:
                self.assertIsNone(self.model.paraphrase_sentences(bad_input))

    def tearDown(self):
        shutil.rmtree(self.test_path)
