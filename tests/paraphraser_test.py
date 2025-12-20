"""
Tests for the paraphraser class.
"""

import shutil
import unittest
from pathlib import Path

import torch
from transformers import PegasusTokenizer

from src.paraphraser import MODEL_PATH, ParaphrasingTransformer


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
        torch.manual_seed(0)
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
        Input check for the paraphrasing method.
        """
        bad_sentences_lists = [0, 3.14, True, {}, (), "string"]
        bad_numbers_of_paraphrases = [0, 3.14, False, {}, (), [], "string"]
        for bad_list in bad_sentences_lists:
            self.assertIsNone(self.model.paraphrase_sentences(bad_list, 1))

        for bad_number in bad_numbers_of_paraphrases:
            self.assertIsNone(self.model.paraphrase_sentences(["sentence"], bad_number))

    def test_paraphrase_sentences_return_type(self) -> None:
        """
        Output check for the paraphrasing method.
        """
        result = self.model.paraphrase_sentences(["sentence1"], 1)
        self.assertIsInstance(result, list)
        for potential_sentence in result:
            self.assertIsInstance(potential_sentence, str)

    def test_paraphrase_sentences_number_of_paraphrases(self) -> None:
        """
        Output check for number of paraphrases returned by the paraphrasing method.
        """
        self.assertEqual(len(self.model.paraphrase_sentences(["sentence1"], 3)), 3)

    def tearDown(self):
        shutil.rmtree(self.test_path.parent)
