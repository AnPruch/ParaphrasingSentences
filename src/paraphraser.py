"""
Paraphraser module.
"""

import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

MODEL_PATH = "tuner007/pegasus_paraphrase"

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ParaphrasingTransformer:
    """
    Transformer for paraphrasing class
    """

    def __init__(self, model_path: str, tokenizer: PegasusTokenizer) -> None:
        """
        Docstring for __init__

        Args:
            model_path (str): Path for downloading the model.
            tokenizer (PegasusTokenizer): Tokenizer for the paraphrasing model.
        """
        self.model = PegasusForConditionalGeneration.from_pretrained(model_path).to(TORCH_DEVICE)
        self.tokenizer = tokenizer.from_pretrained(model_path)

    def paraphrase_sentences(
        self, input_sentences: list[str], num_paraphrases: int
    ) -> list[str] | None:
        """
        Paraphrase the sentences.

        Args:
            input_sentences (list[str]): Sentences to paraphrase.
            num_paraphrases (int): Number of paraphrases to create total
            (including the initial phrases).

        Returns:
            list[str] | None: Paraphrased sentences. None in case of corrupt arguments.
        """
        if not (
            isinstance(input_sentences, list)
            and all(isinstance(sent, str) for sent in input_sentences)
            and isinstance(num_paraphrases, int)
            and num_paraphrases > 0
        ):
            return None

        batch = self.tokenizer(
            input_sentences, truncation=True, padding="longest", max_length=100, return_tensors="pt"
        ).to(TORCH_DEVICE)

        translated = self.model.generate(
            **batch,
            max_length=100,
            num_beams=8,
            num_return_sequences=num_paraphrases,
            temperature=1.5,
        )

        paraphrased_sentences = self.tokenizer.batch_decode(translated, skip_special_tokens=True)

        if paraphrased_sentences:
            return paraphrased_sentences
        return None
