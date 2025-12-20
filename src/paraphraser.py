import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import logging
import json
from pathlib import Path


logger = logging.getLogger("../src/paraphrazer")

logging.basicConfig(level=logging.INFO)

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
        self.model = PegasusForConditionalGeneration.from_pretrained(MODEL_PATH).to(TORCH_DEVICE)
        self.tokenizer = PegasusTokenizer.from_pretrained(MODEL_PATH)

    def paraphrase_sentences(
        self, input_sentences: list[str], num_return_sequences: int
    ) -> list[str]:
        """ """
        batch = self.tokenizer(
            input_sentences, truncation=True, padding="longest", max_length=100, return_tensors="pt"
        ).to(TORCH_DEVICE)

        translated = self.model.generate(
            **batch,
            max_length=100,
            num_beams=8,
            num_return_sequences=num_return_sequences,
            temperature=1.5,
        )

        paraphrised_sentences = self.tokenizer.batch_decode(translated, skip_special_tokens=True)

        return paraphrised_sentences

    def save2json(sentences: list[str], file_path: Path, number_of_paraphrases) -> None:
        grouped_sentences = {}
        for index in range(0, len(sentences), number_of_paraphrases):
            grouped_sentences[sentences[index]] = sentences[
                index + 1 : index + number_of_paraphrases
            ]

        with open(file_path, "w") as file:
            json.dump(grouped_sentences, file, indent=4)


def main():
    """
    Running the generation
    """
    model = ParaphrasingTransformer(MODEL_PATH, PegasusTokenizer)
    sentences = ["This is an example sentence", "Each sentence is converted"]
    number_of_paraphrases = 5
    result = model.paraphrase_sentences(sentences, number_of_paraphrases)

    logger.info(result)

    model.save2json(result, Path("src/assets/paraphrased_sentences.json"), number_of_paraphrases)


if __name__ == "__main__":
    main()
