"""
Example logic for the project.
"""

import logging
from pathlib import Path

from transformers import PegasusTokenizer

from src.dataset_manager import load_json, save2json
from src.paraphraser import MODEL_PATH, ParaphrasingTransformer

logger = logging.getLogger("../src/start")

logging.basicConfig(level=logging.INFO)


def main():
    """
    Running the generation.
    """
    model = ParaphrasingTransformer(MODEL_PATH, PegasusTokenizer)
    sentences = [
        "I wish I was a person who I always wanted to be and who could love themselves.",
        "This is the sentence, which could be counted as the second one.",
    ]
    number_of_paraphrases = 5
    result = model.paraphrase_sentences(sentences, number_of_paraphrases)

    logger.info("Paraphrase results.")
    logger.info(result)

    paraphrased_path = Path("src/assets/paraphrased_sentences.json")
    if result:
        save2json(sentences, result, paraphrased_path)

    loaded_paraphrases = load_json(paraphrased_path)
    logger.info("")
    logger.info("Loaded dictionary of paraphrases")
    logger.info(loaded_paraphrases)

    list_paraphrases = load_json(paraphrased_path, True)
    logger.info("")
    logger.info("Loaded list of paraphrases")
    logger.info(list_paraphrases)


if __name__ == "__main__":
    main()
