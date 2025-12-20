"""
Dataset Manager module.
"""
import json
from pathlib import Path


def save2json(sentences: list[str], paraphrased_sentences: list[str], file_path: Path) -> None:
    """
    Function for saving paraphrases to a JSON file

    Args:
        sentences (list[str]): initial sentences
        paraphrased_sentences (list[str]): Paraphrased sentences.
        file_path (Path): Path to save.
    """
    grouped_sentences = {}
    number_of_paraphrases = len(paraphrased_sentences) // len(sentences)

    for index in range(0, len(paraphrased_sentences), number_of_paraphrases):
        grouped_sentences[paraphrased_sentences[index]] = paraphrased_sentences[
            index + 1 : index + number_of_paraphrases
        ]

    # here, it is important to point out that first paraphrased sentence is
    # the same as the one that was given the model to modify.

    with open(file_path, "w") as file:
        json.dump(grouped_sentences, file, indent=4)


def load_json(file_path: Path, list_view: bool = False) -> dict | list:
    """
    Function for loading paraphrases from a JSON file

    Args:
        file_path (Path): Path to load from.
        list_view (bool): Use a list as the type of the output
        instead of a dictionary, which might be useful for
        further usage.

    Returns:
        dict | list: Output
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        paraphrases = json.load(file)

    if not list_view and isinstance(paraphrases, dict):
        return paraphrases

    paraphrases_list = []
    for key_sent, value_sents in paraphrases.items():
        paraphrases_list.append(key_sent)
        paraphrases_list.extend(value_sents)
    return paraphrases_list
