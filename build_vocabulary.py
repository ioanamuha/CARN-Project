import json
import re

import numpy as np
import torch


def build_vocabulary(filepath):
    unique_words = set()

    with open(filepath, 'r') as f:
        data = json.load(f)

    for entry in data.values():
        full_text = f"{entry['precontext']} {entry['sentence']} {entry['ending']} {entry['judged_meaning']}"

        tokens = re.findall(r"\w+|[^\w\s]", full_text.lower(), re.UNICODE)

        unique_words.update(tokens)

    return unique_words


def create_dictionaries(unique_words):
    word_to_ix = {"<PAD>": 0, "<UNK>": 1, "<H>": 2, "</H>": 3}
    ix_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<H>", 3: "</H>"}

    for word in unique_words:
        if word not in word_to_ix:
            index = len(word_to_ix)
            word_to_ix[word] = index
            ix_to_word[index] = word

    return word_to_ix, ix_to_word


def vectorize(text, word_to_ix):
    """
    Converts a string of text into a list of integers.
    """
    tokens = text.lower().split()
    indices = []

    for token in tokens:
        if token in word_to_ix:
            indices.append(word_to_ix[token])
        else:
            indices.append(word_to_ix["<UNK>"])

    return indices


def load_glove_embeddings(path, word_to_ix, embed_dim=100):
    """
    Reads GloVe file and creates an embedding matrix for the specific vocabulary.
    """
    print(f"Loading GloVe embeddings from {path}...")
    embeddings_index = {}

    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]

                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    except FileNotFoundError:
        print(f"Error: GloVe file not found at {path}. Please download glove.6B.100d.txt")
        return None

    vocab_size = len(word_to_ix)

    embedding_matrix = torch.randn((vocab_size, embed_dim))
    embedding_matrix[0] = torch.zeros(embed_dim)

    hits = 0
    misses = 0

    for word, i in word_to_ix.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = torch.from_numpy(embedding_vector)
            hits += 1
        else:
            misses += 1

    print(f"GloVe loaded. Hits: {hits} (found), Misses: {misses} (unknown/special tokens)")

    embedding_matrix[0] = torch.zeros(embed_dim)

    return embedding_matrix
