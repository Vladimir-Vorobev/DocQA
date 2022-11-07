import numpy as np


def cosine_similarity(a: list, b: list):
    """
    Calculates cosine similarity between 2 vectors
    :param a: vector a
    :param b: vector b
    :return: similarity value from -1 to 1
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
