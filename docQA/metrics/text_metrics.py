import numpy as np


def cosine_similarity(a: list, b: list):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
