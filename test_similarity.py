import numpy as np
from src.similarity import jaccard_similarity, lp_distance, pearson_similarity

def test_jaccard_similarity():
    user1 = np.array([1, 0, 1, 0])
    user2 = np.array([1, 1, 0, 0])
    assert jaccard_similarity(user1, user2) == 0.5

def test_lp_distance():
    user1 = np.array([1, 0, 1, 0])
    user2 = np.array([1, 1, 0, 0])
    assert lp_distance(user1, user2) == np.sqrt(2)

def test_pearson_similarity():
    user1 = np.array([1, 0, 1, 0])
    user2 = np.array([1, 1, 0, 0])
    assert pearson_similarity(user1, user2) == -0.5
