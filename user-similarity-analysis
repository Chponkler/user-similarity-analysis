import numpy as np
from scipy.spatial.distance import jaccard
from scipy.stats import pearsonr
from scipy.spatial import distance
import argparse

# Матрица взаимодействия: строки - пользователи, столбцы - видео
# 1 - просмотрено, 0 - не просмотрено
user_video_matrix = np.array([
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [0, 1, 1, 1],
    [1, 0, 1, 1]
])

def jaccard_similarity(user1, user2):
    """
    Вычисляет коэффициент сходства Жаккара между двумя пользователями.
    
    :param user1: Вектор просмотров первого пользователя
    :param user2: Вектор просмотров второго пользователя
    :return: Коэффициент сходства Жаккара
    """
    return 1 - jaccard(user1, user2)

def lp_distance(user1, user2, p=2):
    """
    Вычисляет Lp-расстояние между двумя пользователями.
    
    :param user1: Вектор просмотров первого пользователя
    :param user2: Вектор просмотров второго пользователя
    :param p: Порядок нормы (по умолчанию 2 для евклидова расстояния)
    :return: Lp-расстояние
    """
    return np.linalg.norm(user1 - user2, ord=p)

def pearson_similarity(user1, user2):
    """
    Вычисляет корреляцию Пирсона между двумя пользователями.
    
    :param user1: Вектор просмотров первого пользователя
    :param user2: Вектор просмотров второго пользователя
    :return: Корреляция Пирсона
    """
    return pearsonr(user1, user2)[0]

def main(user1_idx, user2_idx):
    """
    Основная функция для вычисления и вывода всех метрик сходства между двумя пользователями.
    
    :param user1_idx: Индекс первого пользователя в матрице
    :param user2_idx: Индекс второго пользователя в матрице
    """
    user1 = user_video_matrix[user1_idx]
    user2 = user_video_matrix[user2_idx]
    
    jaccard_score = jaccard_similarity(user1, user2)
    lp_score = lp_distance(user1, user2)
    pearson_score = pearson_similarity(user1, user2)
    
    print(f"Жаккаровское сходство между пользователем {user1_idx} и {user2_idx}: {jaccard_score}")
    print(f"L2-расстояние между пользователем {user1_idx} и {user2_idx}: {lp_score}")
    print(f"Корреляция Пирсона между пользователем {user1_idx} и {user2_idx}: {pearson_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Вычисление сходства между пользователями на основе их просмотров.")
    parser.add_argument("user1_idx", type=int, help="Индекс первого пользователя в матрице")
    parser.add_argument("user2_idx", type=int, help="Индекс второго пользователя в матрице")
    args = parser.parse_args()
    
    main(args.user1_idx, args.user2_idx)
