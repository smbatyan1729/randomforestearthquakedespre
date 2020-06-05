import unittest
import time

import numpy as np
import pandas as pd

from practical2.random_forest import RandomForestClassifier
from practical2.random_forest import f1_score, data_preprocess


def cross_val(X: np.ndarray, Y: np.ndarray, *, folds: int = 5):
    """
    Given initial data X & Y,
    divide the data, and return `folds` number of tuples of the form
    [(x_train, y_train), (x_test, y_test)]
    """
    c = np.c_[X.reshape(len(X), -1), Y.reshape(len(Y), -1)]
    np.random.shuffle(c)
    X1 = c[:, :X.size // len(X)].reshape(X.shape)
    Y1 = c[:, -1].reshape(Y.shape)
    N = len(X)
    step = N / folds
    for i in range(folds):
        train_data_x = [X1[j] for j in range(len(X1)) if j < (i * step) or j >= (i + 1) * step]
        train_data_y = [Y1[j] for j in range(len(Y1)) if j < (i * step) or j >= (i + 1) * step]
        test_data = (X1[int(i * step): int((i + 1) * step)], Y1[int(i * step): int((i + 1) * step)])
        yield (train_data_x, train_data_y), test_data


class TestRandomForestClassifier(unittest.TestCase):
    # def test_end_to_end(self):
    #     train_data = pd.read_csv("data/train.csv")
    #     labels = train_data['label'].values
    #     x = np.array(train_data.drop('label', axis=1))
    #     y = labels
    #     max_f1, best_rand_percent, best_tree_count, best_depth = None, None, None, None
    #     for rand_percent in range(20, 80, 10):
    #         for tree_count in range(1, 10):
    #             for depth in range(4, 10):
    #                 result = 0
    #                 for [(x, y), (tx, ty)] in cross_val(x, y, folds=5):
    #                     print(rand_percent, tree_count, depth, max_f1)
    #                     model = RandomForestClassifier(rand_percent, tree_count, depth)
    #                     model.fit(data_preprocess(np.array(x)), np.array(y))
    #                     y_predict = model.predict(data_preprocess(np.array(tx)))
    #                     result += f1_score(np.array(ty), np.array(y_predict))
    #                 if max_f1 is None or result / 5 > max_f1:
    #                     max_f1 = result / 5
    #                     best_rand_percent = rand_percent
    #                     best_tree_count = tree_count
    #                     best_depth = depth
    #     print(max_f1, best_rand_percent, best_tree_count, best_depth)

    def test_end_to_end(self):
        start_time = time.time()
        model = RandomForestClassifier()

        train_data = pd.read_csv("data/train.csv")
        train_data = train_data[:300]

        labels = train_data['label'].values
        x = np.array(train_data.drop('label', axis=1))
        y = labels

        model.fit(data_preprocess(x), y)
        y_predict = model.predict(data_preprocess(x))
        self.assertGreater(f1_score(y, y_predict), 0)


    def test_f1_Score(self):
        self.assertEqual(f1_score([1, 1, 1], [1, 1, 1]), 1)
        self.assertEqual(f1_score([1, 1, 1], [0, 0, 0]), 0)
        self.assertEqual(f1_score([1, 1, 1], [0, 1, 0]), 0.5)
