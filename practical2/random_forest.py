import numpy as np
from .decision_tree import DecisionTree

class RandomForestClassifier(object):
    def __init__(self, rand_percent=70, tree_count=5, depth=4):
        """
        you can add as many parameters as you want to your classifier
        """
        self.trees = []
        for i in range(tree_count):
            self.trees.append(DecisionTree(depth))
        self.rand_percent = rand_percent


    def fit(self, data: np.ndarray, labels: np.ndarray):
        """
        :param data: array of features for each point
        :param labels: array of labels for each point
        """
        for tree in self.trees:
            indexes = np.array(range(len(labels)))
            np.random.shuffle(indexes)
            labels_count = (len(labels) * self.rand_percent) // 100
            tree.fit(data[indexes[:labels_count], :], labels[indexes[:labels_count]])


    def predict(self, data: np.ndarray) -> np.ndarray:
        result = []
        for point in data:
            predicted = []
            for tree in self.trees:
                predicted.append(tree.predict(point))
            true_count = 0
            false_count = 0
            for i in predicted:
                if i == 0:
                    false_count += 1
                else:
                    true_count += 1
            if true_count >= false_count:
                result.append(1)
            else:
                result.append(0)
        return result



def f1_score(y_true: np.ndarray, y_predicted: np.ndarray):
    """
    only 0 and 1 should be accepted labels and 1 is the positive class
    """
    assert set(y_true).union({1, 0}) == {1, 0}
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(y_true)):
        if y_predicted[i] == 1 and y_true[i] == 1:
            true_positive += 1
        if y_predicted[i] == 1 and y_true[i] == 0:
            false_positive += 1
        if y_predicted[i] == 0 and y_true[i] == 1:
            false_negative += 1
    if true_positive == 0:
        recall = 0
        precision = 0
    else:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
    if recall + precision == 0:
        return 0
    return 2 * (recall * precision) / (recall + precision)



def data_preprocess(data: np.array) -> np.array:
    keys = {
        0: 'age',
        1: 'foundation_type',
        2: 'roof_type',
        3: 'position',
        4: 'loc_id',
        5: 'num_floors',
        6: 'area',
        7: 'height',
        8: 'num_families',
        9: 'ownership_type',
        10: 'configuration',
        11: 'surface_condition'
    }
    foundation_type = {list(set(data.T[1]))[i]: i for i in range(len(list(set(data.T[1]))))}
    surface_condition = {list(set(data.T[11]))[i]: i for i in range(len(list(set(data.T[11]))))}
    preprocessed_data = []
    for row in data:
        vector = []
        for i in range(len(row)):
            if keys[i] == 'age':
                vector.append(row[i])
            elif keys[i] == 'foundation_type':
                vector.append(foundation_type[row[i]])
            elif keys[i] == 'num_floors':
                vector.append(row[i])
            elif keys[i] == 'height':
                vector.append(row[i])
            elif keys[i] == 'area':
                vector.append(row[i])
            elif keys[i] == 'surface_condition':
                vector.append(surface_condition[row[i]])
        preprocessed_data.append(np.array(vector))
    return np.array(preprocessed_data)
