import numpy as np


class DecisionNode:
    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 column: int = None,
                 value: float = None,
                 false_branch=None,
                 true_branch=None,
                 is_leaf: bool = False):
        """
        Building block of the decision tree.

        :param data: numpy 2d array data can for example be
         np.array([[1, 2], [2, 6], [1, 7]])
         where [1, 2], [2, 6], and [1, 7] represent each data point
        :param labels: numpy 1d array
         labels indicate which class each point belongs to
        :param column: the index of feature by which data is splitted
        :param value: column's splitting value
        :param true_branch(false_branch): child decision node
        true_branch(false_branch) is DecisionNode instance that contains data
        that satisfies(doesn't satisfy) the filter condition.
        :param is_leaf: is true when node has no child

        """
        self.data = data
        self.labels = labels
        self.column = column
        self.value = value
        self.false_branch = false_branch
        self.true_branch = true_branch
        self.is_leaf = is_leaf


class DecisionTree:

    def __init__(self,
                 max_tree_depth=4,
                 criterion="gini",
                 task="classification"):
        self.tree = None
        self.max_depth = max_tree_depth
        self.task = task

        if criterion == "entropy":
            self.criterion = self._entropy
        elif criterion == "square_loss":
            self.criterion = self._square_loss
        elif criterion == "gini":
            self.criterion = self._gini
        else:
            raise RuntimeError(f"Unknown criterion: '{criterion}'")

    @staticmethod
    def _gini(labels: np.ndarray) -> float:
        """
        Gini criterion for classification tasks.

        """
        probability = []
        for e in set(labels):
            count = 0
            for label in labels:
                if label == e:
                    count += 1
            probability.append(float(count) / float(labels.size))
        return 1.0 - np.sum(np.array(probability)**2)

    @staticmethod
    def _entropy(labels: np.ndarray) -> float:
        """
        Entropy criterion for classification tasks.

        """
        probability = []
        for e in set(labels):
            count = 0
            for label in labels:
                if label == e:
                    count += 1
            probability.append(float(count) / float(labels.size))
        return 0.0 - np.sum(np.array(probability) * np.log2(np.array(probability)))

    @staticmethod
    def _square_loss(labels: np.ndarray) -> float:
        """
        Square loss criterion for regression tasks.

        """
        c = labels.mean()
        return ((labels - c)**2).mean()

    def _iterate(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 current_depth=0) -> DecisionNode:
        """
        This method creates the whole decision tree, by recursively iterating
         through nodes.
        It returns the first node (DecisionNode object) of the decision tree,
         with it's child nodes, and child nodes' children, ect.
        """

        if len(labels) == 1:
            # return a node is_leaf=True
            return DecisionNode(data, labels, is_leaf=True)

        impurity = self.criterion(labels)
        best_column, best_value = None, None

        min_value = 0.0
        data_left_indexes = []
        data_right_indexes = []
        for column, column_values in enumerate(data.T):
            if max(column_values) - min(column_values) == 0:
                continue
            for split_value in np.arange(
                    min(column_values), max(column_values),
                    (max(column_values) - min(column_values)) / 50):
                # find optimal way of splitting the data
                labels_left = [labels[i] for i in range(len(column_values)) if column_values[i] >= split_value]
                labels_right = [labels[i] for i in range(len(column_values)) if column_values[i] < split_value]

                if len(labels_right) == 0 or len(labels_left) == 0:
                    continue
                impurity_left = self.criterion(np.array(labels_left))
                impurity_right = self.criterion(np.array(labels_right))
                value = (float(len(labels_left)) / float(labels.size)) * impurity_left +\
                        (float(len(labels_right)) / float(labels.size)) * impurity_right

                if best_value == None:
                    min_value = value
                    best_value = split_value
                    best_column = column
                    data_left_indexes = [i for i in range(len(column_values)) if column_values[i] >= split_value]
                    data_right_indexes = [i for i in range(len(column_values)) if column_values[i] < split_value]
                elif min_value > value:
                    min_value = value
                    best_value = split_value
                    best_column = column
                    data_left_indexes = [i for i in range(len(column_values)) if column_values[i] >= split_value]
                    data_right_indexes = [i for i in range(len(column_values)) if column_values[i] < split_value]

        if best_column is None or current_depth == self.max_depth:
            return DecisionNode(data, labels, is_leaf=True)
        else:
            data_left = data[data_left_indexes, :]
            data_right = data[data_right_indexes, :]
            # return DecisionNode with true(false)_branch=self._iterate(...)
            return DecisionNode(data, labels, best_column, best_value,
                                self._iterate(data_right, labels[data_right_indexes], current_depth + 1),
                                self._iterate(data_left, labels[data_left_indexes], current_depth + 1))

    def fit(self, data: np.ndarray, labels: np.ndarray):
        self.tree = self._iterate(data, labels)

    def predict(self, point: np.ndarray) -> float or int:
        """
        This method iterates nodes starting with the first node i. e.
        self.tree. Returns predicted label of a given point (example [2.5, 6],
        where 2.5 and 6 are points features).

        """
        node = self.tree

        while True:
            if node.is_leaf:
                if self.task == "classification":
                    # predict and return the label for classification task
                    predicted = node.labels[0]
                    max_predicted_count = 0
                    for e in set(node.labels):
                        count = 0
                        for label in node.labels:
                            if label == e:
                                count += 1
                        if count > max_predicted_count:
                            max_predicted_count = count
                            predicted = e
                    return predicted
                else:
                    # predict and return the label for regression task
                    return node.labels.mean()
            if point[node.column] >= node.value:
                node = node.true_branch
            else:
                node = node.false_branch
