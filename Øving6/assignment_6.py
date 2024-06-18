import numpy as np
from pathlib import Path
from typing import Tuple

POSSIBLE_VALUES = [1, 2]

class Node:
    """ Node class used to build the decision tree"""
    def __init__(self):
        self.children = {}
        self.parent = None
        self.attribute = None
        self.value = None

    def classify(self, example):
        if self.value is not None:
            return self.value
        return self.children[example[self.attribute]].classify(example)



def plurality_value(examples: np.ndarray) -> int:
    """Implements the PLURALITY-VALUE (Figure 19.5)"""
    labels = examples[:, -1]
    value, count = 0, 0
    for label in np.unique(labels):
        label_count = np.count_nonzero(labels == label)
        if label_count > count:
            value = label
            count = label_count

    return value

def b(q: float) -> float:
    if q in [0, 1]:
        return 0
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    
def importance(attributes: np.ndarray, examples: np.ndarray, measure: str) -> int:
    """
    This function should compute the importance of each attribute and choose the one with highest importance,
    A ← argmax a ∈ attributes IMPORTANCE (a, examples) (Figure 19.5)

    Parameters:
        attributes (np.ndarray): The set of attributes from which the attribute with highest importance is to be chosen
        examples (np.ndarray): The set of examples to calculate attribute importance from
        measure (str): Measure is either "random" for calculating random importance, or "information_gain" for
                        caulculating importance from information gain (see Section 19.3.3. on page 679 in the book)

    Returns:
        (int): The index of the attribute chosen as the test

    """
    if measure == "random":
        return np.random.choice(attributes)
    p = sum(1 for e in examples if e[-1] == 1)
    goal =  b(p / len(examples))
    
    importances = []
    for A in attributes:
        values = [[e for e in examples if e[A] == v] for v in POSSIBLE_VALUES]
        remainder = sum(len(v) / len(examples) * b(sum(1 for e in v if e[-1] == 1) / len(v)) for v in values)
        importances.append(goal - remainder)
    return attributes[importances.index(max(importances))]

def uniform_classification(examples: np.ndarray) -> bool:
        """Check if all examples have the same classification."""
        return len(np.unique(examples[:, -1])) == 1

def learn_decision_tree(examples: np.ndarray, attributes: np.ndarray, parent_examples: np.ndarray,
                        parent: Node, branch_value: int, measure: str):
    """
    This is the decision tree learning algorithm. The pseudocode for the algorithm can be
    found in Figure 19.5 on Page 678 in the book.

    Parameters:
        examples (np.ndarray): The set data examples to consider at the current node
        attributes (np.ndarray): The set of attributes that can be chosen as the test at the current node
        parent_examples (np.ndarray): The set of examples that were used in constructing the current node’s parent.
                                        If at the root of the tree, parent_examples = None
        parent (Node): The parent node of the current node. If at the root of the tree, parent = None
        branch_value (int): The attribute value corresponding to reaching the current node from its parent.
                        If at the root of the tree, branch_value = None
        measure (str): The measure to use for the Importance-function. measure is either "random" or "information_gain"

    Returns:
        (Node): The subtree with the current node as its root
    """

    # Creates a node and links the node to its parent if the parent exists
    node = Node()
    if parent is not None:
        parent.children[branch_value] = node
        node.parent = parent

    if len(examples) == 0:
        node.value = plurality_value(parent_examples)
    elif uniform_classification(examples):
        node.value = examples[0][-1]
    elif len(attributes) == 0:
        node.value = plurality_value(examples)
    else:
        A = importance(attributes=attributes, examples=examples, measure=measure)
        node.attribute = A
        
        for v in POSSIBLE_VALUES:
            exs = examples[examples[:, A] == v]
            subtree = learn_decision_tree(
                examples=exs,
                attributes=[a for a in attributes if a != A],
                parent_examples=examples,
                parent=node,
                branch_value=v,
                measure=measure
            )
            node.children[v] = subtree
    return node



def accuracy(tree: Node, examples: np.ndarray) -> float:
    """ Calculates accuracy of tree on examples """
    correct = 0
    for example in examples:
        pred = tree.classify(example[:-1])
        correct += pred == example[-1]
    return correct / examples.shape[0]


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """ Load the data for the assignment,
    Assumes that the data files is in the same folder as the script"""
    with (Path.cwd() / "train.csv").open("r") as f:
        train = np.genfromtxt(f, delimiter=",", dtype=int)
    with (Path.cwd() / "test.csv").open("r") as f:
        test = np.genfromtxt(f, delimiter=",", dtype=int)
    return train, test




if __name__ == '__main__':

    train, test = load_data()

    # information_gain or random
    measure = "random"

    tree = learn_decision_tree(examples=train,
                    attributes=np.arange(0, train.shape[1] - 1, 1, dtype=int),
                    parent_examples=None,
                    parent=None,
                    branch_value=None,
                    measure=measure)

    print(f"Training Accuracy {accuracy(tree, train)}")
    print(f"Test Accuracy {accuracy(tree, test)}")