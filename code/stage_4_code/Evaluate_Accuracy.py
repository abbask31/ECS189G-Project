'''
Concrete Evaluate class for a specific evaluation metrics
'''
import numpy as np
import torch

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy:
    def __init__(self, evaluate_name, description):
        self.evaluate_name = evaluate_name
        self.description = description

    def evaluate(self, true_y, pred_y):
        # Convert to NumPy arrays if tensors
        if torch.is_tensor(true_y):
            true_y = true_y.cpu().numpy()
        if torch.is_tensor(pred_y):
            pred_y = pred_y.cpu().numpy()

        # Calculate accuracy
        accuracy = np.mean(true_y == pred_y)
        return accuracy


