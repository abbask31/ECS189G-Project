'''
Concrete Evaluate class for a specific evaluation metrics
'''
import numpy as np
import torch
from sklearn.metrics import accuracy_score

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy:
    data = None
    def __init__(self, evaluate_name, description):
        self.evaluate_name = evaluate_name
        self.description = description

    def evaluate(self):

        acc = accuracy_score(self.data['true_y'], self.data['pred_y'])

        return acc


