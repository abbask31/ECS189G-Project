'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score


class Evaluate_Model(evaluate):
    data = None

    def evaluate(self):
        print('evaluating additional performance metrics...')

        # Compute precision
        precision = precision_score(self.data['true_y'], self.data['pred_y'], average='macro')

        # Compute recall
        recall = recall_score(self.data['true_y'], self.data['pred_y'], average='macro')

        # Compute F1 score
        f1 = f1_score(self.data['true_y'], self.data['pred_y'], average='macro')


        return [precision, recall, f1]