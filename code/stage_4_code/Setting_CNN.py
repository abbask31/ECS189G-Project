'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
import numpy as np
import code.stage_3_code.Evaluate_Model as additinal_evals

class Setting_CNN(setting):

    evaluate_additional_metrics = None

    def load_run_save_evaluate(self):

        self.evaluate_additional_metrics = additinal_evals.Evaluate_Model()

        # load datasets
        train_data_map, test_data_map = self.dataset.load()

        # print(len(train_data_map['X']), len(train_data_map['X'][0]), len(train_data_map['X'][0])[0])

        self.method.data = {'train': train_data_map, 'test': test_data_map}

        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        self.evaluate_additional_metrics.data = learned_result

        return self.evaluate.evaluate(learned_result['true_y'], learned_result['pred_y']), self.evaluate_additional_metrics.evaluate()
