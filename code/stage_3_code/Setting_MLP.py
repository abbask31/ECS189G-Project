'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
import numpy as np
import code.stage_3_code.Evaluate_Model as additinal_evals

class Setting_MLP(setting):

    evaluate_additional_metrics = None

    def load_run_save_evaluate(self):

        self.evaluate_additional_metrics = additinal_evals.Evaluate_Model()

        # load datasets
        train_data_map, test_data_map = self.dataset.load()

        self.method.data = {'train': train_data_map, 'test': test_data_map}

        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        self.evaluate_additional_metrics.data = learned_result

        return self.evaluate.evaluate(), self.evaluate_additional_metrics.evaluate()
