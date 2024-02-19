'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
import numpy as np
import code.stage_2_code.Dataset_Loader as loader
import code.stage_2_code.Evaluate_Model as additinal_evals

class Setting_MLP(setting):

    evaluate_additional_metrics = None

    test_dataset = None
    test_data_folder_path = r'data\stage_2_data'
    test_data_file_name = r'\test.csv'


    def load_run_save_evaluate(self):

        self.evaluate_additional_metrics = additinal_evals.Evaluate_Model()

        # setup testing set
        self.test_dataset = loader.Dataset_Loader()
        self.test_dataset.dataset_source_folder_path = self.test_data_folder_path
        self.test_dataset.dataset_source_file_name = self.test_data_file_name

        # load datasets
        train_data = self.dataset.load()
        test_data = self.test_dataset.load()

        self.method.data = {'train': train_data, 'test': test_data}

        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        self.evaluate_additional_metrics.data = learned_result

        return self.evaluate.evaluate(), self.evaluate_additional_metrics.evaluate()
