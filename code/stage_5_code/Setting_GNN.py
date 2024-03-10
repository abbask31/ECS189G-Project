from code.stage_5_code.Method_GNN_Cora import Method_GNN_Cora
from code.stage_5_code.Dataset_Loader import Dataset_Loader
from code.base_class.setting import setting

class Setting_GNN(setting):

    evaluate_additional_metrics = None

    def load_run_save_evaluate(self):

        # load datasets
        data = self.dataset.load()

        self.method.data = data

        # learned_result = self.method.run()
        self.method.run()

        # save raw ResultModule
        self.result.data = self.method.state_dict()
        self.result.save()

        # self.evaluate.data = learned_result
        # self.evaluate_additional_metrics.data = learned_result
        # return self.evaluate.evaluate(), self.evaluate_additional_metrics.evaluate()
