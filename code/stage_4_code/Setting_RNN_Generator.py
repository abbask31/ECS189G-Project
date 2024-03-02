from code.base_class.setting import setting
class Setting_RNN(setting):

    evaluate_additional_metrics = None

    def load_run_save_evaluate(self):

        self.evaluate_additional_metrics = additinal_evals.Evaluate_Model()

        # load datasets
        data = self.dataset.load()
        print(data)

        self.method.data = data

        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = self.method.state_dict()
        self.result.save()

        self.evaluate.data = learned_result
        self.evaluate_additional_metrics.data = learned_result
        return self.evaluate.evaluate(), self.evaluate_additional_metrics.evaluate()

