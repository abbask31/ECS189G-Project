from code.base_class import setting

class Setting_RNN(setting):

    evaluate_additional_metrics = None

    def load_run_save_evaluate(self):

        # self.evaluate_additional_metrics = additinal_evals.Evaluate_Model()

        # load datasets
        data = self.dataset.load()

        self.method.data = data

        self.method.run()

        # save raw ResultModule
        # self.result.data = learned_result
        # self.result.save()

        # self.evaluate.data = learned_result
        # self.evaluate_additional_metrics.data = learned_result

        # return self.evaluate.evaluate(learned_result['true_y'], learned_result['pred_y']), self.evaluate_additional_metrics.evaluate()
