from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Method_RNN_Classifier import Method_RNN_Classifier
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_RNN import Setting_RNN
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_4_code.Evaluate_Model import Evaluate_Model
import numpy as np
import torch


def init_model_objs(dataset_name):
    data_obj = Dataset_Loader(dataset_name, '')
    data_obj.dataset_source_folder_path = r'data\stage_4_data\text_classification'
    data_obj.task = 'classification'
    data_obj.train_classifer_path = r'data\stage_4_data\loaded_data\train_loader.pkl'
    data_obj.test_classifier_path = r'data\stage_4_data\loaded_data\test_loader.pkl'

    method_obj = Method_RNN_Classifier()

    result_obj = Result_Saver(dataset_name + ' saver', '')
    result_obj.result_destination_folder_path = rf'result\stage_4_result\text_classification'
    result_obj.result_destination_file_name = rf'\text_classification_model.pth'

    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    setting_obj = Setting_RNN(dataset_name)

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)

    return setting_obj


def run_model(settings_obj):
    print('************ Start ************')
    settings_obj.print_setup_summary()
    # settings_obj.load_run_save_evaluate()
    accuracy, additional_metrics = settings_obj.load_run_save_evaluate()
    precision, recall, f1 = additional_metrics
    print('************ Overall Performance ************')
    print('CNN Accuracy: ' + str(accuracy))
    print('CNN Precision: ' + str(precision))
    print('CNN Recall: ' + str(recall))
    print('CNN F1-Score: ' + str(f1))
    print('************ Finish ************')


# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------


    # ---- objection initialization setction ---------------

    setting_rnn = init_model_objs('Classification RNN')

    # ------------------------------------------------------

    # ---- running section ---------------------------------
    run_model(setting_rnn)
    # ------------------------------------------------------
