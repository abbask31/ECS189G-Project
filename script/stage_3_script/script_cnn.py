from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_CNN import Setting_CNN
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch


def init_model_objs(dataset_name):
    data_obj = Dataset_Loader(dataset_name, '')
    data_obj.dataset_source_folder_path = r'data\stage_3_data'
    data_obj.dataset_source_file_name = rf'\{dataset_name}'

    method_obj = None

    if dataset_name == 'MNIST':
        method_obj = Method_CNN_MNIST('MNIST CNN', '')
    elif dataset_name == 'ORL':
        method_obj = Method_CNN_ORL('ORL CNN', '')
    elif dataset_name == 'CIFAR':
        method_obj = Method_CNN_CIFAR('CIFAR CNN', '')


    result_obj = Result_Saver(dataset_name + ' saver', '')
    result_obj.result_destination_folder_path = rf'result\stage_3_result\{dataset_name}'
    result_obj.result_destination_file_name = rf'\{dataset_name}_results'

    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    setting_obj = Setting_CNN('CNN Model for ' + dataset_name)

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)

    return setting_obj


def run_model(settings_obj):
    print('************ Start ************')
    settings_obj.print_setup_summary()
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

    # setting_orl = init_model_objs('ORL')
    # setting_mnist = init_model_objs('MNIST')
    setting_cifar = init_model_objs('CIFAR')

    # ------------------------------------------------------

    # ---- running section ---------------------------------
    # run_model(setting_orl)
    # run_model(setting_mnist)
    run_model(setting_cifar)
    # ------------------------------------------------------
