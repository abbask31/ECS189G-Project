from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_2_code.Setting_MLP import Setting_MLP
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_2_code.Evaluate_Model import Evaluate_Model
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('number data', '')
    data_obj.dataset_source_folder_path = r'data\stage_2_data'
    data_obj.dataset_source_file_name = r'\train.csv'

    method_obj = Method_MLP('multi-layer perceptron', '')
    method_obj.device = device

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = r'result\stage_2_result\MLP_'
    result_obj.result_destination_file_name = r'\prediction_result'

    setting_obj = Setting_MLP('MLP Model Basic Setting with training and testing dataset')
    # setting_obj = Setting_KFold_CV('k fold cross validation', '')
    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    additional_eval_ojb = Evaluate_Model('additional eval metrics','')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    accuracy, additional_metrics = setting_obj.load_run_save_evaluate()
    precision, recall, f1 = additional_metrics
    print('************ Overall Performance ************')
    # print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('MLP Accuracy: ' + str(accuracy))
    print('MLP Precision: ' + str(precision))
    print('MLP Recall: ' + str(recall))
    print('MLP F1-Score: ' + str(f1))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    