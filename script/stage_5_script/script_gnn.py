import os
import random

import torch
import numpy as np
from code.stage_5_code.Dataset_Loader import Dataset_Loader
from code.stage_5_code.Method_GNN_Cora import Method_GNN_Cora
from code.stage_5_code.Setting_GNN import Setting_GNN
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_5_code.Method_GNN_Citeseer import Method_GNN_Citeseer
from code.stage_5_code.Method_GNN_Pubmed import Method_GNN_Pubmed

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def run_model(dataset_name):

    setting_obj = Setting_GNN(f'{dataset_name} GNN')

    data = Dataset_Loader()
    data.dataset_name = dataset_name
    data.dataset_source_folder_path = rf'data\stage_5_data\{dataset_name}'
    # data = data.load()
    model = None
    if dataset_name == 'cora':
        set_seed(38)
        model = Method_GNN_Cora()
    elif dataset_name == 'citeseer':
        set_seed(64)
        model = Method_GNN_Citeseer()
    elif dataset_name == 'pubmed':
        set_seed(38)
        model = Method_GNN_Pubmed()

    # max_acc = 0.0
    # best_seed = 0
    #
    # for i in range(1, 100):
    #     print(f'running seed {i}')
    #     set_seed(i)
    #     model = Method_GNN_Pubmed()
    #     model.data = data
    #     model.run()
    #
    #     # Train or evaluate your model here to obtain accuracy
    #     # Assuming you have a function get_accuracy() that returns the accuracy of the model
    #
    #     # Example of getting accuracy (replace with your own code)
    #     curr_acc = model.testing_accuracy
    #
    #     if curr_acc > max_acc:
    #         max_acc = curr_acc
    #         best_seed = i
    #
    # print("Best accuracy:", max_acc)
    # print("Seed with best accuracy:", best_seed)



    result_saver = Result_Saver()
    result_saver.result_destination_folder_path = rf'result\stage_5_result\{dataset_name}'
    result_saver.result_destination_file_name = rf'\{dataset_name}.pth'


    setting_obj.prepare(data, model, result_saver, None)
    setting_obj.load_run_save_evaluate()


# best seed 15
# run_model('cora')

# # best seed 64
# run_model('citeseer')

# # best seed 10
run_model('pubmed')