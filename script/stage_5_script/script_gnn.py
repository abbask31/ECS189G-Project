import torch
import numpy as np
from code.stage_5_code.Dataset_Loader import Dataset_Loader
from code.stage_5_code.Method_GNN_Cora import Method_GNN_Cora
from code.stage_5_code.Setting_GNN import Setting_GNN
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_5_code.Method_GNN_Citeseer import Method_GNN_Citeseer
from code.stage_5_code.Method_GNN_Pubmed import Method_GNN_Pubmed

np.random.seed(2)
torch.cuda.manual_seed(2)

def run_model(dataset_name):

    setting_obj = Setting_GNN(f'{dataset_name} GNN')

    data = Dataset_Loader()
    data.dataset_name = dataset_name
    data.dataset_source_folder_path = rf'data\stage_5_data\{dataset_name}'

    model = None
    if dataset_name == 'cora':
        np.random.seed(2)
        torch.cuda.manual_seed(2)
        model = Method_GNN_Cora()
    elif dataset_name == 'citeseer':
        model = Method_GNN_Citeseer()
    elif dataset_name == 'pubmed':
        model = Method_GNN_Pubmed()

    result_saver = Result_Saver()
    result_saver.result_destination_folder_path = rf'result\stage_5_result\{dataset_name}'
    result_saver.result_destination_file_name = rf'\{dataset_name}.pth'

    setting_obj.prepare(data, model, result_saver, None)
    setting_obj.load_run_save_evaluate()

# best seed 2
run_model('cora')

# # best seed 123
# run_model('citeseer')
#
# # best seed 10
# run_model('pubmed')