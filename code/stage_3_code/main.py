import Dataset_Loader as loader
import Method_MLP as mlp
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Create Dataset loaders for testing and training set

train_loader = loader.Dataset_Loader()

# Data pickle path
train_data_folder_path = r'data\stage_3_data'
train_data_file_name = r'\MNIST'


# Init training and tesing loaders
train_loader.dataset_source_folder_path = train_data_folder_path
train_loader.dataset_source_file_name = train_data_file_name

# Load training and testing data
train_data_map, test_data_map = train_loader.load()


# Create map to pass into models
data = {'train': train_data_map, 'test': test_data_map}

# print(len(data['test']['y']))
