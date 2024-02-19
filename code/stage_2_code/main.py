import Dataset_Loader as loader
import Method_MLP as mlp
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Create Dataset loaders for testing and training set

train_loader = loader.Dataset_Loader()
test_loader = loader.Dataset_Loader()

# Training paths
train_data_folder_path = r'data\stage_2_data'
train_data_file_name = r'\train.csv'

# Testing paths
test_data_folder_path = r'data\stage_2_data'
test_data_file_name = r'\test.csv'

# Init training and tesing loaders
train_loader.dataset_source_folder_path = train_data_folder_path
train_loader.dataset_source_file_name = train_data_file_name

test_loader.dataset_source_folder_path = test_data_folder_path
test_loader.dataset_source_file_name = test_data_file_name

# Load training and testing data
train_data_map = train_loader.load()
test_data_map = test_loader.load()


# Create map to pass into models
data = {'train': train_data_map, 'test': test_data_map}

# Access GPUs
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Init MLP Model
mlp_model = mlp.Method_MLP('base method', 'mlp model with no changes')
mlp_model.data = data
mlp_model.device = device


# Train model
result = mlp_model.run()

# Extract predicted and true labels from the results
pred_y = result['pred_y']
true_y = result['true_y']

# Compute accuracy
accuracy = accuracy_score(true_y, pred_y)

# Compute precision
precision = precision_score(true_y, pred_y, average='macro')

# Compute recall
recall = recall_score(true_y, pred_y, average='macro')

# Compute F1 score
f1 = f1_score(true_y, pred_y, average='macro')

# Print the computed metrics
print("Testing Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
