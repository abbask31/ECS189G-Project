import torch
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
from code.stage_4_code.Method_RNN_Classifier import RNNClassifier

from torch.nn.utils.rnn import pad_sequence
input_size = 100  # Size of GloVe embeddings (100)
hidden_size = 128  # Size of hidden state in RNN
output_size = 2  # Number of classes (neg and pos
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = r'result\stage_4_result\text_classification\model.pkl'
model = RNNClassifier(input_size, hidden_size, output_size)

# Load the state dictionary into your model
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model = model.to(device)
# print(model)

# Define the tokenizer and load the pre-trained GloVe embeddings
tokenizer = get_tokenizer('basic_english')
glove_file_path = r'data\stage_4_data\embedding'

# Load GloVe embeddings
glove = GloVe(name='6B', dim=100, cache=glove_file_path)

# Define your input text
input_text = "Hello i am very happy."

# Tokenize the input text
tokens = tokenizer(input_text.lower())

# Convert tokens into indices based on the vocabulary
oov_value = torch.tensor([0] * 100, dtype=torch.long)
indexed_tokens = [glove[token] if token in glove.stoi else oov_value for token in tokens]

# Pad the sequence of indices
max_length = 100  # You need to use the same max_length as during training
pad_len = max_length - len(indexed_tokens)
if pad_len > 0:
    padded_tensors = [torch.tensor([0] * 100, dtype=torch.long)] * pad_len
    indexed_tokens += padded_tensors

# Convert the sequence of indices into a PyTorch tensor
input_tensor = torch.stack(indexed_tokens)

# Ensure the input tensor is on the correct device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = input_tensor.to(device)

# Pass the input tensor through your model for inference
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation for inference
    output = model(input_tensor.unsqueeze(0))  # Add batch dimension (batch_size=1) for inference

# Process the model output
probabilities = torch.exp(output)  # Convert logits to probabilities using softmax
predicted_class = torch.argmax(probabilities, dim=1).item()  # Get the predicted class index

# Depending on your model's output, you might want to map the class index to class labels
class_labels = ['negative', 'positive']  # Assuming your classes are negative and positive
predicted_label = class_labels[predicted_class]

print(f"The input text is classified as: {predicted_label}")