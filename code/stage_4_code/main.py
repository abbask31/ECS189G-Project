import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

from collections import Counter
from torchtext.vocab import Vocab


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Initialize hidden state
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        # RNN layer
        out, hidden = self.rnn(x, hidden)

        # Get output of the last time step
        out = self.fc(hidden.squeeze(0))

        # Apply softmax activation
        out = self.softmax(out)
        return out

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

class TextDataset(Dataset):
    def __init__(self, data_dir, tokenizer, vocab):
        self.data = []
        self.tokenizer = tokenizer
        self.vocab = vocab

        # Read data from neg and pos folders
        for label in ['neg', 'pos']:
            label_dir = os.path.join(data_dir, label)
            label_id = 0 if label == 'neg' else 1

            for filename in os.listdir(label_dir):
                with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    tokens = tokenizer(text.lower())
                    indexed_tokens = [self.vocab[token] if token in self.vocab else 0 for token in tokens]
                    self.data.append((indexed_tokens, label_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tokens, label = self.data[index]

        max_length = max(len(tokens) for tokens, label in self.data)

        # Step 2: Pad each sequence of indexed tokens to match the maximum length
        padded_data = []
        for tokens, label in self.data:
            # Pad the tokens using torch.nn.utils.rnn.pad_sequence
            padded_tokens = torch.tensor(tokens)  # Convert tokens to tensor
            padded_tokens = pad_sequence([padded_tokens], batch_first=True, padding_value=0)  # Pad the tokens
            padded_tokens = padded_tokens[0]  # Extract the padded tokens from the batch
            padded_data.append((padded_tokens, label))

        return padded_data, torch.tensor(label)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# data_dir = r'data\stage_4_data'
glove_file_path = r'data\stage_4_data\embedding'

# Load GloVe embeddings
glove = GloVe(name='6B', dim=100, cache=glove_file_path)


train_dir = r'data\stage_4_data\text_classification\train'
test_dir = r'data\stage_4_data\text_classification\test'

tokenizer = get_tokenizer('basic_english')

# Create datasets
train_dataset = TextDataset(train_dir, tokenizer, glove)
test_dataset = TextDataset(test_dir, tokenizer, glove)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

input_size = 100  # Size of GloVe embeddings (100 in your case)
hidden_size = 128  # Size of hidden state in RNN
output_size = 2  # Number of classes (neg and pos)

# Create RNN model
model = RNNClassifier(input_size, hidden_size, output_size).to(device)

# Define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Compute statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print statistics
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

# print(vocab)