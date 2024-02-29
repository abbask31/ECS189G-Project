import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import nltk
from nltk.corpus import stopwords
import string

from collections import Counter
from torchtext.vocab import Vocab
nltk.download('stopwords')
nltk.download('punkt')

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

class TextDataset(Dataset):
    def __init__(self, data_dir, tokenizer, vocab, embedding):
        self.data = []
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.embedding = embedding
        self.stop_words = set(stopwords.words('english'))

        # Read data from neg and pos folders
        for label in ['neg', 'pos']:
            label_dir = os.path.join(data_dir, label)
            label_id = 0 if label == 'neg' else 1

            for filename in os.listdir(label_dir):
                with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    # print(text)
                    tokens = tokenizer(text.lower())

                    # Remove stopwords and punctuations, and normalize words
                    filtered_tokens = [word for word in tokens if
                                       word.isalpha() and word not in string.punctuation and word not in self.stop_words]
                    # print(tokens)
                    # print(len(tokens))
                    oov_value = torch.tensor([0] * self.embedding, dtype=torch.long)
                    indexed_tokens = [self.vocab[token] if token in self.vocab else oov_value for token in
                                      filtered_tokens]
                    self.data.append((indexed_tokens, label_id))


    def __len__(self):
        return len(self.data)

    def pad_sequences(self, max_length):
        for i, (embeddings, label) in enumerate(self.data):
            pad_len = max_length - len(embeddings)
            if pad_len > 0:
                padded_tensors = [torch.tensor([0] * self.embedding, dtype=torch.long)] * pad_len
                self.data[i] = (embeddings + padded_tensors, label)

    def __getitem__(self, index):
        tokens, label = self.data[index]
        return torch.stack(tokens), label


def get_dataset(dataset_dir, embedding_len):
    glove_file_path = r'data\stage_4_data\embedding'

    # Load GloVe embeddings
    glove = GloVe(name='6B', dim=100, cache=glove_file_path)

    tokenizer = get_tokenizer('basic_english')

    dataset = TextDataset(dataset_dir, tokenizer, glove, embedding_len)

    # Compute maximum sequence length
    max_length = max(len(tokens) for tokens, _ in dataset)

    # Pad the sequences
    dataset.pad_sequences(max_length)

    return dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



train_dir = r'data\stage_4_data\text_classification\train'
test_dir = r'data\stage_4_data\text_classification\test'


# Create datasets
train_dataset = get_dataset(train_dir, 100)
print("train loaded")
test_dataset = get_dataset(test_dir, 100)
print("test loaded")

batch_size = 250
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

input_size = 100  # Size of GloVe embeddings (100)
hidden_size = 128  # Size of hidden state in RNN
output_size = 2  # Number of classes (neg and pos)

# Create RNN model
model = RNNClassifier(input_size, hidden_size, output_size).to(device)

# Define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

num_epochs = 10

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

model_save_path = r'result\stage_4_result\text_classification\model.pkl'

# Save the model to a .pkl file
torch.save(model.state_dict(), model_save_path)


# print(vocab)