import os
import torch
import time
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import nltk
from nltk.corpus import stopwords
import string


nltk.download('stopwords')
nltk.download('punkt')


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output size changed to 1

    def forward(self, x):
        embedded = self.embedding(x)
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(embedded, hidden)
        out = self.fc(hidden.squeeze(0))  # Remove softmax activation
        return out.squeeze(1)  # Ensure output shape is [batch_size]

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class TextDataset(Dataset):
    def __init__(self, data_dir, tokenizer, vocab, embedding_len):
        self.data = []
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.embedding_len = embedding_len
        self.stop_words = set(stopwords.words('english'))

        # Read data from neg and pos folders
        for label in ['neg', 'pos']:
            label_dir = os.path.join(data_dir, label)
            label_id = 0 if label == 'neg' else 1

            for filename in os.listdir(label_dir):
                with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    tokens = tokenizer(text.lower())

                    # Remove stopwords and punctuations, and normalize words
                    filtered_tokens = [word for word in tokens if
                                       word.isalpha() and word not in string.punctuation and word not in self.stop_words]

                    indexed_tokens = [self.vocab[token] if token in self.vocab else 0 for token in filtered_tokens]
                    self.data.append((indexed_tokens, label_id))

    def __len__(self):
        return len(self.data)

    def pad_sequences(self, max_length):
        for i, (embeddings, label) in enumerate(self.data):
            pad_len = max_length - len(embeddings)
            if pad_len > 0:
                padded_tensors = [0] * pad_len
                self.data[i] = (embeddings + padded_tensors, label)

    def __getitem__(self, index):
        tokens, label = self.data[index]
        return torch.tensor(tokens), torch.tensor([label],
                                                  dtype=torch.float32)  # Ensure label is a tensor with shape [1]


def get_dataset(dataset_dir, embedding_len):
    glove_file_path = r'data\stage_4_data\embedding'

    # Load GloVe embeddings
    glove = GloVe(name='6B', dim=embedding_len, cache=glove_file_path)

    tokenizer = get_tokenizer('basic_english')

    vocab = glove.stoi

    dataset = TextDataset(dataset_dir, tokenizer, vocab, embedding_len)

    # Compute maximum sequence length
    max_length = max(len(tokens) for tokens, _ in dataset)

    # Pad the sequences
    dataset.pad_sequences(max_length)

    return dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_dir = r'data\stage_4_data\text_classification\train'
test_dir = r'data\stage_4_data\text_classification\test'

# Define the size of GloVe embeddings
embedding_len = 100

# Create datasets
train_dataset = get_dataset(train_dir, embedding_len)
print("train loaded")
test_dataset = get_dataset(test_dir, embedding_len)
print("test loaded")

# Define the size of vocabulary
vocab_size = len(train_dataset.vocab)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the input size for the model
input_size = embedding_len
hidden_size = 128  # Size of hidden state in RNN
output_size = 1  # Output classes (neg and pos)

# Create RNN model
model = RNNClassifier(vocab_size, input_size, hidden_size).to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

num_epochs = 100

start = time.time()
# Training loop
# Inside the training loop
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

        # Ensure the shapes are compatible
        # outputs = torch.squeeze(outputs)  # Remove unnecessary dimensions
        # labels = torch.squeeze(labels)    # Remove unnecessary dimensions

        # Compute loss
        loss = criterion(outputs, labels.float())  # Convert labels to float

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Compute statistics
        running_loss += loss.item()
        total += labels.size(0)
        predicted = (outputs > 0.5).float()  # Threshold at 0.5 for binary classification
        correct += (predicted == labels).sum().item()

    # Print statistics
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')


end = time.time()
elapsed_time_seconds = end - start

# Convert to hours, minutes, and seconds
hours = int(elapsed_time_seconds // 3600)

minutes = int((elapsed_time_seconds % 3600) // 60)
seconds = int(elapsed_time_seconds % 60)
elapsed_time_str = f"{hours} hr : {minutes} min : {seconds} sec"

model_save_path = r'result\stage_4_result\text_classification\model.pkl'

# Save the model to a .pkl file
torch.save(model.state_dict(), model_save_path)
print(f"Elapsed time: {elapsed_time_str}")



# print(vocab)