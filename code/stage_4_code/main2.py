import os
import re
import string
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from torchtext.vocab import GloVe
from torch.nn.utils.rnn import pad_sequence
import pickle

nltk.download('wordnet')

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
train_dir = r'data\stage_4_data\text_classification\train'
test_dir = r'data\stage_4_data\text_classification\test'

# Load GloVe embeddings
glove = GloVe(name='6B', dim=100, cache=r'data\stage_4_data\embedding')

# Create word-to-index mapping
word_to_idx = {word: idx for idx, word in enumerate(glove.itos)}

# Define preprocessing functions


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)


def lemmatize_text(text):
    # Initialize WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    lemmatized_words = [lemmatizer.lemmatize(word) for word in text.split()]
    return ' '.join(lemmatized_words)


def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = remove_punctuation(text)  # Remove punctuation
    text = remove_stopwords(text)  # Remove stopwords
    text = lemmatize_text(text)  # Lemmatize words
    return text


def create_dataset(dataset, pad_value, shuffle,  batch_size=128):

    sequences = [sample[0] for sample in dataset]  # Extract sequences (X values)
    labels = [sample[1] for sample in dataset]  # Extract labels

    # sequence_lengths = [len(seq) for seq in sequences]
    #
    # # Step 2: Create a histogram of sequence lengths
    # plt.figure(figsize=(10, 6))
    # plt.hist(sequence_lengths, bins=50, color='skyblue', edgecolor='black')
    # plt.title('Histogram of Sequence Lengths')
    # plt.xlabel('Sequence Length')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()

    # Pad sequences
    pad_len = 200
    padded_sequences = pad_sequence([seq[:pad_len] for seq in sequences],
                                    batch_first=True,
                                    padding_value=pad_value)

    # Convert labels to tensor
    labels_tensor = torch.tensor(labels)

    t_dataset = TensorDataset(padded_sequences, labels_tensor)

    return DataLoader(t_dataset, batch_size=batch_size, shuffle=shuffle)


# Define dataset class
class IMDbDataset(Dataset):
    def __init__(self, directory, word_to_idx):
        self.directory = directory
        self.samples = []
        self.word_to_idx = word_to_idx

        for label in ['pos', 'neg']:
            label_path = os.path.join(directory, label)
            label_id = 1 if label == 'pos' else 0

            for filename in os.listdir(label_path):
                with open(os.path.join(label_path, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    text = preprocess_text(text)
                    self.samples.append((text, label_id))



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        word_indices = [self.word_to_idx[word] if word in self.word_to_idx else 0 for word in text.split()]
        return torch.tensor(word_indices), torch.tensor(label, dtype=torch.float)

# Define RNN model
class SentimentAnalysisRNN(nn.Module):
    def __init__(self, embedding, hidden_size, num_layers, output_size):
        super(SentimentAnalysisRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding.from_pretrained(embedding.vectors)

        self.lstm = nn.LSTM(100, hidden_size, num_layers, batch_first=True, dropout=0.5, bidirectional=False)
        self.rnn = nn.RNN(100, 1)
        self.fc1 = nn.Linear(2*hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.5)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        #
        # embedded = self.embedding(x)
        # # idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        # # decoded_sentence = [idx_to_word[idx.item()] for idx in x[0]]
        # # print(len(decoded_sentence))
        # # print(decoded_sentence)
        # out, _ = self.lstm(embedded, (h0, c0))
        # # print(f'before reshape out:{out}')
        # out = self.fc(out[:, -1, :])
        # # print(out)
        # # print(f'after reshape out:{out}')
        # out = torch.sigmoid(out)
        # # print(out)
        # # print(" ")
        # return out

        # Get the word embeddings of the batch
        embedded = self.embedding(x)
        # Propagate the input through LSTM layer/s
        _, (hidden, _) = self.lstm(embedded)

        # Extract output of the last time step
        # Extract forward and backward hidden states of the
        # last time step
        out = torch.cat([hidden[-2, :, :], hidden[-1, :, :]], dim=1)

        out = self.dropout(out)
        out = self.fc1(out)
        out = self.sig(out)

        return out






train_loader_path = r'data\stage_4_data\loaded_data\train_loader.pkl'
test_loader_path = r'data\stage_4_data\loaded_data\test_loader.pkl'
# Check if loaders exist
if os.path.exists(train_loader_path) and os.path.exists(test_loader_path):
    # Load loaders from the saved files
    with open(train_loader_path, 'rb') as f:
        train_loader = pickle.load(f)
    with open(test_loader_path, 'rb') as f:
        test_loader = pickle.load(f)
else:
    # Create dataset and data loader
    train_data = IMDbDataset(train_dir, word_to_idx)
    test_data = IMDbDataset(test_dir, word_to_idx)

    train_loader = create_dataset(dataset=train_data, pad_value=0, shuffle=True)
    test_loader = create_dataset(dataset=test_data, pad_value=0, shuffle=False)

    # Save the loaders
    with open(train_loader_path, 'wb') as f:
        pickle.dump(train_loader, f)
    with open(test_loader_path, 'wb') as f:
        pickle.dump(test_loader, f)


# Initialize model, loss, and optimizer
embedding = glove

# Define hyperparameters
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

model = SentimentAnalysisRNN(embedding, 256, 2, 1).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)

        # Compute loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        predicted = (outputs > 0.5).float()  # Convert to binary predictions
        correct = (predicted == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)

        total_loss += loss.item()



    print(f"correct: {total_correct} total sample: {total_samples}")
    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = total_correct / total_samples

    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs).squeeze(1)
        predicted = torch.round(torch.sigmoid(outputs))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the networks: {100 * correct / total}%')

# Save the model
torch.save(model.state_dict(), 'sentiment_analysis_rnn_model.pth')
