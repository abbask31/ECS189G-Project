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
from nltk.corpus import wordnet as wn
import numpy as np

#from code.stage_4_code.main2 import SentimentAnalysisRNN

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
train_dir = r'data\stage_4_data\text_generation\data'

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

def get_first_three_words(text):
    return ' '.join(text.split()[:3])


def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = remove_punctuation(text)  # Remove punctuation
    #text = remove_stopwords(text)  # Remove stopwords
    text = lemmatize_text(text)  # Lemmatize words
    text = get_first_three_words(text)  # Get first three words of text
    return text

def create_dataset(dataset, pad_value, shuffle,  batch_size=128):
    # Convert sequences to tensors and truncate if necessary
    pad_len = 200
    sequences = [sample[0] for sample in dataset]  # Extract sequences (X values)
    #sequences = [torch.tensor(seq[:pad_len], dtype=torch.long) for seq in dataset.samples]
    pad_len = 200
    padded_sequences = pad_sequence([seq[:pad_len] for seq in sequences],
                                    batch_first=True,
                                    padding_value=pad_value)

    t_dataset = TensorDataset(padded_sequences)

    # Return a DataLoader
    return DataLoader(t_dataset, batch_size=batch_size, shuffle=shuffle)

# Define dataset class
class IMDbDataset(Dataset):
    def __init__(self, directory, word_to_idx):
        self.directory = directory
        self.samples = []
        self.word_to_idx = word_to_idx

        file_path = directory  # Update this path to your file's location

        with open(file_path, 'r') as file:
            next(file)  # Skip the first row (header)
            for line in file:
                values = line.strip().split(',')  # Split each line into a list of values
                if len(values) > 1:  # Ensure there is at least a second value
                    text = values[1]
                    text = preprocess_text(text)
                    self.samples.append(text)  # Add the second value of each row to the list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Retrieve the text sample at the specified index
        text = self.samples[idx]
        word_indices = [self.word_to_idx[word] if word in self.word_to_idx else 0 for word in text.split()]
        return torch.tensor(word_indices)


class RNNGenerator(nn.Module):
    data = None
    def __init__(self, mName='Classifer RNN', hidden_size=512, num_layers=2, output_size=1):
        super(RNNGenerator, self).__init__()

        # Setup
        self.method_name = mName
        self.glove = GloVe(name='6B', dim=100, cache=r'data\stage_4_data\embedding')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Layers
        self.lstm = nn.LSTM(100, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=False)
        self.embedding = nn.Embedding.from_pretrained(self.glove.vectors)
        self.rnn = nn.RNN(100, 1)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.3)
        self.sig = nn.Sigmoid()

        # Hyperparams
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_epochs = 12
        self.lr = 0.001
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.to(self.device)


    def forward(self, x):

        x = x.long()

        x = self.embedding(x)

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.dropout(out)

        out = self.fc1(out)

        out = self.sig(out)

        return out
def generate_text(model, vocab, initial_seq="The", num_words=3):
    model.eval()  # Put the model in evaluation mode

    # Convert initial sequence to tensor of indices
    indices = [vocab[word] for word in initial_seq.split()]
    input_seq = torch.tensor([indices], dtype=torch.long)

    generated_words = []
    hidden = None

    for _ in range(num_words):
        output, hidden = model(input_seq, hidden)
        probabilities = torch.softmax(output[0, -1],
                                      dim=0).detach().numpy()  # Get probabilities for the last output word
        predicted_index = np.random.choice(len(probabilities),
                                           p=probabilities)  # Sample a word index based on output probabilities
        generated_words.append(predicted_index)

        # Update the input sequence with the predicted word index
        input_seq = torch.cat((input_seq[0], torch.tensor([[predicted_index]])), dim=1)
        input_seq = input_seq[:, -1:].reshape((1, 1))  # Keep the sequence length consistent

    # Convert generated indices back to words
    inv_vocab = {index: word for word, index in vocab.items()}
    generated_text = ' '.join([inv_vocab[index] for index in generated_words])

    return generated_text

train_loader_path = r'data\stage_4_data\loaded_data\train_loader.pkl'
train_data = []
train_loader = []

# Check if loaders exist
if os.path.exists(train_loader_path):
    # Load loaders from the saved files
    with open(train_loader_path, 'rb') as f:
        train_loader = pickle.load(f)
else:
    # Create dataset and data loader
    train_data = IMDbDataset(train_dir, word_to_idx)

    train_loader = create_dataset(dataset=train_data, pad_value=0, shuffle=True)

    # Save the loaders
    with open(train_loader_path, 'wb') as f:
        pickle.dump(train_loader, f)

# Initialize model, loss, and optimizer
embedding = glove

# Define hyperparameters
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

model = RNNGenerator(embedding, 512, 2, 1).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model

for epoch in range(NUM_EPOCHS):

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        output = model(inputs).squeeze(1)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
    print('epoch ' + str(epoch) + ' complete')

# Save the model
torch.save(model.state_dict(), 'joke_generator_rnn_model.pth')

# Build vocabulary
vocab = {lemma.name(): idx for idx, lemma in enumerate(set(wn.all_lemma_names()))}
vocab['<pad>'] = len(vocab)  # Adding a padding token
inv_vocab = {idx: word for word, idx in vocab.items()}  # Inverse mapping

generated_sequence = generate_text(model, vocab, initial_seq="The", num_words=3)
print(generated_sequence)