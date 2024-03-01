'''
Concrete IO class for a specific dataset
'''
import os
import re
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from code.base_class.dataset import dataset
import nltk
from torchtext.vocab import GloVe
from torch.nn.utils.rnn import pad_sequence
import pickle

nltk.download('wordnet')

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

    # Pad sequences
    pad_len = 200
    padded_sequences = pad_sequence([seq[:pad_len] for seq in sequences],
                                    batch_first=True,
                                    padding_value=pad_value)

    # Convert labels to tensor
    labels_tensor = torch.tensor(labels)

    t_dataset = TensorDataset(padded_sequences, labels_tensor)

    return DataLoader(t_dataset, batch_size=batch_size, shuffle=shuffle)


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


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    train_classifer_path = None
    test_classifier_path = None
    task = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading dataset...')

        if self.task == 'classification':
            if os.path.exists(self.train_classifer_path) and os.path.exists(self.test_classifier_path):
                # Load loaders from the saved files
                with open(self.train_classifer_path, 'rb') as f:
                    train_loader = pickle.load(f)
                with open(self.test_classifier_path, 'rb') as f:
                    test_loader = pickle.load(f)
            else:
                # Create dataset and data loader
                train_data = IMDbDataset(self.dataset_source_folder_path + r'\train', word_to_idx)
                test_data = IMDbDataset(self.dataset_source_folder_path + r'\test', word_to_idx)

                train_loader = create_dataset(dataset=train_data, pad_value=0, shuffle=True)
                test_loader = create_dataset(dataset=test_data, pad_value=0, shuffle=True)

                # Save the loaders
                with open(self.train_classifer_path, 'wb') as f:
                    pickle.dump(train_loader, f)
                with open(self.test_classifier_path, 'wb') as f:
                    pickle.dump(test_loader, f)
        else:
            ## add generator files here
            pass


        data = {'train':train_loader, 'test':test_loader}

        print('done loading')

        return data
