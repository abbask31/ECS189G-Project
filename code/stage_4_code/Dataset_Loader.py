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

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading dataset...')
        train_X = []
        train_y = []

        test_X = []
        test_y = []

        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()

        mean = (0.5, )
        std = (0.5, )

        if self.dataset_source_file_name[1:] == 'CIFAR':
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)

        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)  # Normalize for a single grayscale channel
        ])

        train_data, test_data = data['train'], data['test']

        train_dataset = CustomDataset(train_data, self.dataset_source_file_name[1:], transform=data_transforms)
        test_dataset = CustomDataset(test_data, self.dataset_source_file_name[1:], transform=data_transforms)

        train_loader = DataLoader(train_dataset, shuffle=True)
        test_loader = DataLoader(test_dataset,  shuffle=False)

        for train_batch in train_loader:
            batch_X, batch_y = train_batch
            train_X.append(batch_X)
            train_y.append(batch_y)

        # Convert the accumulated lists to tensors
        train_X = torch.cat(train_X, dim=0)
        train_y = torch.cat(train_y, dim=0)

        for test_batch in test_loader:
            batch_X, batch_y = test_batch
            test_X.append(batch_X)
            test_y.append(batch_y)

        # Convert the accumulated lists to tensors
        test_X = torch.cat(test_X, dim=0)
        test_y = torch.cat(test_y, dim=0)

        train_data = {'X': train_X, 'y': train_y}
        test_data = {'X': test_X, 'y': test_y}

        print('done loading')

        return [train_data, test_data]

