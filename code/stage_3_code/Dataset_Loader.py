'''
Concrete IO class for a specific dataset
'''
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pickle

class CustomDataset(Dataset):
    def __init__(self, data, dataset_name, transform=None):
        self.data = data
        self.transform = transform
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        label = item['label']

        if self.dataset_name == 'ORL':
            image = image[:,:,0]
            label -= 1

        if self.transform:
            image = self.transform(image)

        return image, label


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

        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize for a single grayscale channel
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

