'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp

class Dataset_Loader(dataset):
    data = None
    dataset_name = None

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)

    def adj_normalize(self, mx):
        """normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def load(self):
        """Load citation network dataset"""
        print('Loading {} dataset...'.format(self.dataset_name))

        # load node data from file
        idx_features_labels = np.genfromtxt(r"{}\node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # load link data from file and build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(r"{}\link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        # the following part, you can either put them into the setting class or you can leave them in the dataset loader
        # the following train, test, val index are just examples, sample the train, test according to project requirements
        if self.dataset_name == 'cora':
            num_classes = 7
            nodes_per_class_train = 20
            nodes_per_class_test = 150

            # Initialize lists to store indices for each class
            class_indices_train = [[] for _ in range(num_classes)]
            class_indices_test = [[] for _ in range(num_classes)]

            # Iterate through the labels to collect indices for each class
            for idx, label in enumerate(labels):
                class_indices_train[label.item()].append(idx)
                class_indices_test[label.item()].append(idx)

            # Sample nodes for the training and testing sets
            idx_train = []
            idx_test = []

            for class_idx in class_indices_train:
                idx_train += np.random.choice(class_idx, nodes_per_class_train, replace=False).tolist()

            for class_idx in class_indices_test:
                idx_test += np.random.choice(class_idx, nodes_per_class_test, replace=False).tolist()

            idx_val = torch.LongTensor(np.random.choice(range(1200, 1500), size=20, replace=False).tolist() * 7)
        elif self.dataset_name == 'citeseer':
            num_classes = 6
            nodes_per_class_train = 20
            nodes_per_class_test = 200

            # Initialize lists to store indices for each class
            class_indices_train = [[] for _ in range(num_classes)]
            class_indices_test = [[] for _ in range(num_classes)]

            # Iterate through the labels to collect indices for each class
            for idx, label in enumerate(labels):
                class_indices_train[label.item()].append(idx)
                class_indices_test[label.item()].append(idx)

            # Sample nodes for the training and testing sets
            idx_train = []
            idx_test = []

            for class_idx in class_indices_train:
                idx_train += np.random.choice(class_idx, nodes_per_class_train, replace=False).tolist()

            for class_idx in class_indices_test:
                idx_test += np.random.choice(class_idx, nodes_per_class_test, replace=False).tolist()
            idx_val = range(1200, 1500)
        elif self.dataset_name == 'pubmed':
            num_classes = 3
            nodes_per_class_train = 20
            nodes_per_class_test = 200

            # Initialize lists to store indices for each class
            class_indices_train = [[] for _ in range(num_classes)]
            class_indices_test = [[] for _ in range(num_classes)]

            # Iterate through the labels to collect indices for each class
            for idx, label in enumerate(labels):
                class_indices_train[label.item()].append(idx)
                class_indices_test[label.item()].append(idx)

            # Sample nodes for the training and testing sets
            idx_train = []
            idx_test = []

            for class_idx in class_indices_train:
                idx_train += np.random.choice(class_idx, nodes_per_class_train, replace=False).tolist()

            for class_idx in class_indices_test:
                idx_test += np.random.choice(class_idx, nodes_per_class_test, replace=False).tolist()
            idx_val = range(6000, 6300)
        #---- cora-small is a toy dataset I hand crafted for debugging purposes ---
        elif self.dataset_name == 'cora-small':
            idx_train = range(5)
            idx_val = range(5, 10)
            idx_test = range(5, 10)

        # idx_train = torch.LongTensor(idx_train)
        # idx_val = torch.LongTensor(idx_val)
        # idx_test = torch.LongTensor(idx_test)
        # get the training nodes/testing nodes
        # train_x = features[idx_train]
        # val_x = features[idx_val]
        # test_x = features[idx_test]
        # print(train_x, val_x, test_x)

        train_test_val = {'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels, 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
        return {'graph': graph, 'train_test_val': train_test_val}

# cora_data = Dataset_Loader()
# cora_data.dataset_name = 'cora'
# cora_data.dataset_source_folder_path = r'data\stage_5_data\cora'
# cora_data = cora_data.load()
#
# print("Number of nodes in Cora dataset:", len(cora_data['graph']['node']))
# print("Number of links in Cora dataset:", len(cora_data['graph']['edge']))
#
# # Inspect Training and Testing Sets
# print("Number of nodes in Cora training set:", len(cora_data['train_test_val']['idx_train']))
# print("Number of nodes in Cora testing set:", len(cora_data['train_test_val']['idx_test']))
#
# # Check if the number of nodes per class is correct in the training set
# class_counts_train = {i: (cora_data['graph']['y'][cora_data['train_test_val']['idx_train']] == i).sum().item() for i in range(3)}
#
# print("Number of nodes per class in Cora training set:", class_counts_train)
#
# # Check if the number of nodes per class is correct in the testing set
# class_counts_test = {i: (cora_data['graph']['y'][cora_data['train_test_val']['idx_test']] == i).sum().item() for i in range(3)}
# print("Number of nodes per class in Cora testing set:", class_counts_test)
#
# # Check Node Features and Labels
# # You can print the features and labels of a few nodes to ensure they are correctly assigned
# sample_node_indices = [0, 1, 2]  # You can choose any random indices
# for node_index in sample_node_indices:
#     print("Node Index:", node_index)
#     print("Features:", cora_data['graph']['X'][node_index])
#     print("Label:", cora_data['graph']['y'][node_index])
#
# # Ensure that the nodes are randomly sampled by checking the first few indices
# print("First 10 indices in Cora training set:", cora_data['train_test_val']['idx_train'][:10])
# print("First 10 indices in Cora testing set:", cora_data['train_test_val']['idx_test'][:10])