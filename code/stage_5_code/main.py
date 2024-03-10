import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from code.stage_5_code.Dataset_Loader import Dataset_Loader
from torch_geometric.nn import GCNConv
import torch.optim as optim
from code.stage_5_code.Method_GNN_Cora import Method_GNN_Cora
class GCN(nn.Module):
    def __init__(self, nfeat=1433, nhid=40, nclass=7, dropout=0.5):
        super(GCN, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

data = Dataset_Loader()
data.dataset_name = 'cora'
data.dataset_source_folder_path = r'data\stage_5_data\cora'
data = data.load()

device = "cuda" if torch.cuda.is_available() else "cpu"


graph = data['graph']
train_test = data['train_test_val']

features = graph['X'].to(device)
labels = graph['y'].to(device)
adj = graph['utility']['A'].to(device)

idx_train = train_test['idx_train']
idx_test = train_test['idx_test']


model = GCN().to(device)
optimizer = optim.Adam(model.parameters(),
                       lr=0.001, weight_decay=5e-2)

criterion = nn.NLLLoss()


for epoch in range(300):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = criterion(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()


    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))

# Testing
model.eval()
output = model(features, adj)
loss_test = F.nll_loss(output[idx_test], labels[idx_test])
acc_test = accuracy(output[idx_test], labels[idx_test])
print("Test set results:",
      "loss= {:.4f}".format(loss_test.item()),
      "accuracy= {:.4f}".format(acc_test.item()))