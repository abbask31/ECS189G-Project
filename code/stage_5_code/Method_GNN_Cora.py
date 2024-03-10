import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class Method_GNN_Cora(nn.Module):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = None

    def __init__(self, nfeat=1433, nclass=7, nhid=40, dropout=0.5):
        super(Method_GNN_Cora, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout

        self.num_epochs = 300
        self.optimizer = optim.Adam(self.parameters(),
                                    lr=0.001, weight_decay=5e-2)
        self.criterion = nn.NLLLoss()

        self.to(self.device)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def train_model(self, graph, idx_train):
        features = graph['X'].to(self.device)
        labels = graph['y'].to(self.device)
        adj = graph['utility']['A'].to(self.device)

        for epoch in range(self.num_epochs):
            t = time.time()
            self.train()
            self.optimizer.zero_grad()
            output = self(features, adj)
            loss_train = self.criterion(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            self.optimizer.step()

            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'time: {:.4f}s'.format(time.time() - t))

    def test_model(self, graph, idx_test):
        features = graph['X'].to(self.device)
        labels = graph['y'].to(self.device)
        adj = graph['utility']['A'].to(self.device)
        self.eval()
        output = self(features, adj)
        loss_test = self.criterion(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

    def run(self):
        graph = self.data['graph']
        train_test = self.data['train_test_val']
        idx_train = train_test['idx_train']
        idx_test = train_test['idx_test']
        print('---start training---')
        self.train_model(graph, idx_train)
        print('---start testing---')
        self.test_model(graph, idx_test)

        # return self.state_dict()
