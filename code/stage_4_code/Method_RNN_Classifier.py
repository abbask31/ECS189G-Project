from torchtext.vocab import GloVe

from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class Method_RNN_Classifier(nn.Module):
    data = None
    def __init__(self, mName, hidden_size=512, num_layers=2, output_size=1):
        super(Method_RNN_Classifier, self).__init__()

        glove = GloVe(name='6B', dim=100, cache=r'data\stage_4_data\embedding')
        self.glove = glove
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding.from_pretrained(glove.vectors)
        self.num_epochs =  20
        self.lr = 0.001

        self.lstm = nn.LSTM(100, hidden_size, num_layers, batch_first=True, dropout=0.2, bidirectional=False)
        self.rnn = nn.RNN(100, 1)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.3)
        self.sig = nn.Sigmoid()
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


    def train(self, train_loader):
        # Initialize model, loss, and optimizer

        # Train the model
        for epoch in range(self.num_epochs):
            # model.train()

            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self(inputs).squeeze(1)
                loss = self.criterion(outputs, labels)

                # Compute loss
                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.parameters(), 5)

                self.optimizer.step()

                # Compute accuracy
                predicted = (outputs > 0.5).float()  # Convert to binary predictions
                correct = (predicted == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)

                total_loss += loss.item()

            print(f"correct: {total_correct} total sample: {total_samples}")
            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = total_correct / total_samples

            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

    def test(self, test_loader):
        test_acc = 0
        test_loss = 0

        with torch.no_grad():
            for feature, target in test_loader:
                feature, target = feature.to(self.device), target.to(self.device)

                out = self(feature).squeeze(1)

                predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5], device=self.device)
                equals = predicted == target
                acc = torch.mean(equals.type(torch.FloatTensor))
                test_acc += acc.item()

                loss = self.criterion(out.squeeze(), target.float())
                test_loss += loss.item()

                # all_target.extend(target.cpu().numpy())
                # all_predicted.extend(predicted.cpu().numpy())

            print(f'Accuracy: {test_acc / len(test_loader):.4f}, Loss: {test_loss / len(test_loader):.4f}')

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train'])
        print('--start testing...')
        self.test(self.data['test'])
