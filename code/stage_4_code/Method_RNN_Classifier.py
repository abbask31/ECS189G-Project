from torchtext.vocab import GloVe
import torch
from torch import nn
import matplotlib.pyplot as plt

class Method_RNN_Classifier(nn.Module):
    data = None
    def __init__(self, mName='Classifer RNN', hidden_size=1028, num_layers=2, output_size=1):
        super(Method_RNN_Classifier, self).__init__()

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


    def train(self, train_loader):
        train_loss_history = []
        train_acc_history = []

        for epoch in range(self.num_epochs):

            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self(inputs).squeeze(1)
                loss = self.criterion(outputs, labels)

                # Compute gradients and update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Compute accuracy
                predicted = (outputs > 0.5).float()  # Convert to binary predictions
                correct = (predicted == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)

                total_loss += loss.item()

            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = total_correct / total_samples

            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_accuracy)

            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}')

        # Plotting
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_loss_history, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_acc_history, label='Train Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()

        plt.show()


    def test(self, test_loader):
        ypred_list = []
        ytrue_list = []

        with torch.no_grad():
            for feature, target in test_loader:
                feature, target = feature.to(self.device), target.to(self.device)

                out = self(feature).squeeze(1)

                predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5], device=self.device)

                # Append predicted and true labels to lists
                ypred_list.extend(predicted.cpu().tolist())
                ytrue_list.extend(target.cpu().tolist())

        return {'pred_y': ypred_list, 'true_y': ytrue_list}

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train'])
        print('--start testing...')
        return self.test(self.data['test'])
