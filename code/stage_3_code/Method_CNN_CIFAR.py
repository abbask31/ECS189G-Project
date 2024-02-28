'''
Concrete MethodModule class for a specific learning MethodModule
'''
from graphviz import Digraph
from torchviz import make_dot

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class Method_CNN_CIFAR(method, nn.Module):
    data = None
    device = None

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # it defines the max rounds to train the model
    max_epoch = 50
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

        # Process on GPU IF AVAILABLE
        self.to(self.device)

    def forward(self, x):
        '''Forward propagation'''
        x = x.to(self.device)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)

        return x

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y, batch_size=64):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        train_loss_history = []
        train_accuracy_history = []

        num_batches = len(X) // batch_size

        for epoch in range(self.max_epoch):
            epoch_loss = 0.0
            epoch_accuracy = 0.0

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size

                batch_X = torch.FloatTensor(X[start_idx:end_idx])
                batch_y = torch.LongTensor(y[start_idx:end_idx])

                if torch.cuda.is_available():
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                # Forward pass
                y_pred = self.forward(batch_X)

                # Compute loss
                batch_loss = loss_function(y_pred, batch_y)
                epoch_loss += batch_loss.item()

                # Backward pass and optimization
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # Evaluate accuracy for the batch
                batch_accuracy = accuracy_evaluator.evaluate(batch_y, y_pred.max(1)[1])
                epoch_accuracy += batch_accuracy

            epoch_loss /= num_batches
            epoch_accuracy /= num_batches

            train_loss_history.append(epoch_loss)
            train_accuracy_history.append(epoch_accuracy)

            print('Epoch:', epoch + 1, 'Accuracy:', epoch_accuracy, 'Loss:', epoch_loss)
        # Plot training loss and accuracy
        plt.figure(figsize=(12, 5))

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.max_epoch + 1), train_loss_history, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        # Plot training accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, self.max_epoch + 1), train_accuracy_history, label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def test(self, X):

        device = next(self.parameters()).device
        X = torch.tensor(X, device=device)

        # do the testing, and result the result
        y_pred = self.forward(X)
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        # return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
        pred_y_tensor = torch.tensor(pred_y, device=self.device) if isinstance(pred_y, np.ndarray) else pred_y
        return {'pred_y': pred_y_tensor.cpu().numpy(), 'true_y': self.data['test']['y']}


# model = Method_CNN_CIFAR("name", "description")
#
# # Create a random input tensor to visualize the model
# x = torch.randn(1, 3, 32, 32)
#
# # Pass the input tensor through the model to generate a computational graph
# y = model(x)
#
# # Visualize the computational graph using Graphviz
# dot = make_dot(y, params=dict(model.named_parameters()))
# dot.render("model_architecture", format="png")