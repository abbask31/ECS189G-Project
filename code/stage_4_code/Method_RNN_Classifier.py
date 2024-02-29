from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        # Initialize hidden state
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        # RNN layer
        out, hidden = self.rnn(x, hidden)

        # Get output of the last time step
        out = self.fc(hidden.squeeze(0))

        # Apply softmax activation
        out = self.softmax(out)
        return out

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)