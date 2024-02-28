from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class Method_RNN_Classifier(method, nn.Module):

    def __init__(self):
        pass