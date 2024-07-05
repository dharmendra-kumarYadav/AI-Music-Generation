import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import torch.nn.functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class GRUSequence(nn.Module): 
    def __init__(self, num_classes, seq_length, embedding_dimension=100, num_layers=3, hidden_size_gru=256):
        super(GRUSequence, self).__init__()
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.hidden_size_gru = hidden_size_gru
        self.embedding_dimension = embedding_dimension
        self.embedding = nn.Embedding(self.num_classes, self.embedding_dimension)
        self.gru = nn.GRU(input_size=self.embedding_dimension, hidden_size=self.hidden_size_gru, num_layers=self.num_layers, batch_first=True)
        self.linear1 = nn.Linear(self.hidden_size_gru, 128)
        self.relu = nn.ReLU(inplace=False)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, self.num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input):
        x = self.embedding(input)
        h_0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size_gru)
        x, _ = self.gru(x, h_0)
        gru_output = x[:, -1, :]
        output = self.linear3(self.relu(self.dropout(self.linear2(self.relu(self.dropout(self.linear1(self.relu(self.dropout(gru_output)))))))))
        return output


