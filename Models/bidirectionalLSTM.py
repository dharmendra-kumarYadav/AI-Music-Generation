import torch
import torch.nn as nn

class Sequence(nn.Module): 
    def __init__(self, num_classes, seq_length, embedding_dimension=100, num_layers=3, hidden_size_lstm=256):
        super(Sequence, self).__init__()
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.hidden_size_lstm = hidden_size_lstm
        self.embedding_dimension = embedding_dimension
        
        # Define the embedding layer
        self.embedding = nn.Embedding(self.num_classes, self.embedding_dimension)
        
        # Define the bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size=self.embedding_dimension, 
                            hidden_size=self.hidden_size_lstm, 
                            num_layers=self.num_layers, 
                            batch_first=True, 
                            bidirectional=True)  # Bidirectional LSTM
        
        # Define the linear layers
        self.linear1 = nn.Linear(self.hidden_size_lstm * 2, 128)  # *2 for bidirectional
        self.relu = nn.ReLU(inplace=False)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, self.num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input):
        # Perform the forward pass
        
        # Apply embedding layer
        x = self.embedding(input)
        
        # Forward pass through the bidirectional LSTM layer
        lstm_output, _ = self.lstm(x)
        
        # Concatenate the hidden states from both directions
        lstm_output = torch.cat((lstm_output[:, -1, :self.hidden_size_lstm], 
                                 lstm_output[:, 0, self.hidden_size_lstm:]), 
                                 dim=1)
        
        # Pass through linear layers
        output = self.linear3(self.relu(self.dropout(self.linear2(self.relu(self.dropout(self.linear1(self.relu(self.dropout(lstm_output)))))))))
        
        return output
