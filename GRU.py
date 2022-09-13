import torch
from torch import nn
from torch.nn import functional as F

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,  # the number of input's features
            hidden_size=hidden_size,  # the number of features of hidden layer
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout
        )
        self.linear = nn.Linear(hidden_size, output_size)  # output layer

    def forward(self, input, h_state):
        """
        input: The value of the sample at all times (batch, seq, hidden_size)
        h_state: h0
        output:(batch ,seq ,hidden_size)/ h_state (batch, num_layer, hidden_size)
        """
        output, h_state = self.gru(input, h_state)

        # Output is processed with a linear layer
        # [batch=1,seq_len,hidden_len]->[seq_len,hidden_len]->[seq_len,output_size=1]->[batch=1,seq_len,output_size=1]
        output = output.view(-1, h_state)
        output = self.linear(output)

        output = output.unsqueeze(dim=0)

        return output, h_state