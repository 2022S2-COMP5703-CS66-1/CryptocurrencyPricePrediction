import torch
import torch.nn as nn

import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=2000, mode="trigon", lambdda=0.1):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        if mode == 'trigon':
            div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        elif mode == 'tanh':
            pe[:, :] = torch.tanh(position) * lambdda
        else:
            raise Exception("mode must be either 'trigon' or 'tanh'.")
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PatternEmbedding(nn.Module):
    def __init__(self, c_in, kernel_size):
        super(PatternEmbedding, self).__init__()
        self.patternConv = nn.Conv1d(in_channels=c_in, out_channels=1,
                                     kernel_size=kernel_size, padding='same', padding_mode='circular')

    def forward(self, x):
        pattern_f = self.patternConv(x.permute(0, 2, 1)).transpose(1, 2)
        x = torch.cat((x, pattern_f), 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, pattern_embedding=False, positional_embedding='trigon'):
        super(DataEmbedding, self).__init__()

        self.pattern_embedding = PatternEmbedding(c_in=c_in, kernel_size=pattern_embedding) \
            if pattern_embedding else None
        self.value_embedding = TokenEmbedding(c_in=c_in + 1, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model, mode=positional_embedding)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.pattern_embedding is not None:
            x = self.pattern_embedding(x)
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x)
