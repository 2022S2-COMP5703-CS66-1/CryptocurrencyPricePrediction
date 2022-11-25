import torch.nn as nn


class Truncater(nn.Module):
    """
    A truncater used to truncate the extra data result from the padding of dilation.
    """
    def __init__(self, padding: int):
        super(Truncater, self).__init__()
        self.padding = padding

    def forward(self, x):
        x = x[:, :, :-self.padding].contiguous()
        return x


class CasualConv(nn.Module):
    """
    Casual Convolution operator.
    """
    def __init__(self, c_in: int, kernel_size=2, n_hidden=2, activation=nn.ELU, dropout=0.):
        super(CasualConv, self).__init__()
        # Generating multiple convolutional layers depend on hyper-parameter
        hiddens = []
        for i in range(n_hidden):
            d = 2 ** i
            # Calculating padding to keep the sequence length after dilated convolution.
            padding = (kernel_size - 1) * d
            hiddens.append(nn.Conv1d(in_channels=c_in,
                                     out_channels=c_in,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     dilation=d))
            hiddens.append(Truncater(padding))
            hiddens.append(activation())
            hiddens.append(nn.Dropout(p=dropout))
        del hiddens[-1]  # Do not dropout at the final output
        self.net = nn.Sequential(*hiddens)

    def forward(self, x):
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)
