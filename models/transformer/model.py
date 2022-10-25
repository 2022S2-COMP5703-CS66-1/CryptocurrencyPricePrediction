import torch.nn as nn
from models.transformer.positionalEncoding import PositionalEncoding


class Transformer(nn.Module):

    def __init__(self, pred_len=1, c_out_dim=1, position_encoding=None, d_model=512, **kwargs):
        super(Transformer, self).__init__()
        
        self.position_encoding = nn.ModuleList()
        
        if d_model % 2 != 0:
            d_model += 1
            self.position_encoding.append(nn.ZeroPad2d((0, 1, 0, 0)))
            
        self.position_encoding.append(PositionalEncoding(d_model) if position_encoding is None else position_encoding)
            
        self.back_bone = nn.Transformer(d_model=d_model, **kwargs)
        self.projection = nn.Linear(d_model, c_out_dim)
        self.pred_len = pred_len

    def forward(self, enc_in, dec_in):
        
        for l in self.position_encoding:
            enc_in = l(enc_in)
            dec_in = l(dec_in)

        out = self.back_bone(enc_in, dec_in)
        out = self.projection(out)
        return out[:, -self.pred_len:, :]