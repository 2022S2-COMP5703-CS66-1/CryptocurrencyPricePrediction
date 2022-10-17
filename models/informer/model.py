import torch
import torch.nn as nn

from models.informer.encoder import Encoder, EncoderLayer, ConvLayer
from models.informer.decoder import Decoder, DecoderLayer
from models.informer.attention import FullAttention, ProbAttention, AttentionLayer
from models.informer.embedding import DataEmbedding


class Informer(nn.Module):
    def __init__(self, enc_in_dim, dec_in_dim, c_out_dim, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', activation='gelu', positional_embedding='trigon',
                 output_attention=False, distil=True, mix=True, pattern_embedding=False, conv_trans=False,
                 conv_trans_kernel_size=8, dilation_n_hidden=2, conv_trans_activation=nn.ELU):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in_dim,
                                           d_model,
                                           dropout=dropout,
                                           pattern_embedding=pattern_embedding,
                                           positional_embedding=positional_embedding)
        self.dec_embedding = DataEmbedding(dec_in_dim,
                                           d_model,
                                           dropout=dropout,
                                           pattern_embedding=pattern_embedding,
                                           positional_embedding=positional_embedding)

        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False, conv_trans=conv_trans,
                                   kernel_size=conv_trans_kernel_size, dilation_n_hidden=dilation_n_hidden,
                                   conv_activation=conv_trans_activation, dropout=dropout),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for _ in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix, conv_trans=conv_trans,
                                   kernel_size=conv_trans_kernel_size, dilation_n_hidden=dilation_n_hidden,
                                   conv_activation=conv_trans_activation, dropout=dropout),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False, conv_trans=conv_trans,
                                   kernel_size=conv_trans_kernel_size, dilation_n_hidden=dilation_n_hidden,
                                   conv_activation=conv_trans_activation, dropout=dropout),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out_dim, bias=True)

    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
