from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
# ==== 设置路径 ====
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from args import make_args
args = make_args()

class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = args.kernel_set
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class ST_Encoder(nn.Module):
    def __init__(self, config, dim_in, dim_out):
        super(ST_Encoder, self).__init__()
        self.feature_dim = dim_in

        self.input_window = args.K
        self.output_dim = args.latent_dim
        self.dropout = args.dropout

        self.dilation_exponential = config.dilation_exponential
        self.conv_channels = config.conv_channels
        self.residual_channels = config.residual_channels
        self.skip_channels = config.skip_channels
        self.layers = config.layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))

        self.skip_channels = config.skip_channels
        self.end_channels = config.end_channels
        self.output_window = args.K
        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels, kernel_size=(1, 1), bias=True)
        
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window, kernel_size=(1, 1), bias=True)

        kernel_size = args.kernel_size
        if self.dilation_exponential > 1:
            self.receptive_field = int(self.output_dim + (kernel_size-1) * (self.dilation_exponential**self.layers-1)
                                       / (self.dilation_exponential - 1))
        else:
            self.receptive_field = self.layers * (kernel_size-1) + self.output_dim

        for i in range(1):
            if self.dilation_exponential > 1:
                rf_size_i = int(1 + i * (kernel_size-1) * (self.dilation_exponential**self.layers-1)
                                / (self.dilation_exponential - 1))
            else:
                rf_size_i = i * self.layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, self.layers+1):
                if self.dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1) * (self.dilation_exponential**j - 1)
                                    / (self.dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(DilatedInception(self.residual_channels,
                                                          self.conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(DilatedInception(self.residual_channels,
                                                        self.conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                     out_channels=self.residual_channels, kernel_size=(1, 1)))
                if self.input_window > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
                                                     kernel_size=(1, self.input_window-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
                                                     kernel_size=(1, self.receptive_field-rf_size_j+1)))

                new_dilation *= self.dilation_exponential

        if self.input_window > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
                                   out_channels=self.skip_channels,
                                   kernel_size=(1, self.input_window), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels,
                                   out_channels=self.skip_channels,
                                   kernel_size=(1, self.input_window-self.receptive_field+1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
                                   out_channels=self.skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels,
                                   out_channels=self.skip_channels, kernel_size=(1, 1), bias=True)

        # self._logger.info('receptive_field: ' + str(self.receptive_field))

    def forward(self, input):
        # (batch_size, feature_dim, num_nodes(1), input_window)
        assert input.size(3) == self.input_window, 'input sequence length not equal to preset sequence length'

        if self.input_window < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field-self.input_window, 0, 0, 0))

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filters = self.filter_convs[i](x)
            filters = torch.tanh(filters)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filters * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        latent = self.end_conv_2(x)
        return latent     

# 预训练结合city id信息
class ST_Decoder(nn.Module):
    def __init__(self, latent_dim, vocab_size, city_num=3, city_emb_dim=16):
        super().__init__()
        self.city_emb = nn.Embedding(city_num, city_emb_dim)

        self.decoder = nn.Sequential(
            nn.LayerNorm(latent_dim + city_emb_dim),
            nn.Linear(latent_dim + city_emb_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, vocab_size)
        )

    def forward(self, latent, city_ids):  # latent: [B, T, 1, D]  city_ids: [B]
        latent = latent.squeeze(2)        # [B, T, D]
        city_emb = self.city_emb(city_ids)  # [B, city_emb_dim]
        fused = torch.cat([latent, city_emb], dim=-1)  # [B, T, D + city_emb_dim]
        return self.decoder(fused)  # [B, T, vocab_size]
