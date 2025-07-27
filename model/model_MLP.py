import torch.nn as nn
import torch

class Mlp(nn.Module):
    def __init__(self, latent_dim, hidden_dim=None, num_layers=1, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = latent_dim * 2

        layers = []
        for i in range(num_layers):
            layers.append(nn.Sequential(
                nn.LayerNorm(latent_dim if i == 0 else hidden_dim),
                nn.Linear(latent_dim if i == 0 else hidden_dim, hidden_dim),
                nn.GELU()
            ))

        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, latent_dim)


    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = layer(x)
        return residual + self.output(x)  # 残差连接 + 输出对齐



