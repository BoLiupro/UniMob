import torch
import torch.nn as nn
import torch.nn.functional as F

class CityToGlobalAlignment(nn.Module):
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.aligner = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, latent):
        """
        latent: [B, T, 1, D]
        return: aligned latent -> [B, T, 1, D]
        """
        latent = latent.squeeze(2)  # [B, T, D]
        aligned = self.aligner(latent)  # [B, T, D]
        return aligned.unsqueeze(2)     # [B, T, 1, D]
    

class Projector(nn.Module):
    def __init__(self, latent_dim, vocab_size):
        super().__init__()

        self.aligner = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)  # 可继续加深但目前推荐保持3层
        )

        self.decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(0.3),  # 稍强些以防 decoder 过拟合
            nn.Linear(latent_dim, vocab_size)
        )

    def forward(self, latent):  # [B, T, 1, D]
        latent = latent.squeeze(2)            # [B, T, D]
        aligned = self.aligner(latent) + latent  # 残差连接
        return self.decoder(aligned)          # [B, T, vocab_size]


