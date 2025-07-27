import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class TravDit(torch.nn.Module):
    def __init__(self, num_location, location_embedding,input_len=48,num_head=2,TrajGenerator_Translayers=1,poi_dim=14,pos_dim=4,seed=2024):
        super(TravDit, self).__init__()
        self.seed=seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.num_location = num_location
        self.input_len=input_len
        self.loc_size=location_embedding
        self.num_head = num_head
        self.temporal_embed = TimeEmbedding(location_embedding) 
        
        self.pos_encoding1 = PositionalEncoding(num_features=location_embedding, dropout=0.1, max_len=self.input_len*2)
        self.pos_encoding2 = PositionalEncoding(num_features=location_embedding, dropout=0.1, max_len=self.input_len*2)
        self.pos_encoding3 = PositionalEncoding(num_features=location_embedding, dropout=0.1, max_len=self.input_len*2)

        self.start_embed = torch.nn.Embedding(num_embeddings=2, embedding_dim=location_embedding)

        self.distance_cross_attention = nn.MultiheadAttention(embed_dim=location_embedding, num_heads=self.num_head, dropout=0.1)


        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.3)

        self.home_embedding_layer=nn.Sequential(
                                nn.LayerNorm(4),
                                nn.Linear(4, location_embedding),  # 坐标2维 → D
                                # nn.GELU()
                            )
        self.poi_projector = nn.Sequential(
            nn.LayerNorm(poi_dim),  # P=14
            nn.Linear(poi_dim, location_embedding),  # P → D
            # nn.GELU()
        )
        self.controlnet = nn.Embedding(input_len + 1,location_embedding)

        self.spatial_transformer = nn.Sequential(*[TransformerBlock(d_model=location_embedding, nhead=self.num_head,
                                                                                                 dim_feedforward=4*location_embedding, dropout=0.1) for _ in range(TrajGenerator_Translayers)])

    def TrajGenerator(self, xt, time, home_conditions, poi_condition,trav_pattern):
        t = self.temporal_embed(time)
        
        et = self.pos_encoding1(xt)

        home_embedding=self.home_embedding_layer(home_conditions)

        poi_condition = self.poi_projector(poi_condition)
        poi_condition = self.pos_encoding3(poi_condition)

        temporal_condition1 = self.controlnet(trav_pattern)
        temporal_condition1 = self.pos_encoding2(temporal_condition1)

        for block in self.spatial_transformer:
            et = block(et,temporal_condition1,home_embedding,poi_condition,t)

        same_class = (trav_pattern.unsqueeze(2) == trav_pattern.unsqueeze(1)).float()

        et_expanded = et.unsqueeze(1)  # 扩展pred_x0以匹配same_class的形状 [512, 1, 24, 64]
        same_class_expanded = same_class.unsqueeze(3)  # 扩展same_class以匹配pred_x0的特征维度 [512, 24, 24, 1]

        same_class_sum = same_class_expanded.sum(dim=1, keepdim=True)  # 计算每个类别的元素数量 [512, 1, 24, 1]

        et_same_class_avg = (et_expanded * same_class_expanded).sum(dim=2) / same_class_sum.squeeze(1)  # 计算平均值 [512, 24, 64]

        # 将每个样本的pred_x0替换为其对应类别的平均值
        et = et_same_class_avg
        return et

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(torch.nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.dim=4
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb


class PositionalEncoding(nn.Module):
    def __init__(self, num_features, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_features))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_features, 2, dtype=torch.float32) / num_features)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class AdaLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(AdaLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape

    def forward(self, x, gamma, beta):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_normalized = (x - mean) / (std + self.eps)
        return gamma * x_normalized + beta

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.pos_encoding1 = PositionalEncoding(d_model, dropout)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
        
        # Encoder-decoder attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = AdaLayerNorm(d_model)
        self.norm2 = AdaLayerNorm(d_model)

        self.norm3 = nn.LayerNorm(d_model)


        self.d_model=d_model
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

        self.adaLN_modulation1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 3 * d_model, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation1[1].weight)
        nn.init.zeros_(self.adaLN_modulation1[1].bias)

        self.adaLN_modulation2 = nn.Sequential(
             nn.SiLU(),
             nn.Linear(d_model, 3 * d_model, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation2[1].weight)
        nn.init.zeros_(self.adaLN_modulation2[1].bias)

    def forward(self, tgt, memory,cls_condition,poi_condition,time_condition, tgt_mask=None, memory_mask=None):

        affine_params_cls = self.adaLN_modulation1(time_condition.unsqueeze(1)+poi_condition+cls_condition.unsqueeze(1)+memory)
        gamma_1, beta_1, alpha_1 = torch.split(affine_params_cls, self.d_model, dim=-1)

        affine_params_mem = self.adaLN_modulation2(time_condition.unsqueeze(1)+poi_condition+cls_condition.unsqueeze(1)+memory)
        gamma_3, beta_3, alpha_3 = torch.split(affine_params_mem, self.d_model, dim=-1)

        tgt1 = self.norm1(tgt,gamma_1,beta_1)
        tgt1 = self.self_attn(tgt1, tgt1, tgt1, attn_mask=tgt_mask)[0]
        tgt = tgt + tgt1*alpha_1

        tgt1 = self.norm3(tgt)
        tgt1 = self.cross_attn(tgt1,memory,memory,attn_mask=memory_mask)[0]
        tgt = tgt + tgt1

        tgt1 = self.norm2(tgt,gamma_3,beta_3)
        tgt1 = self.dropout(self.linear2(F.relu(self.linear1(tgt1))))
        tgt = tgt +tgt1*alpha_3

        return tgt
