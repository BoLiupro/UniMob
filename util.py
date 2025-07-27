import colorsys
import math
import folium
import branca.colormap as cm
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import geopandas as gpd
from tqdm import tqdm
import torch
import json
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from args import make_args

args = make_args()

# ==== 工具函数 ====
def generate_trajectory_prompt(tod, dow, departure_poi_dist, poi_category_list):
    prompt = (
        f"Given the POI category distribution of the departure location (a list of percentages "
        f"corresponding to the categories below), and the departure time, please predict the POI "
        f"category distribution of the next destination."
        f"The output should be a list of percentages summing to 1, where each element corresponds "
        f"to the same POI category as in the input list.\n\n"
        f"- Departure time: {tod} on {dow}\n"
        f"- Departure POI distribution: {departure_poi_dist}\n"
        f"- POI categories (fixed order): {poi_category_list}\n\n"
        f"Predict the most likely POI distribution of the next location the vehicle may reach, "
        f"considering temporal behavior and spatial function correlation."
    )
    return prompt

def generate_trajectory_prompt_temp(tod, dow, departure_poi_dist, poi_category_list, city, topk=14):
    """
    Generate a simple prompt to predict:
        - stay time at departure
        - destination POI distribution
    """
    # 取 top-k 非零 POI 类别及比例
    poi_with_scores = sorted(
        zip(poi_category_list, departure_poi_dist),
        key=lambda x: x[1],
        reverse=True
    )
    top_pois = [(n, s) for n, s in poi_with_scores[:topk] if s > 0]
    poi_mixed_str = ", ".join([f"{name}: {round(float(score), 3)}" for name, score in top_pois])

    prompt = (
        "<Instruction>\n"
        f"You are given a private car trip with an office worker driver that starts in {city}.\n"
        "Based on the POI distribution of the starting location and the departure time, "
        "predict:\n"
        "1. Predict the time interval (in 30-minute units) between the previous arrival and the upcoming arrival;\n"
        "2. The POI distribution of the destination location (over 14 categories).\n\n"
        "<Input>\n"
        f"- Departure time: {tod} on {dow}\n"
        f"- POI distribution at departure location (ordered): {poi_mixed_str}"
    )
    return prompt


# 提取并扁平化标签向量
def flatten_label(label_list):
    vec = []
    for dest in label_list:
        vec.extend([dest[k] for k in args.unique_poi_types])
    return vec

def position_embedding(lat, lon, pos_dim):
    """
    输入：
        latitudes: Tensor of shape [B, T]  (纬度，范围大约 [-90, 90])
        longitudes: Tensor of shape [B, T] (经度，范围大约 [-180, 180])
        embed_dim: 输出维度（必须是偶数）

    输出：
        embedding: Tensor of shape [B, T, embed_dim]
    """
    assert pos_dim % 2 == 0, "Embedding dimension must be even."

    dim_half = pos_dim // 2
    div_term = torch.exp(
        torch.arange(0, dim_half, dtype=torch.float32) * (-math.log(10000.0) / dim_half)
    )
    lat_embed = lat * div_term
    lon_embed = lon * div_term
    lat_sin = torch.sin(lat_embed)
    lat_cos = torch.cos(lat_embed)
    lon_sin = torch.sin(lon_embed)
    lon_cos = torch.cos(lon_embed)
    embedding = torch.cat([lat_sin, lat_cos, lon_sin, lon_cos], dim=-1)  # [B, T, 2×dim_half]
    return embedding  # [B, T, embed_dim]

def load_vec(city):
    with open(f'/root/liubo/TravDiT/data/grid/{city}/voronoi_clipped.geojson') as f:
        centers = json.load(f)

    poi_distribution = pd.read_csv(f"/root/liubo/TravDiT/data/poi/{args.city}_poi.csv").values
    coords = [center['properties']['center'] for center in centers['features']]
    lats = np.array([c[0] for c in coords])
    lons = np.array([c[1] for c in coords])

    # ✅ 自动计算 min-max 范围
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    vec = []
    for idx, (lat, lon) in enumerate(zip(lats, lons)):
        # ✅ min-max normalize to [0, 1]
        lat_norm = (lat - lat_min) / (lat_max - lat_min + 1e-8)
        lon_norm = (lon - lon_min) / (lon_max - lon_min + 1e-8)

        # ✅ 使用 sin/cos 编码，提升平滑性和周期性表达
        lat_enc = [math.sin(2 * math.pi * lat_norm), math.cos(2 * math.pi * lat_norm)]
        lon_enc = [math.sin(2 * math.pi * lon_norm), math.cos(2 * math.pi * lon_norm)]
        pos_vec = lat_enc + lon_enc  # [4-dim]

        poi_vec = poi_distribution[idx]
        full_vec = np.concatenate([poi_vec, pos_vec])  # [poi_dim + 4]
        vec.append(full_vec)

    vec = torch.tensor(vec, dtype=torch.float32)
    print(f"[INFO] Generated spatial POI+pos vec: {vec.shape}")
    return vec


def load_raw_data(city):
    if args.equal_time_slot:
        with open(f'/root/liubo/TravDiT/data/trip/{city}/{city}_data_K48_ETS.json') as f:
            raw_data = json.load(f)
    else:
        with open(f"data/trip/{args.city}/{args.city}_data.json") as f:
            raw_data = json.load(f)
    return raw_data

def save_dit_checkpoint(model, optimizer, scaler, epoch, val_loss, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),  # AMP 使用时必须保存
        "epoch": epoch,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, path)
    print(f"✅ Saved checkpoint to {path}")

# 主要函数
def load_dit_for_inference(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ Loaded model for inference from {path}")

def select_region(probs,k):
    # 确保probs第二维等于轨迹长度，第三维等于vocab size
    assert probs.dim() == 3, "probs should be a 3D tensor"
    # assert probs.size(0) == args.travDiT_batch_size, "probs second dimension should match batch_size"
    assert probs.size(1) == args.K, "probs second dimension should match traj_len"
    assert probs.size(2) == args.vocab_size, "probs third dimension should match vocab_size"
    probs = torch.softmax(probs, dim=-1)  # 确保概率分布
    # 选出最大的k个概率对应的索引
    idx1 = torch.topk(probs, k, dim=-1).indices  # shape: [B, K, k]
    return idx1

def print_metrics(y_true,y_pred):
        # 将 list 中的 Tensor 拼接为一个大 tensor
        y_true = torch.cat(y_true,dim=0)
        # y_pred:[B,K,k],多个可能的预测结果

        y_pred = torch.cat(y_pred, dim=0)  # shape: [total_B, K]
        # 转成 numpy 并 flatten
        y_true = y_true.cpu().numpy().flatten()
        y_pred = y_pred.cpu().numpy().flatten()
        # 报告各项指标
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Precision (macro):", precision_score(y_true, y_pred, average='macro'))
        print("Recall (macro):", recall_score(y_true, y_pred, average='macro'))
        print("F1-score (macro):", f1_score(y_true, y_pred, average='macro'))
        # 更详细的报告（含每一类）
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=3))
        # 保存y_true和y_pred到csv文件
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        df.to_csv(f"/root/liubo/TravDiT/results/{args.city}/result.csv", index=False)
        print(f"✅ Saved y_true and y_pred to results/{args.city}/result.csv")

from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
# Warmup schedule (e.g., 10% warmup of total steps)
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

def find_closest_cell(poi_dis, vec, poi_dim):
    """
    poi_dis: torch.Tensor of shape [1, P] or [P], predicted POI distribution
    vec: numpy array, shape [N, >= P+2], POI 信息向量表
    """
    if isinstance(poi_dis, torch.Tensor):
        poi_dis = poi_dis.squeeze().detach().cpu().numpy()  # [P]
    # 截取每一行的 POI 向量
    poi_vectors = vec[:, 2:2 + poi_dim]
    # 计算每一行的欧氏距离
    distances = np.linalg.norm(poi_vectors - poi_dis, axis=1)
    return np.argmin(distances)  # 返回最相近的 index

def find_closest_cell_batch(poi_pred, vec, poi_dim):
    pred_norm = poi_pred / np.linalg.norm(poi_pred, axis=-1, keepdims=True)
    vec_poi = vec[:, 2:2+poi_dim]
    vec_norm = vec_poi / np.linalg.norm(vec_poi, axis=-1, keepdims=True)
    sim = np.dot(pred_norm, vec_norm.T)  # [B, N]
    return np.argmax(sim, axis=-1)

def temporal_pattern_one_hot(traj):
    """
    traj变成one hot 编码
    """
    unique_ids = {}
    mapped = []
    current_id = 0
    for region_id in traj:
        if region_id not in unique_ids:
            unique_ids[region_id] = current_id
            current_id += 1
        mapped.append(unique_ids[region_id])
    return mapped

def temporal_pattern_one_hot_batch(trajs: torch.Tensor) -> torch.Tensor:
    """
    将 [B, K] 的 region_id 序列转为 [B, K] 的局部 ID 编码，每条轨迹内做 label encoding。
    """
    batch_size, seq_len = trajs.shape
    output = torch.zeros_like(trajs)

    for i in range(batch_size):
        traj = trajs[i].tolist()
        unique_ids = {}
        current_id = 0
        mapped = []
        for region_id in traj:
            if region_id not in unique_ids:
                unique_ids[region_id] = current_id
                current_id += 1
            mapped.append(unique_ids[region_id])
        output[i] = torch.tensor(mapped, device=trajs.device)

    return output

