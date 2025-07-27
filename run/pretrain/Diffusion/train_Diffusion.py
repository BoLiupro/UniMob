import os,sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append('/root/liubo/TravDiT')  # 添加项目根目录到 Python 路径
from args import make_args
from util import load_raw_data,load_vec,temporal_pattern_one_hot_batch,generate_trajectory_prompt_temp
import random
from tqdm import tqdm 
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = make_args()
unique_poi_types = args.unique_poi_types
task = args.task # 任务标识符，用于保存模型

from dataset.dit_dataset import DiTDataset
from torch.utils.data import DataLoader, random_split
import random
import torch

all_data = []  # 存储 (raw_data, vec) 对
dow_map = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
city_list = ['Changsha','Guangzhou','Shenzhen']  # 城市列表
sample_size = 500  # 每个城市采样
random.seed(args.seed)  # 设置随机种子

for city_id, city in enumerate(city_list):
    raw_data = load_raw_data(city=city)
    vec = load_vec(city=city).to(device)
    vec = vec[:, 2:2 + args.poi_dim + args.pos_dim]  # [num_region, feat_dim]
    indices = random.sample(range(len(raw_data)), sample_size)
    for i in indices:
        item = raw_data[i]
        item['city_id'] = city_id
        region_ids = item['traj_region_id']  # [K]
        vec_seq = torch.stack([vec[int(rid)] for rid in region_ids])  # [K, feat_dim]
        all_data.append((item, vec_seq))

# 打乱
random.shuffle(all_data)
all_raw_data, all_vec_seq = zip(*all_data)
all_raw_data = list(all_raw_data)
all_vec_seq = list(all_vec_seq)

# 构造数据集
full_dataset = DiTDataset(all_raw_data, all_vec_seq)

# 拆分
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 加载器
train_dataloader = DataLoader(train_dataset, batch_size=args.generator_batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.generator_batch_size, shuffle=False)

import torch
from transformers import AutoTokenizer

base_model  = "/root/liubo/TravDiT/checkpt_ETS/pretrain/llm"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True,cache_dir = "./models/")
tokenizer.pad_token = tokenizer.eos_token
from model.model_LLM import temporalTripModel, TrajectoryHead

# 构建 head 模块并加载参数
head = TrajectoryHead()
state_dict = torch.load(f"{base_model}/trajectory_head.pt", map_location="cuda")
head.load_state_dict(state_dict)
head = head.to(dtype=torch.float16, device="cuda")
# 构建完整模型
llm = temporalTripModel.load_model(llm_path=base_model, trajectory_head=head)
llm.to(device)
llm.eval()
# 加载encoder
from model.model_Encoder import ST_Encoder
from model.st_layers_config.args import parse_args

encoder_config = parse_args()
encoder = ST_Encoder(config = encoder_config,dim_in = args.poi_dim+args.pos_dim, dim_out = args.latent_dim)
encoder.load_state_dict(torch.load(f"/root/liubo/TravDiT/{args.checkpt_path}/tuning/encoder/encoder_{task}.pth", map_location=device))
encoder.to(device)
encoder.eval()

from model.model_Encoder import ST_Decoder
decoder = ST_Decoder(latent_dim = args.latent_dim, vocab_size = args.vocab_size,city_emb_dim=16)
decoder.load_state_dict(torch.load(f"/root/liubo/TravDiT/{args.checkpt_path}/tuning/encoder/decoder_{task}.pth", map_location=device))
decoder.to(device)
decoder.eval()
from model.dit import TravDit
from model.diffusion import TrajectoryDiffusion

trajgenerator = TravDit(num_location=args.vocab_size, location_embedding=args.latent_dim,input_len=args.K,poi_dim=args.poi_dim,pos_dim=args.pos_dim,seed=args.seed).to(device)
trajgenerator = trajgenerator.type(torch.float32)

traj_diffusion = TrajectoryDiffusion(model=trajgenerator,linear_start=0.0005, linear_end=0.05,full_n_steps=1000,device=device)
traj_diffusion = traj_diffusion.type(torch.float32)
from model.model_MLP import Mlp
mlp = Mlp(latent_dim=args.latent_dim,num_layers=2).to(device)

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.optim import AdamW

scaler = GradScaler()
optimizer = AdamW(list(trajgenerator.parameters()) + list(mlp.parameters()), lr=1e-2, weight_decay=0.01)
total_steps = len(train_dataloader)*args.generator_epoch
warmup_steps = int(total_steps * 0.05)  # 10% of total steps for warmup

from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

best_acc = 0.0
patience = 5
patience_counter = 0
from datetime import timedelta

def generate_poi_condition_with_LLM(
        city_name,dow, tod, depature_poi, unique_poi_types, tokenizer, model, args
        ):
    # 构造 prompt
    prompt = generate_trajectory_prompt_temp(
        tod=tod.strftime("%H:%M"),
        dow=dow,
        departure_poi_dist=depature_poi,
        city=city_name,
        poi_category_list=unique_poi_types
    )

    tokenized = tokenizer(prompt, max_length=256, truncation=True,
                            padding="max_length", return_tensors="pt")
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    with torch.no_grad():
        outputs = model(**tokenized)

    return  outputs # shape: [K, poi_dim]

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import timedelta
# kl_div = torch.nn.KLDivLoss(reduction='batchmean')
mse_loss = torch.nn.MSELoss()


dow_map = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

import hashlib
import os

poi_cache_dir = f"/root/liubo/TravDiT/cache/poi_condition/"
os.makedirs(poi_cache_dir, exist_ok=True)

def hash_condition_key(city_name, tod, dow, depature_poi, traj_pattern):
    """生成唯一key用于缓存"""
    key_str = f"{city_name}_{tod}_{dow}_{','.join(map(str, depature_poi))}_{''.join(map(str, traj_pattern))}"
    return hashlib.md5(key_str.encode()).hexdigest()

def build_poi_condition(vec_seq, start_time_seq, city_id_seq, traj_pattern_batch):
    poi_condition_batch = []

    for i in range(vec_seq.size(0)):
        city_id = city_id_seq[i].item()
        city_name = city_list[city_id]
        traj_pattern = traj_pattern_batch[i].tolist()
        start_time = pd.to_datetime(start_time_seq[i])
        tod = start_time.replace(hour=0, minute=0)
        dow = dow_map[start_time.dayofweek]
        depature_poi = vec_seq[i, 0, 2:2 + args.poi_dim].detach().cpu().numpy()

        cache_key = hash_condition_key(city_name, tod.strftime("%H:%M"), dow, depature_poi, traj_pattern)
        cache_file = os.path.join(poi_cache_dir, f"{cache_key}.npy")

        if os.path.exists(cache_file):
            sample_poi_condition = np.load(cache_file)
        else:
            sample_poi_condition = [depature_poi]
            for j in range(len(traj_pattern) - 1):
                if traj_pattern[j + 1] == traj_pattern[j]:
                    sample_poi_condition.append(sample_poi_condition[-1])
                else:
                    output = generate_poi_condition_with_LLM(
                        city_name=city_name, tod=tod, dow=dow,
                        depature_poi=sample_poi_condition[-1],
                        unique_poi_types=args.unique_poi_types,
                        tokenizer=tokenizer, model=llm, args=args
                    )
                    poi_logits = output['poi_pred']
                    poi_pred = F.softmax(poi_logits, dim=-1).squeeze().detach().cpu().numpy()
                    sample_poi_condition.append(poi_pred)
                tod += timedelta(minutes=30)
            sample_poi_condition = np.array(sample_poi_condition)
            np.save(cache_file, sample_poi_condition)

        poi_condition_batch.append(sample_poi_condition)

    poi_condition_tensor = torch.from_numpy(np.array(poi_condition_batch)).float().to(device)
    return poi_condition_tensor


def forward_and_loss(vec_seq, region_seq, start_time_seq, city_id_seq, mode='train'):
    vec_seq = vec_seq.to(device)
    city_id_seq = city_id_seq.to(device)
    region_seq = region_seq.to(device)
    pos_seq = vec_seq[:, 0, -args.pos_dim:]
    latent = encoder(input=vec_seq.permute(0, 2, 1).unsqueeze(2)).squeeze(2)  # [B, K, latent_dim]
    traj_pattern = temporal_pattern_one_hot_batch(region_seq)
    poi_condition = build_poi_condition(vec_seq, start_time_seq, city_id_seq, traj_pattern)

    with autocast(enabled=(mode == 'train')):
            latent_std = torch.std(latent, dim=1, keepdim=True)
            latent_mean = torch.mean(latent, dim=1, keepdim=True)
            # latent_mean = 0
            # latent_std = 4
            # print(latent_std,latent_mean)
            latent_norm = (latent - latent_mean) / (latent_std + 1e-8)  # 标准化 latent

            if mode == 'train':
                # === 训练时用 standard diffusion 训练流程 ===
                pred_x0 = traj_diffusion.generation_training(
                    latent=latent_norm,
                    home_locations=pos_seq,
                    poi_condition=poi_condition,
                    trav_pattern=traj_pattern.to(device).to(dtype=torch.long)
                )
            else:
                pred_x0 = traj_diffusion.TrajGenerating(
                    num_samples=latent_norm.size(0),
                    home_locations=pos_seq,
                    poi_condition=poi_condition,
                    trav_pattern=traj_pattern.to(device).to(dtype=torch.long)
                )
            latent_pred = pred_x0* (latent_std + 1e-8) + latent_mean  # 恢复标准化

            latent_pred = mlp(latent_pred)  # shape: [B, D]

            loss = mse_loss(latent_pred, latent.detach())

            city_id_seq_exp = city_id_seq.unsqueeze(1).expand(-1, args.K)
            # === optional: 精度分析，用 argmax 比较（仅用于监控）===
            y_true = torch.argmax(decoder(latent,city_id_seq_exp), dim=-1)
            y_pred = torch.argmax(decoder(latent_pred,city_id_seq_exp), dim=-1)
            correct = (y_true == y_pred).float().sum().item()
            total = y_true.numel()

    return loss, correct, total

# ==== Training loop ====
for epoch in range(args.generator_epoch):
    trajgenerator.train(); traj_diffusion.train(); mlp.train()
    train_loss, train_correct, train_total = [], 0, 0

    tqdm_bar = tqdm(train_dataloader, desc=f"[Epoch {epoch}] Training")
    for batch in tqdm_bar:
        loss, correct, total = forward_and_loss(
            batch['vec_seq'], batch['region_seq'],
            batch['startTime_seq'], batch['city_id_seq'],
            mode='train'
        )
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss.append(loss.item())
        train_correct += correct
        train_total += total
        tqdm_bar.set_postfix(loss=loss.item(), acc=correct / total)

    avg_train_loss = np.mean(train_loss)
    avg_train_acc = train_correct / train_total
    print(f"[Epoch {epoch}] ✅ Train Loss: {avg_train_loss:.4f} | Accuracy: {avg_train_acc:.4f}")

    # ==== Validation ====
    trajgenerator.eval(); traj_diffusion.eval(); mlp.eval()
    val_loss, val_correct, val_total = [], 0, 0

    with torch.no_grad():
        tqdm_bar = tqdm(val_dataloader, desc=f"[Epoch {epoch}] Validating")
        for batch in tqdm_bar:
            loss, correct, total = forward_and_loss(
                batch['vec_seq'], batch['region_seq'],
                batch['startTime_seq'], batch['city_id_seq'],
                mode='eval'
            )
            val_loss.append(loss.item())
            val_correct += correct
            val_total += total
            tqdm_bar.set_postfix(loss=loss.item(), acc=correct / total)

    avg_val_loss = np.mean(val_loss)
    avg_val_acc = val_correct / val_total
    print(f"[Epoch {epoch}] ✅ Val Loss: {avg_val_loss:.4f} | Accuracy: {avg_val_acc:.4f}")

    # ==== Early Stopping ====
    if avg_val_acc > best_acc:
        best_acc = avg_val_acc
        patience_counter = 0
        print(f"🎯 Accuracy improved to {avg_val_acc:.4f}")
    else:
        patience_counter += 1
        print(f"⚠️ No improvement. Early stop counter = {patience_counter}/{patience}")
        if patience_counter >= patience:
            print(f"⛔ Early stopping at epoch {epoch}")
            break
        
os.makedirs(f"/root/liubo/TravDiT/{args.checkpt_path}/tuning/diffusion", exist_ok=True)
torch.save(traj_diffusion, f"/root/liubo/TravDiT/{args.checkpt_path}/tuning/diffusion/diffusion_{task}.pkl")
torch.save(mlp.state_dict(), f"/root/liubo/TravDiT/{args.checkpt_path}/tuning/diffusion/mlp_{task}.pth")