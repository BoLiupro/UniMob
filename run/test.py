import os
import sys
import random
import hashlib
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datetime import timedelta
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

# ==== 环境设置 ====
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append('/root/liubo/TravDiT')

from args import make_args
from util import load_raw_data, load_vec, temporal_pattern_one_hot_batch, generate_trajectory_prompt_temp
from dataset.dit_dataset import DiTDataset
from model.model_LLM import temporalTripModel, TrajectoryHead
from model.model_Encoder import ST_Encoder, ST_Decoder
from model.st_layers_config.args import parse_args
from model.dit import TravDit
from model.diffusion import TrajectoryDiffusion
from model.model_MLP import Mlp
from model.model_Projector import Projector

# ==== 参数和设备 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = make_args()
unique_poi_types = args.unique_poi_types

# ==== 加载数据 ====
dow_map = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
city_list = [args.city]
task = args.task  # 任务标识符
print(task)
tune_or_pretrain = 'tuning'
sample_size = 5000
random.seed(args.seed - 1)

all_data = []
for city_id, city in enumerate(city_list):
    raw_data = load_raw_data(city=city)
    vec = load_vec(city=city).to(device)
    vec = vec[:, 2:2 + args.poi_dim + args.pos_dim]
    indices = random.sample(range(len(raw_data)), sample_size)
    for i in indices:
        item = raw_data[i]
        item['city_id'] = city_id
        region_ids = item['traj_region_id']
        vec_seq = torch.stack([vec[int(rid)] for rid in region_ids])
        all_data.append((item, vec_seq))

random.shuffle(all_data)
all_raw_data, all_vec_seq = zip(*all_data)
full_dataset = DiTDataset(list(all_raw_data), list(all_vec_seq))
val_dataloader = DataLoader(full_dataset, batch_size=args.generator_batch_size, shuffle=False)

# ==== 加载模型 ====
base_model = "/root/liubo/TravDiT/checkpt_ETS/pretrain/llm"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, cache_dir="./models/")
tokenizer.pad_token = tokenizer.eos_token
head = TrajectoryHead()
head.load_state_dict(torch.load(f"{base_model}/trajectory_head.pt", map_location=device))
head = head.to(dtype=torch.float16, device=device)
llm = temporalTripModel.load_model(llm_path=base_model, trajectory_head=head)
llm.to(device).eval()

encoder_config = parse_args()
encoder = ST_Encoder(config=encoder_config, dim_in=args.poi_dim + args.pos_dim, dim_out=args.latent_dim)
encoder.load_state_dict(torch.load(f"/root/liubo/TravDiT/{args.checkpt_path}/tuning/encoder/encoder_2to1_gz_few.pth", map_location=device))
encoder.to(device).eval()

traj_diffusion = torch.load(f"/root/liubo/TravDiT/{args.checkpt_path}/tuning/diffusion/diffusion_2to1_gz_few.pkl",weights_only=False)
traj_diffusion.to(device).eval()
from model.model_MLP import Mlp
mlp = Mlp(latent_dim=args.latent_dim,num_layers=2)
mlp.load_state_dict(torch.load(f"/root/liubo/TravDiT/{args.checkpt_path}/tuning/diffusion/mlp_2to1_gz_few.pth", map_location=device))
mlp.to(device).eval()


cityproj = Projector(latent_dim=args.latent_dim, vocab_size=args.vocab_size)
cityproj.load_state_dict(torch.load(f"/root/liubo/TravDiT/{args.checkpt_path}/tuning/projector/projector_2to1_gz_few_5k.pth", map_location=device))
cityproj.to(device).eval()

# ==== POI condition 缓存 ====
poi_cache_dir = f"/root/liubo/TravDiT/cache/poi_condition_office_worker/"
os.makedirs(poi_cache_dir, exist_ok=True)

def hash_condition_key(city_name, tod, dow, depature_poi, traj_pattern):
    key_str = f"{city_name}_{tod}_{dow}_{','.join(map(str, depature_poi))}_{''.join(map(str, traj_pattern))}"
    return hashlib.md5(key_str.encode()).hexdigest()

def generate_poi_condition_with_LLM(
        city_name,dow, tod, depature_poi, unique_poi_types, tokenizer, model, args
        ):
    # 构造 prompt
    prompt = generate_trajectory_prompt_temp(
        tod=tod,
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

def build_poi_condition(vec_seq, start_time_seq, city_id_seq, traj_pattern_batch):
    poi_condition_batch = []
    for i in range(vec_seq.size(0)):
        city_name = city_list[city_id_seq[i].item()]
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
                    output = generate_poi_condition_with_LLM(city_name, tod, dow, sample_poi_condition[-1], unique_poi_types, tokenizer, llm, args)
                    poi_logits = output['poi_pred']
                    poi_pred = F.softmax(poi_logits, dim=-1).squeeze().detach().cpu().numpy()
                    sample_poi_condition.append(poi_pred)
                tod += timedelta(minutes=30)
            sample_poi_condition = np.array(sample_poi_condition)
            np.save(cache_file, sample_poi_condition)
        poi_condition_batch.append(sample_poi_condition)
    return torch.tensor(np.array(poi_condition_batch)).float().to(device)

def forward_and_loss(vec_seq, region_seq, start_time_seq,city_id_seq, mode ='train'):
    vec_seq = vec_seq.to(device)
    city_id_seq = city_id_seq.to(device)
    region_seq = region_seq.to(device)
    pos_seq = vec_seq[:, 0, -args.pos_dim:]
    latent = encoder(input=vec_seq.permute(0, 2, 1).unsqueeze(2)).squeeze(2)
    latent_std = torch.std(latent, dim=1, keepdim=True)
    latent_mean = torch.mean(latent, dim=1, keepdim=True)
    latent_norm = (latent - latent_mean) / (latent_std + 1e-8)  # 标准化 latent
    traj_pattern = temporal_pattern_one_hot_batch(region_seq)
    poi_condition = build_poi_condition(vec_seq, start_time_seq, city_id_seq, traj_pattern)

    if mode == 'train':
        with autocast():
            pred_x0 = traj_diffusion.generation_training(latent=latent_norm, home_locations=pos_seq, poi_condition=poi_condition, trav_pattern=traj_pattern.to(device))
    else:
        pred_x0 = traj_diffusion.TrajGenerating(
            num_samples=pos_seq.size(0),
            home_locations=pos_seq,
            poi_condition=poi_condition,
            trav_pattern=traj_pattern.to(device).to(dtype=torch.long)
        )
    latent_pred = pred_x0* (latent_std + 1e-8) + latent_mean
    latent_pred = mlp(latent_pred)
    logits = cityproj(latent_pred)
    y_true = region_seq
    y_pred = torch.argmax(logits, dim=-1)
    return y_true, y_pred

# ==== 验证流程 ====
y_true_list = []
y_pred_list = []
traj_diffusion.eval(); mlp.eval()

with torch.no_grad():
    for batch in tqdm(val_dataloader, desc="Validating"):
        y_true, y_pred = forward_and_loss(batch['vec_seq'], batch['region_seq'], batch['startTime_seq'], batch['city_id_seq'], mode='eval')
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

y_true = torch.cat(y_true_list, dim=0)
y_pred = torch.cat(y_pred_list, dim=0)
y_true_flat = y_true.view(-1).cpu().tolist()
y_pred_flat = y_pred.view(-1).cpu().tolist()

# 保存为 CSV（每行一个轨迹 step）
df = pd.DataFrame({
    'y_true': y_true_flat,
    'y_pred': y_pred_flat
})
df.to_csv(f"/root/liubo/TravDiT/results/{city}/results.csv", index=False)