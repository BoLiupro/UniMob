import random
from datasets import Dataset
import torch
import os
import sys
# ==== 设置路径 ====
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from args import make_args
args = make_args()

class CustomDataset(Dataset):
    def __init__(self, raw_data, vec):
        self.raw_data = raw_data # data from data.json
        self.vec = vec # input for encoder

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        vec_seq = []
        region_seq = []
        city_id_seq = []
        for id in idx:
            region_id=self.raw_data[id]['traj_region_id']
            region_seq.append(torch.tensor(region_id, dtype=torch.float))
            vec_seq.append(self.vec[id])
            city_id_seq.append(self.raw_data[id]['city_id'])
        return {
            "vec_seq": vec_seq,
            "region_seq": region_seq,
            "city_id_seq":city_id_seq
        }

