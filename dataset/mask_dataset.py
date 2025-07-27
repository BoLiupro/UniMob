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

class MaskDataset(Dataset):
    def __init__(self, raw_data,num_region, mask_prob=args.mask_prob, mask_token_id=args.vocab_size):
        """
        raw_data: List[Dict[str,str]] where each dict contains a 'region' field with a list of region IDs
        vocab_size: total number of region tokens
        """
        self.raw_data = raw_data
        self.num_region = num_region
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        region_seq = []
        city_ids = []
        for id in idx:
            region_seq.extend(self.raw_data[id]['traj_region_id'])
            city_id = self.raw_data[id]['city_id']
            city_tensor = torch.full((args.K,), city_id, dtype=torch.long) 
            city_ids.extend(city_tensor)
        
        region_seq = torch.tensor(region_seq, dtype=torch.long)   # [K]
        mask = torch.rand(region_seq.shape) < self.mask_prob
        input_region = region_seq.clone()
        mask_region = region_seq.clone()
        input_region[mask] = self.mask_token_id                        # 替换为 [MASK] token
        mask_region[~mask] = args.vocab_size                 

        # MLM只会预测并计算region_id是mask_token_id的loss
        return {
            "region_id": input_region.clone(), # 其中10%是mask_token_id,其他正常,[B,K,1]
            "labels": mask_region.clone(), # 其中10%是原始的region_id,其他是mask_token_id,[B,K]
            "city_id": torch.tensor(city_ids, dtype=torch.long) # 添加城市ID
        }
