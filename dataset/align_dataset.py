import random
from datasets import Dataset
import torch

class AlignDataset(Dataset):
    def __init__(self, raw_data,vec):
        self.raw_data = raw_data # data from data.json
        self.vec = vec # input for encoder

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        vec_seq = []
        text_seq = []
        for id in idx:
            region_seq=self.raw_data[id]['region']
            vec_seq.append(self.vec[region_seq])
            text = self.raw_data[id]['text']
            text_seq.append(text)
        return {
            "vec_seq": vec_seq,
            "text_seq": text_seq,
            "index": idx
        }
