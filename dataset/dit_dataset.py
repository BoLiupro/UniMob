from datasets import Dataset
import torch
import numpy as np

class DiTDataset(Dataset):
    def __init__(self, raw_data,vec):
        self.raw_data = raw_data # data from data.json
        self.vec = vec # input for encoder

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        vec_seq = []
        region_seq = []
        time_seq = []
        start_time_seq = []
        city_id_seq = []
        for id in idx:
            region_id=self.raw_data[id]['traj_region_id']
            region_seq.append(torch.tensor(region_id, dtype=torch.float))
            vec_seq.append(self.vec[id])
            time_feature = self.get_time_features([self.raw_data[id]['dow']], [self.raw_data[id]['tod']])
            time_seq.append(torch.tensor(time_feature, dtype=torch.float))
            start_time_seq.append(self.raw_data[id]['startTime'])
            city_id_seq.append(self.raw_data[id]['city_id'])
        return {
            "vec_seq": vec_seq,
            "region_seq": region_seq,
            "time_seq": time_seq,
            "index": idx,
            "startTime_seq": start_time_seq,
            "city_id_seq":city_id_seq
        }
    

    def get_time_features(self, dow_list, tod_list):
        """
        Args:
            dow_list: List[str], e.g. ['Monday', 'Tuesday', ...]
            tod_list: List[str], e.g. ['14:30', '23:45', ...]

        Returns:
            np.ndarray of shape [B, 4]
        """
        dow_map = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_to_index = {day: i for i, day in enumerate(dow_map)}

        features = []

        for dow_str, tod_str in zip(dow_list, tod_list):
            # Day of week
            dow_idx = dow_to_index[dow_str]
            dow_sin = np.sin(2 * np.pi * dow_idx / 7)
            dow_cos = np.cos(2 * np.pi * dow_idx / 7)

            # Time of day
            hour, minute = map(int, tod_str.split(":"))
            minutes = hour * 60 + minute
            tod_sin = np.sin(2 * np.pi * minutes / 1440)
            tod_cos = np.cos(2 * np.pi * minutes / 1440)
            features.append([dow_sin, dow_cos, tod_sin, tod_cos])
            
        return np.array(features)  # shape: [B, 4]
