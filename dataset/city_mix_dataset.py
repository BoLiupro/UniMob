import torch
from torch.utils.data import Dataset
import random

class CityMixDataset(Dataset):
    def __init__(self, city_list, sample_size, poi_dim, pos_dim, device, load_raw_data, load_vec):
        """
        Args:
            city_list: ['Changsha', 'Guangzhou', 'Shenzhen']
            sample_size: 每个城市采样的数据量
            poi_dim: POI 维度
            pos_dim: 位置维度
            device: torch 设备
            load_raw_data: 函数，加载原始轨迹数据
            load_vec: 函数，加载轨迹特征（POI + 位置）
        """
        self.raw_data = []
        self.vec_list = []

        for city_id, city in enumerate(city_list):
            city_raw = load_raw_data(city)
            city_vec = load_vec(city).to(device)
            city_vec = city_vec[:, 2:2+poi_dim+pos_dim]  # 按需截取特征

            indices = random.sample(range(len(city_raw)), sample_size)
            for i in indices:
                item = city_raw[i]
                item['city_id'] = city_id
                self.raw_data.append(item)
                self.vec_list.append(city_vec[i])  # 单条特征向量

        # 随机打乱样本
        combined = list(zip(self.raw_data, self.vec_list))
        random.shuffle(combined)
        self.raw_data, self.vec_list = zip(*combined)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return {
            'raw': self.raw_data[idx],
            'vec': self.vec_list[idx]
        }
