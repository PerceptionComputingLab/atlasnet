import os

import numpy as np
from torch.utils.data import Dataset
import pickle


class CETUS(Dataset):

    def __init__(self, dataroot='./dataset_pkl/', split='train'):
        # 初始化
        self.target_file = dataroot + split + '/'
        self.files = os.listdir(self.target_file)
        self.data_len = len(self.files)

    def __len__(self):
        # 返回数据集的大小
        return self.data_len

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意

        file_current = self.files[index]
        f = open(self.target_file + file_current, 'rb')
        data = pickle.load(f)
        for key in data:
            data[key] = data[key][np.newaxis, ...]
        return data


