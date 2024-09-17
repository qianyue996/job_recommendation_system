import pandas as pd
import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        super().__init__()
        self.mode = mode
        # 读取训练文件
        df = pd.read_json('data_cleaning/tran.json', encoding='utf-8', lines=True)
        self.s1 = df['s1'].to_list()
        self.s2 = df['s2'].to_list()

    def __len__(self):
        return len(self.s1)

    def __getitem__(self, idx):
        if self.mode != 'train':
            return self.s2[idx]
        return self.s1[idx]
