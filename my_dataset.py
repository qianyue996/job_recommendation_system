import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder

def prepare_data(flag=False):
    datapath='myspider/train.txt'
    if flag:
        data_list = []
        with open(datapath, 'r', encoding='utf-8') as f:
            txt_data = f.readlines()
        for i in txt_data:
            i = i.strip().split('__')
            data_list.append(i)
            print(f'写入数据中...: {i}\n')
        df = pd.DataFrame(data_list)
        df.to_csv('data/train.csv', encoding='utf_8_sig')

def get_classes(flag=False):
    datapath='data/train.csv'
    if flag:
        df = pd.read_csv(datapath, encoding='utf_8_sig')
        name_list = df['0'].to_list()
        with open('data/train_type.txt', 'w', encoding='utf-8') as f:
                for i in set(name_list):
                    f.write(i+'\n')

class MyDataset(Dataset):
    def __init__(self, datadir, mode="train"):
        super().__init__()
        self.datadir = datadir
        df = pd.read_csv("data/train.csv", encoding='utf_8_sig')

        # 标签编码
        label_big = df['0'].to_list()
        label_middle = df['1'].to_list()
        label_small = df['2'].to_list()
        label_specific = df['3'].to_list()

        encoder_level_1 = LabelEncoder()
        encoder_level_2 = LabelEncoder()
        encoder_level_3 = LabelEncoder()
        encoder_level_4 = LabelEncoder()

        encoded_categories_level_1 = encoder_level_1.fit_transform(label_big)
        encoded_categories_level_2 = encoder_level_2.fit_transform(label_middle)
        encoded_categories_level_3 = encoder_level_3.fit_transform(label_small)
        encoded_categories_level_4 = encoder_level_4.fit_transform(label_specific)

        # 构造标签张量
        labels_level_1 = torch.tensor(encoded_categories_level_1)
        labels_level_2 = torch.tensor(encoded_categories_level_2)
        labels_level_3 = torch.tensor(encoded_categories_level_3)
        labels_level_4 = torch.tensor(encoded_categories_level_4)

        self.inputs = []
        self.tokenizer = BertTokenizer.from_pretrained('models/bert-base-multilingual-cased')
        data_file = os.path.join(self.datadir, f'{mode}.csv')
        
        name_list = df['0'].to_list()
        with open('data/train_type.txt', 'r', encoding='utf-8') as f:
            self.name_typelist = [i.strip() for i in f.readlines()]
        for i in range(len(df)):
            area = df['1'].to_list()[i]
            name = df['2'].to_list()[i]
            detail = df['3'].to_list()[i]
            self.inputs.append(f'地点是{area}, 公司名字是{name}, 工作要求是{detail}')
        self.labels = name_list
        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.inputs[idx]

        encoder = self.tokenizer(text, truncation=True, add_special_tokens=True, return_token_type_ids=False,
                                    return_attention_mask=True, max_length=512, padding='max_length', return_tensors='pt')

        return {
            'input_ids':  encoder['input_ids'].squeeze(),
            'attention_mask': encoder['attention_mask'].squeeze()
        }

if __name__ == '__main__':
    # 第一次运行把下面的flag打开，保证生成data/train.csv
    prepare_data(datapath='myspider/train.txt', flag=False)
    # datadir = 'data'
    # dataset = MyDataset(datadir, mode="train")
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # for i, j in enumerate(dataloader):
    #     input_ids, label = j
    #     input('press enter to contiune...')
