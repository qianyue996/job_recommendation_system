import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer, BertModel

from my_loss import MyLoss as loss_fc
from my_model import MyModel

class MyDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        df = pd.read_excel('data/可视化.xlsx')
        # 删除含有空值的行，在原数据集上操作
        df.dropna(inplace=True)
        self.df_data = df['工作描述'].to_list()
        self.df_label1 = df['行业'].to_list()
        self.df_label2 = df['岗位名'].to_list()
        self.tokenizer = BertTokenizer.from_pretrained('models/bert-base-multilingual-cased')

        self.class_df_label1 = df.drop_duplicates(subset=['行业'], keep='first', inplace=False)['行业'].to_list()
        self.class_df_label2 = df.drop_duplicates(subset=['岗位名'], keep='first', inplace=False)['岗位名'].to_list()
    def class_num(self):
        return self.class_df_label1, self.class_df_label2

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        text = self.df_data[idx]
        encoder = self.tokenizer(text, truncation=True,
                    add_special_tokens=True,
                    return_token_type_ids=False,
                    return_attention_mask=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt')
        input_ids = encoder['input_ids'].squeeze()
        attention_mask = encoder['attention_mask'].squeeze()
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        poi_level1 = self.class_df_label1.index(self.df_label1[idx])
        label_level1 = torch.zeros(len(self.class_df_label1))
        label_level1[poi_level1] = 1

        poi_level2 = self.class_df_label2.index(self.df_label2[idx])
        label_level2 = torch.zeros(len(self.class_df_label2))
        label_level2[poi_level2] = 1
        labels = {
            'label_level1': label_level1,
            'label_level2': label_level2
        }
        return inputs, labels

if __name__ == '__main__':
    train_file = 'train.xlsx'
    label_file = 'data/_label.json'
    dataset = MyDataset(train_file)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

    num1, num2 = dataset.class_num()
    model = MyModel(len(num1), len(num2)).to('cuda')
    optimizer = AdamW(model.parameters(), lr=1e-4)
    epochs = 20
    model.train()
    for epoch in range(epochs):
        bar1 = tqdm.tqdm(enumerate(dataloader), desc='Progress', unit='step', total=len(dataloader))
        avg_loss = 0
        for i, data in bar1:
            input, labels = data
            input_ids = input['input_ids'].to('cuda')
            attention_mask = input['attention_mask'].to('cuda')
            outputs = model(input_ids, attention_mask)
            label1 = labels['label_level1'].to('cuda')
            label2 = labels['label_level2'].to('cuda')
            loss = loss_fc.forward(outputs, label1, label2)
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            avg_loss += loss
            bar1.set_postfix({
                'epoch' : f'{epoch}',
                'avg_loss' : f'{avg_loss.item()/(i+1):.3f}',
                'loss' : f'{loss.item():.3f}'
                })