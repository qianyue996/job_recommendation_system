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
        print('''
---------------
正在加载数据集...
---------------
        ''')
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
    train_size = int(len(dataset) * 0.1)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=True)

    num1, num2 = dataset.class_num()
    model = MyModel(len(num1), len(num2)).to('cuda')
    optimizer = AdamW(model.parameters(), lr=5e-4)
    epochs = 20
    model.train()
    for epoch in range(epochs):
        bar1 = tqdm.tqdm(enumerate(train_dataloader), desc='Progress', unit='step', total=len(train_dataloader))
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
            bar1.set_postfix({
                'epoch' : f'{epoch}',
                'loss' : f'{loss.item():.3f}',
                })
        # 验证阶段
        model.eval()
        # 评估参数
        num_examples = 0
        total_correct = 0
        total_loss = 0.
        with torch.no_grad():
            val_bar = tqdm.tqdm(enumerate(val_dataloader), desc='Progress', unit='step', total=len(val_dataloader))
            for i, data in val_bar:
                input, labels = data
                input_ids = input['input_ids'].to('cuda')
                attention_mask = input['attention_mask'].to('cuda')
                label1 = labels['label_level1'].to('cuda')
                label2 = labels['label_level2'].to('cuda')
                preds = model(input_ids, attention_mask)

                num_examples += len(input_ids)
                for i in range(len(preds)):
                    level1_correct = (preds[0].argmax(dim=-1) == label1.argmax(dim=-1)).sum().item()
                    if level1_correct == 0:
                        correct = 0
                    else:
                        level2_correct = (preds[1].argmax(dim=-1) == label2.argmax(dim=-1)).sum().item()
                        correct = level1_correct + level2_correct
                total_correct += correct
                accuracy = total_correct / num_examples
                total_loss += loss.item()
                avg_loss = total_loss / num_examples
                val_bar.set_postfix(epoch=epoch, accuracy=accuracy, avg_loss=avg_loss)