import torch
import pandas as pd
from transformers import BertTokenizer

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, train, labels1, labels2):
        super().__init__()
        train_text = []
        train_labels1 = []
        train_labels2 = []
        target_labels1 = []
        target_labels2 = []
        # 读取训练文件
        with open(train, 'r', encoding='utf-8') as file:
            for data in file.readlines():
                label1, label2, text = data.strip().split('|')
                train_text.append(text)
                train_labels1.append(label1)
                train_labels2.append(label2)
        with open(labels1, 'r', encoding='utf-8') as file:
            for label in file.readlines():
                target_labels1.append(label.strip())
        with open(labels2, 'r', encoding='utf-8') as file:
            for label in file.readlines():
                target_labels2.append(label.strip())
        # 加载分词器
        self.tokenizer = BertTokenizer.from_pretrained('models/bert-base-multilingual-cased')
        # ...
        self.train_text = train_text
        self.train_labels1 = train_labels1
        self.train_labels2 = train_labels2
        self.target_labels1 = target_labels1
        self.target_labels2 = target_labels2

    def class_num(self):
        return [len(self.target_labels1), len(self.target_labels2)]

    def __len__(self):
        return len(self.train_text)

    def __getitem__(self, idx):
        text = self.train_text[idx]
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
            'attention_mask': attention_mask.to(torch.float)
        }

        poi_train_label1 = self.target_labels1.index(self.train_labels1[idx])
        train_label1 = torch.zeros(len(self.target_labels1))
        train_label1[poi_train_label1] = 1

        poi_train_label2 = self.target_labels2.index(self.train_labels2[idx])
        train_label2 = torch.zeros(len(self.target_labels2))
        train_label2[poi_train_label2] = 1

        labels = [train_label1, train_label2]

        return inputs, labels
