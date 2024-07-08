import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertForSequenceClassification
from my_dataset import MyDataset

class MyModel(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        with open('data/train_type.txt', 'r', encoding='utf-8') as f:
            name_typelist = [i.strip() for i in f.readlines()]
        self.bert = BertModel.from_pretrained("models/bert-base-multilingual-cased")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, data):
        inputs = data
        _, pooled_output = self.bert(**inputs, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

if __name__ == '__main__':
    # from transformers import BertTokenizer, BertModel
    # tokenizer = BertTokenizer.from_pretrained('models/bert-base-multilingual-cased')
    # model = BertModel.from_pretrained("models/bert-base-multilingual-cased")
    # text = "Replace me by any text you'd like."
    # encoded_input = tokenizer(text, return_tensors='pt')
    # output = model(**encoded_input)

    datadir = 'data'
    dataset = MyDataset(datadir, mode="train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = MyModel()
    # model = BertForSequenceClassification.from_pretrained('models/bert-base-multilingual-cased')

    for i, j in enumerate(dataloader):
        input_ids = j['input_ids']
        attention_mask = j['attention_mask']
        label = j['label']
        pred = model(input_ids, attention_mask, label)
        print(pred)
