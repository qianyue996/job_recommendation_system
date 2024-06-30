import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertForSequenceClassification
from my_dataset import MyDataset

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        with open('data/train_type.txt', 'r', encoding='utf-8') as f:
            name_typelist = [i.strip() for i in f.readlines()]
        self.bert = BertForSequenceClassification.from_pretrained("models/bert-base-multilingual-cased", num_labels=len(name_typelist))

    def forward(self, input_ids, attention_mask, label):
        input = {'input_ids': input_ids,
                 'attention_mask': attention_mask,
                 'labels': label}
        pred = self.bert(**input)
        return pred

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
