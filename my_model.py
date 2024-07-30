from mimetypes import init
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertForSequenceClassification
from my_dataset import MyDataset

class MyModel(nn.Module):
    def __init__(self, num_classes_level_1, num_classes_level_2):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier_level_1 = nn.Linear(self.bert.config.hidden_size, num_classes_level_1)
        self.classifier_level_2 = nn.Linear(self.bert.config.hidden_size, num_classes_level_2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits_level_1 = self.classifier_level_1(pooled_output)
        logits_level_2 = self.classifier_level_2(pooled_output)
        return logits_level_1, logits_level_2

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
