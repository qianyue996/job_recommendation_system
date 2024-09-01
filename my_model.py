import torch
import torch.nn as nn
from torch.nn.functional import pad
# torch.nn.functional.pad(t, (1, 1, 1, 1))# 左、右、上、下各填充‘’1‘’个0
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from my_dataset import MyDataset

class MyModel(nn.Module):
    def __init__(self, num_classes_level1, num_classes_level2):
        super().__init__()
        # 加载预训练的BERT模型
        print('''
------------------
正在加载预训练模型...
------------------
        ''')
        self.bert = BertForSequenceClassification.from_pretrained('models/bert-base-multilingual-cased')
        # 定义每个层级的分类器
        self.classifier_level1 = nn.Linear(self.bert.config.hidden_size, num_classes_level1)
        self.classifier_level2 = nn.Linear(self.bert.config.hidden_size, num_classes_level2)
        # self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        # 使用BERT模型提取特征
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取出 [CLS] token 的输出（即句子的特征表示）
        cls_output = bert_outputs.last_hidden_state[:, 0, :]
        # 分别对三个层级进行分类
        output_level1 = self.classifier_level1(cls_output)
        # dropout_level1 = self.dropout(output_level1)
        relu_level1 = self.relu(output_level1)
        
        output_level2 = self.classifier_level2(cls_output)
        # dropout_level2 = self.dropout(output_level2)
        relu_level2 = self.relu(output_level2)
        return relu_level1, relu_level2
