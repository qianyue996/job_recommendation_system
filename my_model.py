import torch
from transformers import BertModel, BertForSequenceClassification

# 创建自定义的模型
class CustomBertForSequenceClassification(torch.nn.Module):
    def __init__(self, bert_model, num_labels):
        super().__init__()
        # 加载预训练的BERT模型
        print('''
------------------
正在加载预训练模型...
------------------
        ''')
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(bert_model.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        dropout = self.dropout(cls_output)
        logits = self.classifier(dropout)
        return logits

class HierarchicalClassifier(torch.nn.Module):
    def __init__(self, num_classes_per_level):
        super().__init__()
        # 加载共享的BERT模型
        self.bert = BertModel.from_pretrained("models/bert-base-multilingual-cased")
        # 为每个层次创建一个分类器
        self.classifiers = torch.nn.ModuleList([torch.nn.Linear(self.bert.config.hidden_size, num_classes) for num_classes in num_classes_per_level])

    def forward(self, input_ids, attention_mask):
        # BERT模型生成文本表示
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        # 对每一层进行分类
        logits_per_level = []
        for classifier in self.classifiers:
            logits = classifier(pooler_output)
            logits_per_level.append(logits)
        
        return logits_per_level  # 返回每层分类的结果[logits1, logits2]
    
class CustomBertModel(torch.nn.Module): # bert_model = BertModel(), num_classes
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        # 不训练，不需要计算梯度
        for param in self.bert.parameters():
            param.requires_grad_(False)
    def forward(self, data):
        return self.bert(**data)
