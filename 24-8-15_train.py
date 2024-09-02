import os

import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from transformers import BertTokenizer, BertModel


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        super().__init__()
        print('''
---------------
正在加载数据集...
---------------
        ''')
        df = pd.read_excel(file_path)
        # 删除含有空值的行，在原数据集上操作
        df.dropna(inplace=True)
        self.df_data = df['工作描述'].to_list()
        self.df_labels = df['岗位名'].to_list()
        self.tokenizer = BertTokenizer.from_pretrained('models/bert-base-multilingual-cased')

        self.num_classes = df.drop_duplicates(subset=['岗位名'], keep='first', inplace=False)['岗位名'].to_list()
    def class_num(self):
        return len(self.num_classes)

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

        poi_labels = self.num_classes.index(self.df_labels[idx])
        labels = torch.zeros(len(self.num_classes))
        labels[poi_labels] = 1

        return inputs, labels

# 创建自定义的分类器
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
        self.classifier = torch.nn.Linear(bert_model.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

def train(model, train_dataloader, criterion, optimizer, writer, epoch):
    model.train()
    bar1 = tqdm.tqdm(enumerate(train_dataloader), desc='Progress', unit='step', total=len(train_dataloader))
    for i, data in bar1:
        # 从批次中提取输入数据和标签
        input, labels = data
        # 将数据添加到设备(cuda or cpu)
        input_ids = input['input_ids'].to('cuda')
        attention_mask = input['attention_mask'].to('cuda')
        labels = labels.to('cuda')
        # 清空优化器的梯度
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # 计算损失
        loss = criterion(outputs, labels.float())
        # 反向传播损失
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 写入log
        writer.add_scalar('loss', loss, i)
        # 更新tqdm进度条
        bar1.set_postfix({
            'epoch' : epoch,
            'loss' : loss.item(),
            'lr' : optimizer.state_dict()['param_groups'][0]['lr']
            })
    return model

def val(model, val_dataloader, criterion, writer, epoch):
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
            labels = labels.to('cuda')
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, labels.float())
            num_examples += len(input_ids)
            correct = (preds.argmax(dim=-1) == labels.argmax(dim=-1)).sum().item()
            total_correct += correct
            accuracy = total_correct / num_examples

            writer.add_scalar('acc', accuracy, i)

            total_loss += loss.item()
            avg_loss = total_loss / (i+1)
            val_bar.set_postfix(epoch=epoch, accuracy=accuracy, avg_loss=avg_loss)
    return accuracy

def model_save(model, accuracy, epoch):
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    model_name = f'epoch={epoch}_accuracy={accuracy}.pth'
    save_dir = os.path.join("checkpoints", model_name)
    torch.save(model, save_dir)

def main():
    train_file = 'train.xlsx'
    dataset = MyDataset(train_file)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=16, shuffle=True)

    num_classes = dataset.class_num()

    # 加载BERT模型，设置输出的类别数量
    bert_model = BertModel.from_pretrained('models/bert-base-multilingual-cased')
    model = CustomBertForSequenceClassification(bert_model, num_labels=num_classes).to('cuda')

    # model = MyModel(len(num1), len(num2)).to('cuda')
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    writer = SummaryWriter('logs')

    epochs = 20
    for epoch in range(epochs):
        model = train(model, train_dataloader, criterion, optimizer, writer, epoch)
        accuracy = val(model, val_dataloader, criterion, writer, epoch)
        if accuracy > 0.7:
            model_save(model, accuracy, epoch)
    writer.close()

if __name__ == '__main__':
    main()