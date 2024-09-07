import os

import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from my_model import *
from my_loss import *
from my_dataset import *

def train(model, train_dataloader, val_dataloader, optimizer):
    # 初始化日志可视化Tensorboard, 按文件夹依次排序分类exp1, exp2...
    # 启动：tensorboard --logdir='logs'
    if not os.path.exists("logs"):
        os.mkdir("logs")
    existing_folders = [f for f in os.listdir('logs') if os.path.isdir(os.path.join('logs', f)) and f.startswith('exp')]
    # 找到最大的数字后缀
    max_num = 0
    for folder in existing_folders:
        try:
            # 提取数字部分
            num = int(folder[len('exp'):])
            if num > max_num:
                max_num = num
        except ValueError:
            pass  # 跳过无法转换为整数的部分
    # 新的文件夹名称
    new_folder_name = f"{'exp'}{max_num + 1}"
    # 创建新的文件夹
    new_folder_path = os.path.join('logs', new_folder_name)
    os.makedirs(new_folder_path)
    writer = SummaryWriter(f'{new_folder_path}')
    # 训练轮数
    epochs = 20
    total_train_step = 0
    total_eval_step = 0
    for epoch in range(epochs):
        model.train()
        # 进度条显示
        train_bar = tqdm.tqdm(train_dataloader, desc='Progress', unit='step', total=len(train_dataloader))
        for data in train_bar:
            total_train_step += 1
            # 从批次中提取输入数据和标签
            inputs, labels = data
            # 将数据添加到设备(cuda or cpu)
            input_ids = inputs['input_ids'].to('cuda')
            attention_mask = inputs['attention_mask'].to('cuda')
            labels = [i.to('cuda') for i in labels]
            # 清空优化器的梯度
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, task='classifier_1')
            # 计算损失
            loss = my_loss(outputs, labels[0])
            # 反向传播损失
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 写入log
            writer.add_scalar('train/loss', round(loss.detach().item(), 3), total_train_step)
            # 更新tqdm进度条
            train_bar.set_postfix({
                'epoch' : epoch,
                'loss' : round(loss.detach().item(), 3)
                })
        # 验证阶段
        model.eval()
        num_examples = 0
        total_correct = 0
        total_loss = 0.
        with torch.no_grad():
            val_bar = tqdm.tqdm(val_dataloader, desc='Progress', unit='step', total=len(val_dataloader))
            for data in val_bar:
                total_eval_step += 1
                inputs, labels = data
                input_ids = inputs['input_ids'].to('cuda')
                attention_mask = inputs['attention_mask'].to('cuda')
                labels = [i.to('cuda') for i in labels]
                outputs = model(input_ids, attention_mask)
                loss = my_loss(outputs, labels)

                num_examples += len(input_ids)
                # 正确率计算
                for i in range(len(input_ids)):
                    if outputs[0][i].argmax(dim=-1) == labels[0][i].argmax(dim=-1):
                        if outputs[1][i].argmax(dim=-1) == labels[1][i].argmax(dim=-1):
                            total_correct += 1
                accuracy = round(total_correct / num_examples, 3)
                # 写入日志
                writer.add_scalar('accuracy/batch', accuracy, total_eval_step)
                total_loss += round(loss.detach().item(), 3)
                avg_loss = total_loss / (total_eval_step+1)
                val_bar.set_postfix(epoch=epoch, accuracy=accuracy, avg_loss=avg_loss)
        model_save(model, accuracy, epoch)
    writer.close()

def model_save(model, accuracy, epoch):
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    model_name = f'epoch={epoch}_accuracy={accuracy}.pth'
    save_dir = os.path.join("checkpoints", model_name)
    torch.save(model, save_dir)

def main():
    # 数据集路径
    train_data = 'data/train/train.txt'
    labels1 = 'data/labels/labels_level1/labels.txt'
    labels2 = 'data/labels/labels_level2/labels.txt'
    # 加载数据集
    dataset = MyDataset(train_data, labels1, labels2)
    # 加载模型
    num_classes = dataset.class_num()
    bert_model = BertModel.from_pretrained("models/bert-base-multilingual-cased")
    model = CustomBertModel(bert_model, num_classes[0]).to('cuda')

    # 划分数据集与验证集
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 数据集加载器
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=16, shuffle=True)

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train(model, train_dataloader, val_dataloader, optimizer)

if __name__ == '__main__':
    main()

# TODO 先训练第一层的模型，然后再根据第一层预测的的概率传递给第二层的分类器，根据概率去设计损失函数的惩罚结果，
# 在这之前要根据第一级类别数量，去逐个归纳在第二层里面的子分类各有多少，于是就形成了层次多分类模型