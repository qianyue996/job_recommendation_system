from my_dataset import MyDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import tqdm
import torch.nn as nn
from my_model import MyModel
from my_dataset import prepare_data, get_classes
import os
import time

def train(epoch):
    model.train()
    epoch_bar = tqdm.tqdm(train_dataloader, desc='Progress', unit='step', total=len(train_dataset))
    for data in epoch_bar:
        optimizer.zero_grad()
        new_data = [data[i].to(device) for i in data]
        inputs = {'input_ids': new_data[0],
                    'attention_mask': new_data[1]
                    }
        label = new_data[2]
        output = model(inputs)
        loss = output.loss
        loss.backward()
        optimizer.step()
        epoch_bar.set_postfix(epoch=epoch, loss=loss)
    return model

def eval(epoch):
    model.eval()
    with torch.no_grad():
        eval_bar = tqdm.tqdm(eval_dataloader, desc='Progress', unit='step', total=len(eval_dataset))
        for i in eval_bar:
            input = i.to(device)
            output = model(i)

def save_model(epoch):
    model_name = f'epoch_{epoch}.pth'
    save_dir = os.path.join("checkpoints", model_name)
    torch.save(model, save_dir)

def main():
    for epoch in range(epochs):
        train(epoch)
        eval(epoch)

if __name__ == '__main__':
    flag = False
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    if not os.path.exists("data"):
        os.mkdir("data")
    
    # 第一次运行保证生成训练数据集 -> data/train.csv
    prepare_data(flag=flag)
    # 第一次运行保证生成一共有多少类别的txt文件 -> data/train_type.txt
    get_classes(flag=flag)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的设备是: {device}")

    datadir = 'data'
    train_dataset = MyDataset(datadir, "train")
    eval_dataset = MyDataset(datadir, "eval")
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=True)

    model = MyModel().to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    epochs = 20

    main()

        # # 模型验证评估标志
        # model.eval()
        # # 评估参数
        # num_examples = 0
        # total_correct = 0
        # total_loss = 0.
        # # 不需要计算梯度
        # with torch.no_grad():
        #     # 定义验证进度条
        #     bar2 = tqdm.tqdm(enumerate(val_dataloader), desc='Progress', unit='step', total=len(val_dataloader))
        #     for k, l in bar2:
        #         input_ids, labels = l
        #         input_ids = input_ids.to(device)
        #         labels = labels.to(device)

        #         preds = model(input_ids)
        #         loss = loss_fc(preds, labels)

        #         num_examples += len(input_ids)
        #         correct = (preds.argmax(dim=-1) == labels.argmax(dim=-1)).sum().item()
        #         total_correct += correct
        #         accuracy = total_correct / num_examples
        #         total_loss += loss.item()
        #         avg_loss = total_loss / num_examples
        #         bar2.set_postfix(epoch=epoch, accuracy=accuracy, avg_loss=avg_loss)

