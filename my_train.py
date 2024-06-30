from my_dataset import MyDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import tqdm
import torch.nn as nn
from my_model import MyModel
from my_classes import get_classes
from my_dataset import prepare_data
import os
import time


if __name__ == '__main__':

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的设备是: {device}")

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    if not os.path.exists("data"):
        os.mkdir("data")

    # 第一次运行保证生成一共有多少类别的txt文件 -> data/train_type.txt
    get_classes(flag=True)
    # 第一次运行保证生成训练数据集 -> data/train.csv
    prepare_data(flag=True)
    datadir = 'data'
    # 加载数据集
    train_dataset = MyDataset(datadir, "train")
    # val_dataset = MyDataset(datadir, "val")
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)

    model = MyModel()
    model.to(device)
    loss_fc = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.005)
    epochs = 20

    for epoch in range(1, epochs+1):
        # 开始训练标志
        model.train()
        # 定义训练进度条
        bar = tqdm.tqdm(enumerate(train_dataloader), desc='Progress', unit='step', total=len(train_dataloader), ncols=80)
        bar.set_postfix(epoch=epoch)
        for i, j in bar:
            input_ids = j['input_ids'].to(device)
            attention_mask = j['attention_mask'].to(device)
            label = j['label'].to(device)
            pred = model(input_ids, attention_mask, label)
            loss = pred.loss
            # 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'loss: {loss}')

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
        # # 模型保存
        # if epoch % 4 == 0:
        #     model_name = f"epoch_{epoch}.pth"
        #     save_dir = os.path.join("checkpoints", model_name)
        #     torch.save(model, save_dir)
        #     print("\n")
        #     print(f"model{epoch} already save")
        #     print("\n")
        #     time.sleep(0.01)

