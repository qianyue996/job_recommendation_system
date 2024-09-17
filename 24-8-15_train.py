import os

import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from my_model import *
from my_loss import *
from my_dataset import *

def train(train_dataloader):
    writer = mkdir_logs_dir()

    features = []
    for i, data in enumerate(train_dataloader):
        for k in data.keys():
            data[k] = data[k].to('cuda')

        features.append(get_feature(data))

        if i % 50 == 0:
            print(i)
        
        # if i == 1000:
        #     break

    model.cpu()
    
    features = torch.cat(features)

    torch.save(features.cpu(), 'models/test.pt')

    writer.close()


def model_save(model, accuracy, epoch):
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    model_name = f'epoch={epoch}_accuracy={accuracy}.pth'
    save_dir = os.path.join("checkpoints", model_name)
    torch.save(model, save_dir)

def mkdir_logs_dir():
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

    return SummaryWriter(f'{new_folder_path}')


def collate_fn(data):
    return token.encode_plus(data,
                            truncation=True,
                            padding=True,
                            max_length=500,
                            return_tensors='pt')


def get_feature(data):
    with torch.no_grad():
        feature = model(data).last_hidden_state
    
    attention_mask = data['attention_mask']

    feature *= attention_mask.unsqueeze(dim=2)

    feature = feature.sum(dim=1)

    attention_mask = attention_mask.sum(dim=1, keepdim=True)

    feature /= attention_mask.clamp(min=1e-8)

    return feature


def dev(val_dataloader):
    features = torch.load('models/test.pt').to('cpu')

    correct = 0
    total = 0
    for i, data in enumerate(val_dataloader):
        for k in data.keys():
            data[k] = data[k].to('cpu')
            
        feature = get_feature(data)

        score = torch.nn.functional.cosine_similarity(feature, features, dim=1)

        argmax = score.argmax().item()

        if i == argmax:
            correct += 1
        total += 1

        if i % 50 == 0:
            print(i, correct / total)

        if i == 1000 * 8:
            break
    
    print(correct / total)


def test():
    dataset = MyDataset(mode='test')

    features = torch.load('models/test.pt').to('cpu')

    data = input('请输入：')
    data = collate_fn(data)

    feature = get_feature(data)

    score = torch.nn.functional.cosine_similarity(feature, features, dim=1)

    argmax = score.argmax().item()

    print(f'得分为：{score.argmax()}')
    print(dataset[argmax])


def main():
    global token
    global model

    dataset = MyDataset()
    # 加载模型
    token = AutoTokenizer.from_pretrained('models/bert-base-multilingual-cased')
    bert_model = BertModel.from_pretrained("models/bert-base-multilingual-cased")
    model = CustomBertModel(bert_model).to('cpu')

    # 数据集加载器
    train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=8, collate_fn=collate_fn, shuffle=False, drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=8, collate_fn=collate_fn, shuffle=False, drop_last=False)

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # train(train_dataloader)
    # dev(val_dataloader)
    test()

if __name__ == '__main__':
    main()