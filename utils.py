import os
import pandas as pd

def init_data(file_path):
    # train.txt labels.txt
    df = pd.read_excel(file_path)
    # 删除含有空值的行，在原数据集上操作
    df.dropna(inplace=True)
    df_data = df['工作描述'].to_list()
    df_labels_1 = df['岗位名'].to_list()
    df_labels_2 = df['行业'].to_list()
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/train"):
        os.mkdir("data/train")
    if not os.path.exists("data/labels"):
        os.mkdir("data/labels")
    if not os.path.exists("data/labels/labels_level1"):
        os.mkdir("data/labels/labels_level1")
    if not os.path.exists("data/labels/labels_level2"):
        os.mkdir("data/labels/labels_level2")
    with open('data/train/train.txt') as file:
        for line in file.readlines:
            pass