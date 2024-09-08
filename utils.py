import os

import ast
import pandas as pd

def init_data(file_path):
    # 创建文件夹
    mkdir()
    df = pd.read_excel(file_path)
    # 删除含有空值的行，在原数据集上操作
    df.dropna(inplace=True)

    # label1
    label1 = []
    hangye = df.drop_duplicates(subset=['行业'], keep='first', inplace=False)['行业'].to_list() # [list]
    with open('data/labels/labels_level1/labels.txt', 'w', encoding='utf-8') as file:
        for i in hangye:
            _class = df[df['行业'] == i]
            if len(_class) > 100:
                    file.write(_class['行业'].to_list()[0])
                    file.write('\n')
                    label1.append(_class['行业'].to_list()[0])

    # label2
    label2 = []
    with open('data/labels/labels_level2/labels.txt', 'w', encoding='utf-8') as file:
        for i in label1:
            gangwei = df[df['行业'] == i].drop_duplicates(subset=['岗位名'], keep='first', inplace=False)['岗位名'].to_list()
            for j in gangwei:
                label2.append(j)
        for i in list(set(label2)):
             file.write(i)
             file.write('\n')

    with open('data/train/train.txt', 'w', encoding='utf-8') as file:
        for i in label1:
            data = df[df['行业'] == i]
            gangwei = data['岗位名'].to_list()
            details = data['工作描述'].to_list()
            for j in range(len(gangwei)):
                file.write(i)
                file.write('|')
                file.write(gangwei[j])
                file.write('|')
                text = ast.literal_eval(details[j])
                for k in text:
                    file.write(k + ' ')
                file.write('\n')

def mkdir():
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

if __name__ == '__main__':
    init_data('train.xlsx')