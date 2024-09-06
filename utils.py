import os

import ast
import pandas as pd

def init_data(file_path):
    df = pd.read_excel(file_path)
    # 删除含有空值的行，在原数据集上操作
    df.dropna(inplace=True)
    df_data = df['工作描述'].to_list()
    df_labels_1 = df['行业'].to_list()
    df_labels_2 = df['岗位名'].to_list()
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
    with open('data/train/train.txt', 'w', encoding='utf-8') as file:
        for idx in range(len(df_data)):
            file.write(df_labels_1[idx])
            file.write('|')
            file.write(df_labels_2[idx])
            file.write('|')
            data = ast.literal_eval(df_data[idx])
            for i in data:
                file.write(i + ' ')
            file.write('\n')
    with open('data/labels/labels_level1/labels.txt', 'w', encoding='utf-8') as file:
        classes_1 = df.drop_duplicates(subset=['行业'], keep='first', inplace=False)['行业'].to_list()
        for data in classes_1:
            file.write(data)
            file.write('\n')
    with open('data/labels/labels_level2/labels.txt', 'w', encoding='utf-8') as file:
        classes_2 = df.drop_duplicates(subset=['岗位名'], keep='first', inplace=False)['岗位名'].to_list()
        for data in classes_2:
            file.write(data)
            file.write('\n')

if __name__ == '__main__':
    init_data('train.xlsx')