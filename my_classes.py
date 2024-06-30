import pandas as pd

def get_classes(flag=False):
    if flag:
        df = pd.read_csv('data/train.csv', encoding='utf_8_sig')
        name_list = df['0'].to_list()
        with open('data/train_type.txt', 'w', encoding='utf-8') as f:
                for i in set(name_list):
                    f.write(i+'\n')
get_classes()