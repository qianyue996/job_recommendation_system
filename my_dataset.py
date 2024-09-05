import torch
import pandas as pd

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer):
        super().__init__()
        print('''
---------------
正在加载数据集...
---------------
        ''')
        # 读取训练文件
        df = pd.read_excel(file_path)
        # 删除含有空值的行，在原数据集上操作
        df.dropna(inplace=True)
        self.df_data = df['工作描述'].to_list()
        self.df_labels_1 = df['行业'].to_list()
        self.df_labels_2 = df['岗位名'].to_list()
        self.tokenizer = tokenizer

        self.num_classes_1 = df.drop_duplicates(subset=['行业'], keep='first', inplace=False)['行业'].to_list()
        self.num_classes_2 = df.drop_duplicates(subset=['岗位名'], keep='first', inplace=False)['岗位名'].to_list()
    def class_num(self):
        return [len(self.num_classes_1), len(self.num_classes_2)]

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

        poi_labels_1 = self.num_classes_1.index(self.df_labels_1[idx])
        labels_1 = torch.zeros(len(self.num_classes_1))
        labels_1[poi_labels_1] = 1

        poi_labels_2 = self.num_classes_2.index(self.df_labels_2[idx])
        labels_2 = torch.zeros(len(self.num_classes_2))
        labels_2[poi_labels_2] = 1

        labels = [labels_1, labels_2]

        return inputs, labels
