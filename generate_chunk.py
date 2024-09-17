import torch
from transformers import BertTokenizer, BertModel
import os
import pandas as pd


def process_text(df):
    data = []
    df_columns = df.columns.to_list()
    for i in range(len(df)):
        detail = f"""
公司名称：{df[df_columns[2]][i]}
公司规模：{df[df_columns[3]][i]}
公司类型：{df[df_columns[4]][i]}
工作类型：{df[df_columns[5]][i]}
岗位名称：{df[df_columns[6]][i]}
学历要求：{df[df_columns[7]][i]}
薪资情况：{df[df_columns[8]][i]}
招聘人数：{df[df_columns[9]][i]}
待遇福利：{df[df_columns[10]][i]}
公司地点：{df[df_columns[11]][i]}
工作经验：{df[df_columns[12]][i]}
业务范围：{df[df_columns[13]][i]}
工作地点：{df[df_columns[14]][i]}
详细信息：{df[df_columns[15]][i]}
"""
        data.append(detail)
        print(f"已格式化完成第 {i + 1} 个文本块")
    return data

def save_chunks_to_files(chunks, output_dir):
    """
    将分割后的文本块保存到文件

    参数:
    chunks (list): 文本块列表
    output_dir (str): 输出目录路径
    """
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将每个文本块保存为单独的文件
    for i, chunk in enumerate(chunks):
        chunk_file_path = os.path.join(output_dir, f"chunk_{i + 1}.txt")
        with open(chunk_file_path, 'w', encoding='utf-8') as file:
            file.write(chunk)
        print(f"已保存第 {i + 1} 个文本块到 {chunk_file_path}")




if __name__ == '__main__':
    # 设置输入和输出路径
    input_file_path = '某招聘平台.xlsx'  # 替换为你的长文本文件路径
    output_dir = './saveChunk/'  # 替换为你希望保存文本块的目录路径

    # 读取长文本
    print("加载raw格式数据集中...")
    df = pd.read_excel(input_file_path)
    print("数据集加载完毕，开始处理raw数据")
    print('\n')
    # 处理文本
    text_chunks = process_text(df)
    # 保存分割后的文本块到指定目录
    save_chunks_to_files(text_chunks, output_dir)