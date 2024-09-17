import os
from typing import List, Dict
import json
import ollama


def read_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def get_result(text):
    client = ollama.Client(host='http://127.0.0.1:11434')
    response = client.generate(model='llama3.1',prompt=
    f"""
    根据以下招聘信息，编写一个描述个人技能和兴趣的“input”部分，并基于这些信息生成一个“output”部分，其中“output”是对个人适合工作的建议。请确保“input”和“output”都是完整的句子，并以JSON格式返回。

    招聘信息：{text}

    生成以下格式的数据，请直接填充JSON对象的内容，不要包含任何额外的说明或文本。确保所有生成的内容都与给定的文本直接相关，生成的是有效的JSON格式，并且内容高质量、准确、详细。
    {{
        "input": "生成的input内容",
        "output": "生成的output内容"
    }}
    """
    )
    return response


def generate_dataset(folder_path: str, entries_per_file: int = 2) -> List[Dict]:
    dataset = []
    total_chunk = len(os.listdir(folder_path))
    processed_chunk = 0
    tmp_num = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            processed_chunk += 1
            file_path = os.path.join(folder_path, filename)
            print(f"正在处理文件: {filename}，处理进度{processed_chunk}/{total_chunk}")
            text = read_file(file_path)
            for j in range(entries_per_file):
                try:
                    result = get_result(text)['response']
                    print(result)
                    dataset.append(result)
                    print(f"成功生成条目{j+1}， 累计生成了{len(dataset)}个条目")
                    if len(dataset) % 10 == 0:
                        # 如果输出目录不存在，则创建
                        if not os.path.exists('saveRawDataTmp'):
                            os.makedirs('saveRawDataTmp')
                        with open(f'saveRawDataTmp/tmp_raw_data{tmp_num}.txt', 'w', encoding='utf8') as f:
                            for i in dataset:
                                f.write(i+'\n')
                        tmp_num += 1
                except Exception as e:
                    print(f"生成条目时发生错误: {str(e)}")

    return dataset


def save_dataset_as_parquet(dataset: List[Dict], output_file: str):
    with open(output_file, 'w', encoding='utf8')as f:
        for i in dataset:
            f.write(i)
            f.write('\n')


if __name__ == "__main__":
    input_folder = "./saveChunk"  # 指定输入文件夹路径
    output_file = "raw_data.txt"

    print("开始生成数据集")
    dataset = generate_dataset(input_folder, entries_per_file=4)
    save_dataset_as_parquet(dataset, output_file)
    print(f"数据集已生成并保存到 {output_file}")
    print(f"共生成 {len(dataset)} 个有效条目")