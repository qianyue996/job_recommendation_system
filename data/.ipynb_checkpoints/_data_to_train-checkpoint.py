import json

def process_file(input_file, output_file):
    # 读取文本文件
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 初始化数据列表
    data = []

    # 遍历每行文本，将当前行作为描述
    i = 0
    while i < len(lines):
        description = lines[i].strip()  # 当前行作为描述
        i += 1
        labels = []
        
        # 收集所有后续行作为标签，直到遇到下一条描述或文件结尾
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("描述"):
            labels.append(lines[i].strip())
            i += 1
        
        # 处理当前描述和标签
        data.append({"data": description, "label": labels})

    # 将数据保存为 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"数据集已保存为 '{output_file}'")

# 设置输入和输出文件名
input_file = 'data/_test.txt'
output_file = 'data/_train.json'

# 调用处理函数
process_file(input_file, output_file)
