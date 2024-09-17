import json
import re


def remove_symbols(text):
    # 去除符号
    symbols = [',', '\"', '\'', ':', '-', '{', '}', '[', ']', '\n', " "]
    for symbol in symbols:
        text = text.replace(symbol, '')
    return text


def data_cleaning():
    with open(input_file, 'r', encoding='utf8')as f:
        text = f.read()
        text = remove_symbols(text)
        # 正则表达式模式，用于匹配从 "input" 到下一个 "output" 之间的内容
        pattern = r'input(.*?)input'
        
        # 使用 re.DOTALL 使得 '.' 特殊字符匹配包括换行符在内的任意字符
        matches = re.findall(pattern, text, re.DOTALL)
        
        # 提取并返回匹配的内容列表
        texts = [match.strip() for match in matches]

    template = []
    for text in texts:
        text = text.split('output')
        input = text[0]
        output = text[1]

        conversation = {
                'input': input,
                'output': output
            }
        template.append(conversation)
    return json.dumps(template, ensure_ascii=False, indent=2)
    

def main():
    json_data = json.loads(data_cleaning())
    # 待填入的模板
    template = []

    for idx, data in enumerate(json_data):
        conversation = [
            {
                "from": "user",
                "value": data["input"]
            },
            {
                "from": "assistant",
                "value": data["output"]
            }
        ]
        template.append(
            {
                "id": f"identity_{idx}",
                "conversations": conversation
            })

    # 输出填充数据后的模板
    print(json.dumps(template, ensure_ascii=False, indent=2))


    # 将template写入到本地文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print(f'\n数据集长度为{len(template)}')
    print(f"处理好的数据已写入到本地文件: {output_file}")


if __name__ == '__main__':
    input_file = 'raw_data.txt'
    output_file = 'train.json'
    main()