from collections import defaultdict
import json

def nested_dict():
    return defaultdict(nested_dict)

def build_tree(label_list):
    tree = nested_dict()
    for path in label_list:
        current_level = tree
        for part in path:
            current_level = current_level[part]
    return tree

labels = [
    ["互联网/AI", "后端开发", "Java"],
    ["互联网/AI", "后端开发", "PHP"],
    ["互联网/AI", "后端开发", "Python"],
    ["互联网/AI", "前端/移动开发", "Android"],
    ["互联网/AI", "前端/移动开发", "iOS"],
    ["互联网/AI", "前端/移动开发", "前端开发工程师"],
    ["互联网/AI", "测试", "测试工程师"],
    ["互联网/AI", "测试", "软件测试"],
    ["互联网/AI", "测试", "自动化测试"],
    ["电子/电气/通信", "电子/硬件开发", "电子工程师"],
    ["电子/电气/通信", "电子/硬件开发", "硬件工程师"],
    ["电子/电气/通信", "电子/硬件开发", "嵌入式软件工程师"],
    ["电子/电气/通信", "半导体/芯片", "集成电路IC设计"],
    ["电子/电气/通信", "半导体/芯片", "数字IC验证工程师"],
    ["电子/电气/通信", "半导体/芯片", "模拟IC设计工程师"],
    ["电子/电气/通信", "电气/自动化", "电气工程师"],
    ["电子/电气/通信", "电气/自动化", "自动化"],
    ["电子/电气/通信", "电气/自动化", "机电工程师"],
    ["产品", "产品经理", "产品经理"],
    ["产品", "产品经理", "电商产品经理"],
    ["产品", "产品经理", "AI产品经理"],
    ["产品", "游戏策划/制作", "游戏策划"],
    ["产品", "游戏策划/制作", "系统策划"],
    ["产品", "游戏策划/制作", "游戏制作人"],
    ["产品", "用户研究", "用户研究员"],
    ["产品", "用户研究", "用户研究经理"],
    ["产品", "用户研究", "用户研究总监"]
]

tree = build_tree(labels)

def convert_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_to_dict(v) for k, v in d.items()}
    return d

tree_dict = convert_to_dict(tree)

with open('data/_label.json', 'w', encoding='utf-8') as file:
    json.dump(tree_dict, file, ensure_ascii=False, indent=2)
