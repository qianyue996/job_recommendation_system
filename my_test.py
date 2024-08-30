import torch
from transformers import AutoTokenizer

if __name__ == '__main__':
    Use_GPU = True
    classes_path = 'data/classes.txt'
    CLASSES = []
    with open(classes_path, 'r', encoding='utf-8') as f:
        for i in f.readlines():
            CLASSES.append(i.strip())

    model_path = 'checkpoints/epoch_20.pth'
    model = torch.load(model_path)

    tokenizer = AutoTokenizer.from_pretrained("models/bert-base-chinese")
    x = input("输入你的兴趣特征: ")
    x = tokenizer(x, truncation=True, max_length=500, padding='max_length', return_tensors='pt', return_length=True)

    x1 = x['input_ids']
    x2 = x['attention_mask']
    if Use_GPU:
        model.to('cuda')
        x1 = x1.to('cuda')
        x2 = x2.to('cuda')
    with torch.no_grad():
        output = model(x1)
        output = int(output.argmax(dim=-1))
        class_ = CLASSES[output]
        print(output)
        print(class_)
