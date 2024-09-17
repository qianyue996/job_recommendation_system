import torch
import pandas as pd
from transformers import AutoTokenizer, BertModel

def collate_fn(data):
    return token.encode_plus(data[0],
                            truncation=True,
                            padding=True,
                            max_length=500,
                            return_tensors='pt')


def get_feature(data):
    with torch.no_grad():
        feature = model(**data).last_hidden_state
    
    attention_mask = data['attention_mask']

    feature *= attention_mask.unsqueeze(dim=2)

    feature = feature.sum(dim=1)

    attention_mask = attention_mask.sum(dim=1, keepdim=True)

    feature /= attention_mask.clamp(min=1e-8)

    return feature


def build_features():
    features = []
    for i, data in enumerate(loader):
        for k in data.keys():
            data[k] = data[k].to('cuda')

        features.append(get_feature(data))

        if i % 50 == 0:
            print(i)
        
        # if i == 1000:
        #     break

    model.cpu()
    
    features = torch.cat(features)

    torch.save(features.cpu(), 'models/test.pt')


def test():
    features = torch.load('models/test.pt').to('cuda')

    correct = 0
    total = 0
    for i, data in enumerate(loader):
        for k in data.keys():
            data[k] = data[k].to('cuda')
            
        feature = get_feature(data)

        score = torch.nn.functional.cosine_similarity(feature, features, dim=1)

        argmax = score.argmax().item()

        if i == argmax:
            correct += 1
        total += 1

        if i % (50 * 8) == 0:
            print(i, correct / total)

        if i == 1000 * 8:
            break
    
    print(correct / total)


loader = torch.utils.data.DataLoader(dataset=MyDataset(), batch_size=8, collate_fn=collate_fn, shuffle=False, drop_last=False)
token = AutoTokenizer.from_pretrained('models/bert-base-multilingual-cased')
model = BertModel.from_pretrained('models/bert-base-multilingual-cased').to('cuda')
for param in model.parameters():
    param.requires_grad = False
model.eval()

# build_features()
test()
