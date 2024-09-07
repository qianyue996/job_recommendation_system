import torch

def my_loss(inputs, labels, task: str): # [logits1, logits2]
    if task == 'classifier_1':
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(inputs, labels)
        pass
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_per_level = [] # [loss_level_1, loss_level_2]
    for i in range(len(inputs)):
        loss = loss_fn(inputs[i], labels[i])
        loss_per_level.append(loss)
    return 