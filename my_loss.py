import torch

def my_loss(inputs, labels, task: str): # [logits1, logits2]
    if task == 'classifier_1': # inputs: [tensor] labels: [list]
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(inputs, labels[0])
    if task == 'classifier_2': # inputs: {'logits':[tensor],'probs':[list]} labels: [list]
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(inputs, labels[0])
    return loss