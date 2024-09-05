import torch

def my_loss(inputs, labels): # [logits1, logits2]
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_per_level = [] # [loss_level_1, loss_level_2]
    for i in range(len(inputs)):
        loss = loss_fn(inputs[i], labels[i])
        loss_per_level.append(loss)
    return loss_per_level[0] + 0.5 * loss_per_level[1]