# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics
import time
from torch.optim import AdamW 
from transformers import get_linear_schedule_with_warmup



def train(config, model, train_dataloader, dev_dataloader, device):

    #optim

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters,
                         lr=config.learning_rate)
    total_steps=len(train_dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = config.warm_up_ratio * total_steps, num_training_steps = total_steps)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))

        model.train()
        for i, input in enumerate(tqdm(train_dataloader)):

            for k,v in input.items():
                v=v.to(device)

            outputs = model(**input)
            loss=outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

            if total_batch % 100 == 2:
                # 每多少轮输出在训练集和验证集上的效果
                true = input['labels'].data.cpu()
                predic = torch.max(outputs.logits, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_dataloader,device)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)

                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc))
                model.train()
            total_batch += 1



def evaluate(config, model, data_loader,device='cpu'):
    model.eval()

    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for input in tqdm(data_loader):

            for k,v in input.items():
                v=v.to(device)
            
            outputs = model(**input)
            loss=outputs.loss
            loss_total += loss

            labels = input['labels'].data.cpu().numpy()
            predic = outputs.logits.argmax(dim=-1).cpu().numpy()

            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)

    if False:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_loader), report, confusion
        
    return acc, loss_total / len(data_loader)
