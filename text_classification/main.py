from transformers import AutoModelForSequenceClassification,AutoTokenizer
from data_loader import SequenceClassificationDataset,collate_fn,collate_fn_test
from torch.utils.data import DataLoader
from train import train

import numpy as np
import argparse
import torch

def set_args():

    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--seed', type=int, default=2022,help='random seed')
    parser.add_argument('--pretrained_path', type=str, default='./pretrained_model/bert-base-chinese',help='choose a model_path')
    parser.add_argument('--train_path', type=str, default='./data/train.txt')
    parser.add_argument('--dev_path', type=str, default='./data/dev.txt')
    parser.add_argument('--learning_rate', type=float, default=0.00003,help='random seed')
    parser.add_argument('--num_epochs', type=int, default=6,help='random seed')
    parser.add_argument('--save_path', type=str, default='./output/model.pth',help='random seed')
    parser.add_argument('--warm_up_ratio', type=float, default=0.03,help='warmup rate')

    args = parser.parse_args()
    return args

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():

    args=set_args()
    set_seed(args.seed)
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    #data
    train_dataset=SequenceClassificationDataset(args.train_path)
    dev_dataset=SequenceClassificationDataset(args.dev_path)

    tokenizer=AutoTokenizer.from_pretrained(args.pretrained_path)

    train_dataloader=DataLoader(train_dataset,batch_size=32,shuffle=True,collate_fn=lambda x:collate_fn(x,tokenizer))
    dev_dataloader=DataLoader(dev_dataset,batch_size=32,shuffle=False,collate_fn=lambda x:collate_fn(x,tokenizer))

    #model
    model=AutoModelForSequenceClassification.from_pretrained(args.pretrained_path,num_labels=10)
    model=model.to(device)

    #train and eval
    train(args,model,train_dataloader,dev_dataloader,device)



if __name__=='__main__':
    main()
