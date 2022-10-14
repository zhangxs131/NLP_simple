from transformers import AutoTokenizer
import torch
import os
from torch.utils.data import Dataset,DataLoader

class SequenceClassificationDataset(Dataset):

    def __init__(self,data_path,mode='train'):
        super().__init__()
        self.mode=mode
        self.data=self._load_file(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        return self.data[i]

    def _load_file(self,file_name):

        if type(file_name)!=str:
            if self.mode=='test':
                return [{'text':i} for i in file_name]

        with open(file_name,'r',encoding='utf-8') as f:
            content=f.readlines()
        data=[]
        if self.mode!='test':
            for i in content:
                t=i.strip('\n')
                t=t.split('\t')
                data.append(
                    {'text':t[0],
                     'label':int(t[1]) 
                    } 
                )#这里label已经换成数字，如果需要str转换，需要简单修改。)
        else:
            for i in content:
                t=i.strip('\n')
                r=t.split('\t')
                if len(r)==1:
                    data.append({'text':t})
                else:
                    data.append({'text':r[0]})
            
        return data

#collate_fn=lambda x: collate_fn(x, info)
def collate_fn(batch,tokenizer):
    input=tokenizer([i['text'] for i in batch],padding=True,truncation=True,max_length=512,return_tensors='pt')
    input['labels']=torch.LongTensor([i['label'] for i in batch])

    return input

def collate_fn_test(batch,tokenizer):
    input=tokenizer([i['text'] for i in batch],padding=True,truncation=True,max_length=512,return_tensors='pt')

    return input



def main():
    train_path='data/train.txt'
    pretrained_path='../pretrained_model/bert-base-chinese'


    train_dataset=SequenceClassificationDataset(train_path)
    tokenizer=AutoTokenizer.from_pretrained(pretrained_path)
    dataloader=DataLoader(train_dataset,batch_size=32,shuffle=True,collate_fn=lambda x:collate_fn(x,tokenizer))
    for i in dataloader:
        print(i['input_ids'].shape)
        break
   

if __name__=='__main__':
    main()
