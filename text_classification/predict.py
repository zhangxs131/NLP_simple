import numpy as np
import torch
from data_loader import SequenceClassificationDataset,collate_fn_test
from transformers import AutoConfig, AutoModelForSequenceClassification,AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def predict_list(text='./data/test.txt',model_path='model.pth',pretrained_path='bert-base-chinese'):
    #可以文件，也可以list形式

    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset=SequenceClassificationDataset(text,mode='test')
    tokenizer=AutoTokenizer.from_pretrained(pretrained_path)

    test_dataloader=DataLoader(test_dataset,batch_size=32,shuffle=False,collate_fn=lambda x:collate_fn_test(x,tokenizer))

    #加载训练好的模型进行预测
    config=AutoConfig.from_pretrained(pretrained_path,num_labels=10)
    model=AutoModelForSequenceClassification.from_config(config)
    model.load_state_dict(torch.load("model.pth"))
    model=model.to(device)

    result=predict(model,test_dataloader,device,class_list=None)
    print(result)

    #save to txt



def predict(model, data_loader,device='cpu',class_list=None):
    model.eval()


    predict_all = np.array([], dtype=int)

    with torch.no_grad():
        for input in tqdm(data_loader):

            input['input_ids']=input['input_ids'].to(device)
            input['token_type_ids']=input['token_type_ids'].to(device)
            input['attention_mask']=input['attention_mask'].to(device)
            
            outputs = model(**input)

            predic = outputs.logits.argmax(dim=-1).cpu().numpy()

            predict_all = np.append(predict_all, predic)

    if class_list==None:
        return predict_all
    else:
        return [class_list[i] for i in predict_all]


if __name__=='__main__':
    predict_list()
