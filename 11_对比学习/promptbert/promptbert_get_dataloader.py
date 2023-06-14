import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset,DataLoader
class P_dataset(Dataset):
    def __init__(self,df,tokenizer,max_len=128,mode='train'):
        self.df = df
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = max_len
        if mode == 'train':
            self.listdf = self.get_listdf(df)
    def get_listdf(self,df):
        lis = df['text_a'].tolist()
        lis.extend(df['text_b'].tolist())
        return lis
    def __getitem__(self, x):
        if self.mode == 'train':
            text = self.listdf[x]
            prompt_texts = self.template(text,mode='train')
            return prompt_texts
        else:
            data = self.df.iloc[x]
            text_a,text_b,label = data['text_a'],data['text_b'],data['label']

            prompt_text_a = self.template(text_a,mode='dev')
            prompt_text_b = self.template(text_b,mode='dev')

            return prompt_text_a+prompt_text_b,label
    def template(self,sentence,mode='train'):
        if mode=='train':
            prompt_tem = ['[X]，这句话的意思是[MASK]','[X]，它的意思是[MASK]']
        else:
            prompt_tem = ['[X]，这句话的意思是[MASK]']
        semtence_tem = []
        for template in prompt_tem:
            prompt_sentence = template.replace('[X]',sentence)
            template_sentence = template.replace('[X]','[X]'*len(sentence))
            semtence_tem+=[prompt_sentence,template_sentence]
        return semtence_tem
    def __len__(self):
        if self.mode=='train':
            return len(self.listdf)
        else:
            return len(self.df)
    def collate_fn(self,batch):
        if self.mode == 'dev':
            label = [x[1] for x in batch]
            batch = [x[0] for x in batch]

        prompt_1 = [x[0] for x in batch]
        template_1 = [x[1] for x in batch]
        prompt_2 = [x[2] for x in batch]
        template_2 = [x[3] for x in batch]

        batch_prompt_1 = self.tokenizer(prompt_1,truncation=True, max_length=self.max_len,padding='longest')
        batch_template_1 = self.tokenizer(template_1, truncation=True, max_length=self.max_len, padding='longest')
        batch_prompt_2 = self.tokenizer(prompt_2, truncation=True, max_length=self.max_len, padding='longest')
        batch_template_2 = self.tokenizer(template_2, truncation=True, max_length=self.max_len, padding='longest')

        return_dict= {
            "input_prompt_1":torch.as_tensor(batch_prompt_1['input_ids'],dtype=torch.long),
            "mask_prompt_1": torch.as_tensor(batch_prompt_1['attention_mask'],dtype=torch.long),
            "input_template_1": torch.as_tensor(batch_template_1['input_ids'],dtype=torch.long),
            "mask_template_1": torch.as_tensor(batch_template_1['attention_mask'],dtype=torch.long),
            "input_prompt_2": torch.as_tensor(batch_prompt_2['input_ids'],dtype=torch.long),
            "mask_prompt_2": torch.as_tensor(batch_prompt_2['attention_mask'],dtype=torch.long),
            "input_template_2": torch.as_tensor(batch_template_2['input_ids'],dtype=torch.long),
            "mask_template_2": torch.as_tensor(batch_template_2['attention_mask'],dtype=torch.long),
        }
        if self.mode=='dev':
            return_dict['label'] = torch.as_tensor(label, dtype=torch.long)
        return return_dict
    # def collate_fn_dev(self,batch):
    #     prompt_1 = [x[0][0] for x in batch]
    #     template_1 = [x[0][1] for x in batch]
    #     prompt_2 = [x[0][2] for x in batch]
    #     template_2 = [x[0][3] for x in batch]
    #     label = [x[1] for x in batch]
    #
    #     batch_prompt_1 = self.tokenizer(prompt_1, truncation=True, max_length=self.max_len, padding='longest')
    #     batch_template_1 = self.tokenizer(template_1, truncation=True, max_length=self.max_len, padding='longest')
    #     batch_prompt_2 = self.tokenizer(prompt_2, truncation=True, max_length=self.max_len, padding='longest')
    #     batch_template_2 = self.tokenizer(template_2, truncation=True, max_length=self.max_len, padding='longest')
    #
    #     return {
    #         "input_prompt_1": torch.as_tensor(batch_prompt_1['input_ids'], dtype=torch.long),
    #         "mask_prompt_1": torch.as_tensor(batch_prompt_1['attention_mask'], dtype=torch.long),
    #         "input_template_1": torch.as_tensor(batch_template_1['input_ids'], dtype=torch.long),
    #         "mask_template_1": torch.as_tensor(batch_template_1['attention_mask'], dtype=torch.long),
    #         "input_prompt_2": torch.as_tensor(batch_prompt_2['input_ids'], dtype=torch.long),
    #         "mask_prompt_2": torch.as_tensor(batch_prompt_2['attention_mask'], dtype=torch.long),
    #         "input_template_2": torch.as_tensor(batch_template_2['input_ids'], dtype=torch.long),
    #         "mask_template_2": torch.as_tensor(batch_template_2['attention_mask'], dtype=torch.long),
    #         "label":torch.as_tensor(label, dtype=torch.long)
    #     }
def load_data(batch_size=30):
    train_df = pd.read_csv('../data/ants/train.csv')
    dev_df = pd.read_csv('../data/ants/dev.csv')
    tokenizer = AutoTokenizer.from_pretrained('../../data/chinese-roberta-wwm-ext')
    tokenizer.add_tokens('[X]')
    train_dataset = P_dataset(train_df,tokenizer,mode='train')
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=train_dataset.collate_fn)
    dev_dataset = P_dataset(dev_df, tokenizer, mode='dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn)

    return train_dataset,dev_dataset,train_dataloader,dev_dataloader

if __name__ == '__main__':
    train_dataloader,dev_dataloader = load_data()
    for data in train_dataloader:
        print(data['input_prompt_1'].shape)
        # break
    for data in dev_dataloader:
        print(data['input_prompt_1'].shape)
        break