import torch
from torch import nn
from transformers import AutoModel
class simcse(nn.Module):
    def __init__(self):
        super(simcse, self).__init__()
        self.bert = AutoModel.from_pretrained('../../data/chinese-roberta-wwm-ext')
        self.meanpooling = Meanpooling()
        self.loss_fn  = nn.CrossEntropyLoss()
    def forward(self,input_ids,attention_mask,mode='train'):
        last_hidden_state,_ = self.bert(input_ids,attention_mask,return_dict=False)
        sen_emb = self.meanpooling(last_hidden_state,attention_mask)

        if mode == 'train':
            loss = self.cal_loss(sen_emb)
            return loss
        else:
            return sen_emb
    def cal_loss(self,sen_emb,tao=0.05):
        id = torch.arange(0, sen_emb.shape[0], dtype=torch.long, device=sen_emb.device)
        y_true = id + 1 - id % 2 * 2

        sim_martix = torch.cosine_similarity(sen_emb.unsqueeze(0),sen_emb.unsqueeze(1),dim=-1,)
        sim_martix = sim_martix - torch.eye(sen_emb.shape[0])*1e9
        sim_martix = sim_martix/tao

        loss = self.loss_fn(sim_martix,y_true)
        return torch.mean(loss)
class Meanpooling(nn.Module):
    def forward(self,last_hidden_state,mask):
        mask = mask.unsqueeze(-1).expand(last_hidden_state.size())
        sum_input = torch.sum(last_hidden_state*mask,dim=1)
        sum_mask = torch.sum(mask,dim=1)

        mask = torch.clamp(mask,min=1e-8)
        avg_input = sum_input/sum_mask

        return avg_input
if __name__ == '__main__':
    from simcse_get_dataloader import load_data

    train_dataloader, dev_dataloader = load_data()
    model = simcse()
    for data in train_dataloader:
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        model(input_ids,attention_mask)