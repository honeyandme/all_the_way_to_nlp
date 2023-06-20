import torch
from torch import nn
from transformers import AutoModel,AutoConfig
class ESimcse(nn.Module):
    def __init__(self,dropout_prob,model_path):
        super(ESimcse, self).__init__()
        # 修dropout率
        conf = AutoConfig.from_pretrained(model_path)
        conf.attention_probs_dropout_prob = dropout_prob
        conf.hiddem_dropout_prob = dropout_prob

        self.bert = AutoModel.from_pretrained(model_path)
        self.meanpooling = Meanpooling()
        self.loss_fn  = nn.CrossEntropyLoss()
    def forward(self,input_ids,attention_mask,mode='train'):
        last_hidden_state,_ = self.bert(input_ids,attention_mask,return_dict=False)
        sen_emb = self.meanpooling(last_hidden_state,attention_mask)

        return sen_emb
class Meanpooling(nn.Module):
    def forward(self,last_hidden_state,mask):
        mask = mask.unsqueeze(-1).expand(last_hidden_state.size())
        sum_input = torch.sum(last_hidden_state*mask,dim=1)
        sum_mask = torch.sum(mask,dim=1)

        mask = torch.clamp(mask,min=1e-8)
        avg_input = sum_input/sum_mask

        return avg_input
if __name__ == '__main__':
    from esimcse_get_dataloader import load_data

    train_dataloader, dev_dataloader = load_data()
    model_path = '../../data/chinese-roberta-wwm-ext'
    model = ESimcse(dropout_prob=0.3,model_path=model_path)
    for data in train_dataloader:
        input_ids = data['input_ids_source']
        attention_mask = data['attention_mask_source']
        model(input_ids,attention_mask)