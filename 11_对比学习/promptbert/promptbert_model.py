import torch
from torch import nn
from transformers import AutoTokenizer,AutoModel,AutoConfig
class PromptBert(nn.Module):
    def __init__(self,model_path,dropout_prob,tokenizer):
        super(PromptBert, self).__init__()
        self.tokenizer = tokenizer

        #修dropout率
        conf = AutoConfig.from_pretrained(model_path)
        conf.attention_probs_dropout_prob = dropout_prob
        conf.hiddem_dropout_prob = dropout_prob

        self.bert = AutoModel.from_pretrained(model_path,config=conf)

        self.mask_id = self.tokenizer.convert_tokens_to_ids('[MASK]')

        self.bert.resize_token_embeddings(len(self.tokenizer))


    def forward(self,input_prompt,mask_prompt,input_template,mask_template):
        promot_out = self.cal_mask_embedding(input_prompt,mask_prompt)
        template_out = self.cal_mask_embedding(input_template,mask_template)
        return promot_out-template_out
    def cal_mask_embedding(self,input_ids,mask):
        last_hidden_state,_ = self.bert(input_ids,mask,return_dict=False)

        mask_index = (input_ids==self.mask_id).long()
        mask_index = mask_index.unsqueeze(-1).expand(last_hidden_state.shape).float()

        return torch.sum(last_hidden_state*mask_index,dim=1)

if __name__ == '__main__':
    from promptbert_get_dataloader import load_data
    train_dataset,dev_dataset,train_loader,dev_loader = load_data()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(device)
    model_path = '../../data/chinese-roberta-wwm-ext'
    model = PromptBert(model_path,dropout_prob=0.3,tokenizer=train_dataset.tokenizer).to(device)

    #测测看能不能跑
    for data in train_loader:
        input_prompt_1 = data['input_prompt_1'].to(device)
        mask_prompt_1 = data['mask_prompt_1'].to(device)
        input_template_1 = data['input_template_1'].to(device)
        mask_template_1 = data['mask_template_1'].to(device)

        input_prompt_2 = data['input_prompt_2'].to(device)
        mask_prompt_2 = data['mask_prompt_2'].to(device)
        input_template_2 = data['input_template_2'].to(device)
        mask_template_2 = data['mask_template_2'].to(device)

        out1 = model(input_prompt_1,mask_prompt_1,input_template_1,mask_template_1)
        out2 = model(input_prompt_2,mask_prompt_2,input_template_2,mask_template_2)
        print('okk')


