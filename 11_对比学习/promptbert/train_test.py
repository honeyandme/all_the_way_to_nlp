import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from promptbert_model import PromptBert
from promptbert_get_dataloader import load_data

def cal_loss(query,key,tao=0.05):
    query = F.normalize(query,dim=1)
    key = F.normalize(key, dim=1)

    N,D = query.shape

    batch_pos = torch.exp(torch.div(torch.bmm(query.view(N,1,D),key.view(N,D,1)).view(N,1),tao))

    batch_all = torch.sum(torch.exp(torch.div(torch.mm(query,torch.t(key)),tao)),dim=1)

    loss = torch.mean(-torch.log(torch.div(batch_pos,batch_all)))
    return loss

if __name__ == '__main__':


    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # device = "cpu"
    lr = 1e-5
    model_path = '../../data/chinese-roberta-wwm-ext'
    epoch = 10
    batch_size = 10

    train_dataset, dev_dataset, train_loader, dev_loader = load_data(batch_size=batch_size)
    model = PromptBert(model_path, dropout_prob=0.3, tokenizer=train_dataset.tokenizer).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=lr)
    for e in range(epoch):
        model.train()
        tq = tqdm(train_loader)
        for data in tq:
            opt.zero_grad()
            input_prompt_1 = data['input_prompt_1'].to(device)
            mask_prompt_1 = data['mask_prompt_1'].to(device)
            input_template_1 = data['input_template_1'].to(device)
            mask_template_1 = data['mask_template_1'].to(device)

            input_prompt_2 = data['input_prompt_2'].to(device)
            mask_prompt_2 = data['mask_prompt_2'].to(device)
            input_template_2 = data['input_template_2'].to(device)
            mask_template_2 = data['mask_template_2'].to(device)

            query = model(input_prompt_1, mask_prompt_1, input_template_1, mask_template_1)
            key = model(input_prompt_2, mask_prompt_2, input_template_2, mask_template_2)

            loss =  cal_loss(query,key)
            loss.backward()
            opt.step()

            tq.update()
            tq.set_description(f'e={e} loss={loss.item():.6f}')

        all_pre = []
        all_label = []
        model.eval()
        best_acc = -1
        for data in tqdm(dev_loader):
            input_prompt_1 = data['input_prompt_1'].to(device)
            mask_prompt_1 = data['mask_prompt_1'].to(device)
            input_template_1 = data['input_template_1'].to(device)
            mask_template_1 = data['mask_template_1'].to(device)

            input_prompt_2 = data['input_prompt_2'].to(device)
            mask_prompt_2 = data['mask_prompt_2'].to(device)
            input_template_2 = data['input_template_2'].to(device)
            mask_template_2 = data['mask_template_2'].to(device)

            label = data['label']
            with torch.no_grad():
                outa = model(input_prompt_1, mask_prompt_1, input_template_1, mask_template_1)
                outb = model(input_prompt_2, mask_prompt_2, input_template_2, mask_template_2)

                sim = torch.cosine_similarity(outa,outb,dim=-1)

                pre = (sim>=0.7).long().detach().cpu().numpy()

                all_pre.extend(pre)
                all_label.extend(label)
        acc=accuracy_score(all_pre,all_label)
        if best_acc<acc:
            print(f'acc={acc:.5f}----------------->best')
            torch.save(model.state_dict(),'best.pt')
            best_acc = acc
        else:
            print(f'acc={acc:.5f}')