import torch
from tqdm import tqdm
from simcse_get_dataloader import load_data
from simcse_model import Simcse
from sklearn.metrics import accuracy_score
def train(threshold=0.5):
    batch_size = 10
    lr = 1e-5
    epoch = 10
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # device = "cpu"

    train_dataloader,dev_dataloader = load_data(batch_size=batch_size)
    model = Simcse().to(device)
    opt = torch.optim.AdamW(model.parameters(),lr = lr)

    for e in range(epoch):
        model.train()
        tq = tqdm(train_dataloader)
        for data in tq:
            opt.zero_grad()
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            loss = model(input_ids,attention_mask,mode='train')
            loss.backward()
            opt.step()
            tq.update()
            tq.set_description(f'e= {e+1} loss={loss.item():.6f}')
        model.eval()
        pre = []
        lab = []
        for data in tqdm(dev_dataloader):
            input_ids_a = data['input_ids_a'].to(device)
            attention_mask_a = data['attention_mask_a'].to(device)
            input_ids_b = data['input_ids_b'].to(device)
            attention_mask_b = data['attention_mask_b'].to(device)
            label = data['label']

            with torch.no_grad():
                sen_emb_a = model(input_ids_a,attention_mask_a,mode='dev')
                sen_emb_b = model(input_ids_b, attention_mask_b, mode='dev')

                sim = torch.cosine_similarity(sen_emb_a,sen_emb_b,dim=-1)

                sim = (sim>=threshold).long().detach().cpu().numpy()
            pre.extend(sim)
            lab.extend(label)
        print(f'acc={accuracy_score(pre,lab)}')







if __name__ == '__main__':
    train()