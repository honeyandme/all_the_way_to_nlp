import torch
import copy
import os
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from esimcse_get_dataloader import load_data
from esimcse_model import ESimcse
from tqdm import tqdm
# 固定seed
def seed_everything(seed=42):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

model_path = '../../data/chinese-roberta-wwm-ext'

batch_size = 32
queue_size = 32
queue_num = int(queue_size * batch_size)

# 动量的系数
momentum = 0.999

# 导入测试集
train_data_loader, dev_data_loader = load_data(batch_size)
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

# 加载BERT模型
query_model = ESimcse(0.3,model_path)
# 将模型移动到GPU上
query_model = query_model.to(device)

# 队列的encoder
key_model = copy.deepcopy(query_model)

for para in key_model.parameters():
    para.requires_grad = False
key_model.eval()

def construct_queue(train_loder,queue_num):
    flag = 0

    # [queue_num,dim]
    queue = None

    with torch.no_grad():
        while True:
            for idx,data in enumerate(train_loder):
                """
                "input_ids_source": batch_input_ids_source,
                "attention_mask_source": batch_attention_mask_source,
                "input_ids_repeat": batch_input_ids_repeat,
                "attention_mask_repeat": batch_attention_mask_repeat,
                """
                if idx<20:
                    continue
                input_ids_source = data['input_ids_source'].to(device)
                attention_masks_source = data['attention_mask_source'].to(device)

                queue_data = query_model(input_ids_source,attention_masks_source)

                if queue is None:
                    queue = queue_data
                else:
                    queue = torch.cat((queue,queue_data),dim=0)
                    if queue.shape[0]>=queue_num:
                        flag = 1
                if flag == 1:
                    break
            if flag == 1:
                break
    queue = queue[-queue_num:]
    # 【queue_num,dim】
    queue = F.normalize(queue, 1)
    return queue
def cal_loss(query,key,queue,tao=0.05):
    """
        :param query:  句子本身的emb [b,d]
        :param key:  正例句子emb  [b,d]
        :param queue: 负例句子的emb  [queue_num,d]
        :param tao:
        :return: loss
    """
    query = F.normalize(query, dim=1)
    key = F.normalize(key, dim=1)
    queue = F.normalize(queue, dim=1)

    N, D = query.shape[0], query.shape[1]

    pos = torch.exp(torch.div(torch.bmm(query.view(N,1,D),key.view(N,D,1)),tao))

    pos_batch = torch.sum(torch.exp(torch.div(torch.mm(query.view(N,D),torch.t(key)),tao)),dim=1)

    queue_all = torch.sum(torch.exp(torch.div(torch.mm(query.view(N, D), torch.t(queue)), tao)), dim=1)

    loss = torch.mean(-torch.log(torch.div(pos, (pos_batch + queue_all))))
    return loss

def trainer():
    seed_everything(3407)

    opt = torch.optim.AdamW(query_model.parameters(),lr = 3e-5)

    best_acc = 0

    for e in range(5):
        print('epoch', e + 1)

        query_model.train()

        #采集负样本队列
        queue_embeddings = construct_queue(train_data_loader,queue_num)

        pbar = tqdm(train_data_loader)

        for data in pbar:
            input_ids_source = data['input_ids_source'].to(device)
            attention_mask_source = data['attention_mask_source'].to(device)
            input_ids_repeat = data['input_ids_repeat'].to(device)
            attention_mask_repeat = data['attention_mask_repeat'].to(device)

            opt.zero_grad()

            query_emb = query_model(input_ids_source,attention_mask_source)
            key_emb = key_model(input_ids_repeat,attention_mask_repeat)

            loss = cal_loss(query_emb,key_emb,queue_embeddings)

            loss.backward()
            opt.step()

            #动态更新queue
            queue_embeddings = torch.cat((queue_embeddings, key_emb), 0)
            queue_embeddings = queue_embeddings[-queue_num:]
            #动量
            for param_q,param_k in zip(query_model.parameters(),key_model.parameters()):
                param_k.data.copy_(momentum*param_k.data + (1-momentum)*param_q.data)
                param_k.requires_grad = False

            pbar.update()
            pbar.set_description(f'loss:{loss.item():.4f}')
        pred = []
        label = []

        query_model.eval()

        for data in tqdm(dev_data_loader):

            input_ids_a = data['input_ids_a'].to(device)
            attention_masks_a = data['attention_mask_a'].to(device)
            input_ids_b = data['input_ids_b'].to(device)
            attention_masks_b = data['attention_mask_b'].to(device)
            labels = data['label'].to(device)

            with torch.no_grad():
                # 前向传播
                sentence_embedding_a = query_model(input_ids_a, attention_masks_a)
                sentence_embedding_b = query_model(input_ids_b, attention_masks_b)

            similarity = F.cosine_similarity(sentence_embedding_a, sentence_embedding_b, dim=-1)
            # 获取预测值
            similarity = similarity.detach().cpu().numpy()
            pred.extend(similarity)
            # 获取标签
            label.extend(labels.cpu().numpy())
        # 计算验证集准确率
        pred = torch.tensor(pred)
        pred = (pred >= 0.7).long().detach().cpu().numpy()
        acc = accuracy_score(pred, label)
        print('dev acc:', acc)
        print()
        # 如果当前准确率大于最佳准确率，则保存模型参数
        if acc > best_acc:
            torch.save(query_model.state_dict(), './best.bin')
            best_acc = acc
if __name__ == '__main__':
    trainer()