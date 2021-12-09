import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import copy
import csv
from models import MyModel

def train_test(config):
    # 读入数据
    train_lists, test_x, test_y = Data_Read(config)
    train_loader_list = []
    for train_data in train_lists:
        train_set = TensorDataset(torch.tensor(train_data))
        train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size,
                              shuffle=True, num_workers=0)
        train_loader_list.append(train_loader)
    test_set = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))
    test_loader = DataLoader(dataset=test_set, batch_size=config.batch_size,
                              shuffle=False, num_workers=0)
    # 模型和优化器初始化
    model = MyModel.MyModel_v3(config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr, weight_decay=config.L2)
    print("***************model structure***************************")
    print(model)
    
    # 训练 && 测试
    print("****************train and test***************************")
    Recall_list = defaultdict(list)
    HitRate_list = defaultdict(list)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        '''*********************  training  **************************'''
        for train_loader in train_loader_list:
            for data in train_loader:
                data = data[0].cuda()
                optimizer.zero_grad()
                loss = model(data)
                total_loss += loss*1e-6
                loss.backward()
                optimizer.step()
        print("epoch: %d   loss: %10.4f\n"%(epoch+1,total_loss))

        '''*********************  testing  **************************'''
        model.eval()
        test_num = 0
        Recall = defaultdict(float)
        HitRate = defaultdict(float)
        for x, y in test_loader:
            x = x.cuda()
            y = y.numpy()
            recall_topN = model.serving(x) #[batch, max_N]

            test_num += x.shape[0]
            for N in config.N_list:
                this_N = recall_topN[:,:N]
                for i in range(y.shape[0]):
                    Recall[N] += (np.intersect1d(this_N[i],y[i]).shape[0])/y[i].shape[0]
                    HitRate[N] += np.intersect1d(this_N[i],y[i]).shape[0] > 0

        for N in config.N_list:
            Recall[N] = Recall[N]/test_num*100
            HitRate[N] = HitRate[N]/test_num*100
            print("Recall%d :  %10.4f    HitRate%d :  %10.4f\n"%(N,Recall[N],N,HitRate[N]))
            Recall_list[N].append(copy.deepcopy(Recall[N]))
            HitRate_list[N].append(copy.deepcopy(HitRate[N]))

    torch.save(model, config.model_save_path + config.model_name + ".pth") 
    torch.save(model.state_dict(),config.model_save_path + config.model_name + "_para.pth")
    with open (config.ans_save_path + config.model_name +".csv", 'w', encoding='utf8') as f:
        writer  = csv.writer(f)
        for N in config.N_list:
            writer.writerow(Recall_list[N])
            writer.writerow(HitRate_list[N])
            
def Data_Read(config):
    seq_list = []
    with open(config.data_dir + config.company_cate + '_seqs.txt') as f:
        for l in f.readlines():
            seq_list.append([int(x) for x in l.split(' ')[:-1]])
    seq_same_len_list = []
    now_len = -1
    for seq in seq_list:
        if(len(seq) != now_len):
            if(now_len != -1):
                seq_same_len_list.append(np.array(seq_same_len))
            now_len = len(seq)
            seq_same_len = []
            seq_same_len.append(seq)
        else:
            seq_same_len.append(seq)
    test_data = np.array(seq_same_len)
    return seq_same_len_list, test_data[:,:config.test_cut], test_data[:, config.test_cut:]