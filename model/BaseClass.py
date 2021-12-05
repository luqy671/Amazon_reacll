import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import faiss
import numpy as np

class Multi_Intere_Model(nn.Module):
    
    def __init__(self, config):
        super(Multi_Intere_Model, self).__init__()
        self.embed_dim = config.embed_dim
        self.intere_num = config.intere_num
        self.item_num = config.item_num
        self.sample_num = config.sample_num
        self.max_N = config.N_list[-1]
        
        self.item_embed = nn.Embedding(config.item_num, self.embed_dim)
    
    def Intere_build(self, seq_embed):  #[batch, seq_l, embed_dim]
        pass 
        
    def Neg_sample(self, seqs):
        '''
        sample_weight = torch.ones([seqs.size()[0], self.item_num])
        idx_x = []
        idx_y = []
        for i,seq in enumerate(seqs):
            idx_x.extend(torch.tensor([i]*seq.size()[0]))
            idx_y.extend([x for x in seq])
        idx_x = torch.tensor(idx_x, dtype = torch.long)
        idx_y = torch.tensor(idx_y, dtype = torch.long)
        sample_weight.index_put_((idx_x, idx_y), torch.tensor(0.))
        samples = torch.multinomial(sample_weight, self.sample_num).cuda() #[batch_size, sample_num]'''
        samples = torch.tensor(np.random.randint(0, self.item_num, (seqs.size()[0], self.sample_num))).cuda()
        sample_embeddings = self.item_embed(samples) #[batch_size, sample_num, embedding_size]
        return sample_embeddings
      
    def forward(self, seqs):   #[batch, seq_l]
        seqs = seqs.long()
        seq_l = seqs.size()[1]
        loss = 0
        for i in range(1,seq_l):
            target = self.item_embed(seqs[:, i].view(-1,1)).squeeze(1)  #[batch, embed_dim]
            seq_embed = self.item_embed(seqs[:, :i])  #[batch, seq_l, embed_dim]
            intere_vec = self.Intere_build(seq_embed)  #[batch, intere_num, embed_dim]
            hitted_idx = torch.argmax(torch.einsum('ijk, ik ->ij', intere_vec, target), dim=1) #[batch]
            hitted_intere = torch.cat([intere_vec[a,b,:] for a,b in enumerate(hitted_idx)], dim=0).reshape(-1,self.embed_dim) #[batch, embed_dim]
            
            pos_score = torch.einsum('ij, ij ->i', target, hitted_intere) #[batch]
            neg_sample = self.Neg_sample(seqs) #[batch, sample_num, embed_dim]
            neg_score = torch.einsum('ijk, ik -> ij', neg_sample, hitted_intere) #[batch, sample_num]
            max_v, max_idx = torch.max(neg_score, 1)
            max_v_expand = max_v.view(-1,1).expand(neg_score.shape[0], neg_score.shape[1])
            neg_score = max_v.view(-1,) + torch.log(torch.sum(torch.exp(neg_score-max_v_expand), dim=1)) #[batch]
            loss += (neg_score - pos_score).sum()
            #print("%f"%loss)
        return loss
    
    def faiss_fun(self, intere_vec, item_embeds):
        intere_vec = intere_vec.cpu().detach().numpy()
        item_embeds = item_embeds.cpu().detach().numpy()
        intere_vec = intere_vec.astype('float32')
        item_embeds = item_embeds.astype('float32')
        
        index = faiss.IndexFlatIP(item_embeds.shape[1])
        index.add(item_embeds)
        D_list = []
        I_list = []
        for vec_u in intere_vec :
            D, I = index.search(np.ascontiguousarray(vec_u), self.max_N)
            D_list.append(D)
            I_list.append(I)
        for items in I_list:
            items.resize(items.shape[0]*items.shape[1])
        for ds in D_list:
            ds.resize(ds.shape[0]*ds.shape[1])
        user_topN = []
        for i in range(len(I_list)):
            I_D = list(zip(I_list[i],D_list[i]))
            I_D.sort(key= lambda k:k[1])
            ID_dict = dict(I_D)
            sort_ID = sorted(ID_dict.items(), key=lambda item:item[1],reverse=True)
            user_topN.append([x[0] for x in sort_ID][:self.max_N])
        return np.array(user_topN)
    
    def serving(self, seqs): #[batch, seq_l]
        seqs = seqs.long()
        seq_embed = self.item_embed(seqs) #[batch, seq_l, embed_dim]
        intere_vec = self.Intere_build(seq_embed)  #[batch, intere_num, embed_dim]
        item_set = torch.tensor([x for x in range(self.item_num)]).cuda()
        item_embeds = self.item_embed(item_set) #[item_num, embedding_size]
        recall_topN = self.faiss_fun(intere_vec, item_embeds) #[batch, max_N]
        return recall_topN  