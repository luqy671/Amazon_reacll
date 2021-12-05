import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import faiss
import numpy as np  
import os

class Multi_Intere_Model(nn.Module):
    
    def __init__(self, config):
        super(Multi_Intere_Model, self).__init__()
        self.embed_dim = config.embed_dim
        self.intere_num = config.intere_num
        self.item_num = config.item_num
        self.sample_num = config.sample_num
        self.max_N = config.N_list[-1]
        
        self.item_embed = nn.Embedding(config.item_num, self.embed_dim)
        
        print("Baseclass norm")
        print("intere_num: %d"%self.intere_num)
        
        
    def __checkvalid__(self, vec, position):  
        nan_num = torch.isnan(vec).int().sum()
        if  nan_num > 0 :
            print("%s is nan!!!!!!!!!!!"%position)
            os._exit()
        inf_num = vec.numel() - torch.isfinite(vec).int().sum()
        if inf_num > 0 :
            print("%s is inf!!!!!!!!!!!"%position)
            os._exit()
        
    def norm_embed(self, seqs):
        seq_embed = self.item_embed(seqs)
        norm_dim = seq_embed.ndim -1
        norm2 = torch.norm(seq_embed, dim=norm_dim).unsqueeze(dim=norm_dim)
        #norm2 += 1e9
        norm_embed = seq_embed/norm2
        return norm_embed
    
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
        sample_embeddings = self.norm_embed(samples) #[batch_size, sample_num, embedding_size]
        self.__checkvalid__(sample_embeddings, "sample embeds")
        return sample_embeddings
      
    def forward(self, seqs):   #[batch, seq_l]
        seqs = seqs.long()
        seq_l = seqs.size()[1]
        loss = 0
        for i in range(1,seq_l):
            target = self.norm_embed(seqs[:, i].view(-1,1)).squeeze(1)  #[batch, embed_dim]
            self.__checkvalid__(target, "target embeds")
            seq_embed = self.norm_embed(seqs[:, :i])  #[batch, seq_l, embed_dim]
            self.__checkvalid__(seq_embed, "seq embeds")
            intere_vec = self.Intere_build(seq_embed)  #[batch, intere_num, embed_dim]
            self.__checkvalid__(intere_vec, "intere_vec")
            hitted_idx = torch.argmax(torch.einsum('ijk, ik ->ij', intere_vec, target), dim=1) #[batch]
            hitted_intere = torch.cat([intere_vec[a,b,:] for a,b in enumerate(hitted_idx)], dim=0).reshape(-1,self.embed_dim) #[batch, embed_dim]
            
            pos_score = torch.einsum('ij, ij ->i', target, hitted_intere) #[batch]
            self.__checkvalid__(pos_score, "pos_score")
            neg_sample = self.Neg_sample(seqs) #[batch, sample_num, embed_dim] 
            neg_score = torch.einsum('ijk, ik -> ij', neg_sample, hitted_intere) #[batch, sample_num]
            self.__checkvalid__(neg_score, "neg_score")
            max_v, max_idx = torch.max(neg_score, 1)
            self.__checkvalid__(max_v, "max_v")
            max_v_expand = max_v.view(-1,1).expand(neg_score.shape[0], neg_score.shape[1])
            neg_score = max_v.view(-1,) + torch.log(torch.sum(torch.exp(neg_score-max_v_expand), dim=1)) #[batch]
            self.__checkvalid__(neg_score, "neg_score - max_v")
            loss += (neg_score - pos_score).sum()
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
        seq_embed = self.norm_embed(seqs) #[batch, seq_l, embed_dim]
        intere_vec = self.Intere_build(seq_embed)  #[batch, intere_num, embed_dim]
        item_set = torch.tensor([x for x in range(self.item_num)]).cuda()
        item_embeds = self.norm_embed(item_set) #[item_num, embedding_size]
        recall_topN = self.faiss_fun(intere_vec, item_embeds) #[batch, max_N]
        return recall_topN  


class MyModel_v3(Multi_Intere_Model):
    
    def __init__(self, config):
        super(MyModel_v3, self).__init__(config)

        self.attent_dim = self.embed_dim
        self.Gene_num = config.Gene_num
        self.BiLinear = config.BiLinear
        
        self.Gene_pool = nn.Parameter(torch.randn(self.Gene_num, self.embed_dim))
        self.seq_W1 = nn.Parameter(torch.randn(self.embed_dim, self.attent_dim))
        self.seq_W2= nn.Parameter(torch.randn(self.attent_dim))
        
        self.print = True
        
        print("Gene_num: %d"%self.Gene_num)
    
    def Gene_activation(self, seq_embed): #[batch, seq_l, embed_dim]
        seq_att_embed = torch.tanh(torch.einsum('ijk, kl -> ijl', seq_embed, self.seq_W1)) #[batch, seq_l, attent_dim]
        attent_score = F.softmax(torch.einsum('ijk, k -> ij', seq_att_embed, self.seq_W2), dim=1) #[batch, seq_l]
        behavior_vec = torch.einsum('ijk, ij -> ik', seq_embed, attent_score) #[batch, embed_dim]
        Gene_sim_score = torch.einsum('ij, kj -> ik', behavior_vec, self.Gene_pool) #[batch, Gene_num]
        sorted_score, indices = torch.sort(Gene_sim_score, descending = False)  #[batch, Gene_num]
        sorted_scor_K, indice_K = sorted_score[:, : self.intere_num], indices[:, : self.intere_num]  #[batch, intere_num]
        activated_Gene = self.Gene_pool[indice_K] #[batch, intere_num, embed_dim]
        #双线性参数：0-不使用；1-复用W1；2-使用单独的W3；3-Gene_vec加权

        if self.BiLinear == "1":
            if self.print:
                print("BiLinear 1")
                self.print = False
            seq_embed = torch.tanh(torch.einsum('ijk, kl -> ijl', seq_embed, self.seq_W1)) #[batch, seq_l, attent_dim]    
            
        Gene_att_score = F.softmax(torch.einsum('ijk, ilk -> ijl', seq_embed, activated_Gene), dim=2) #[batch, seq_l, intere_num]
        return Gene_att_score
    
    def Intere_build(self, seq_embed):  #[batch, seq_l, embed_dim]
        Gene_att = self.Gene_activation(seq_embed) #[batch, seq_l, intere_num]
        intere_vec = torch.einsum('ijk, ijl -> ikl', Gene_att, seq_embed) #[batch, intere_num, embed_dim]      
        return intere_vec
