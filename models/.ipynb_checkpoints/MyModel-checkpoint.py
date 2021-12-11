import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import faiss
import numpy as np  
import os
from models.MultiClass import Multi_Intere_Model
 


class MyModel_v3(Multi_Intere_Model):
    
    def __init__(self, config):
        super(MyModel_v3, self).__init__(config)

        self.attent_dim = self.embed_dim
        self.Candidate_num = config.Candidate_num
        
        self.Candidate_pool = nn.Parameter(torch.randn(self.Candidate_num, self.embed_dim))
        # 将item_embedding转换到 attention 域
        self.seq_W1 = nn.Parameter(torch.randn(self.embed_dim, self.attent_dim))
        # 和这个vec越相似， item得分越高
        self.seq_W2= nn.Parameter(torch.randn(self.attent_dim))
        print("init My v3")
    
    def Candidate_activation(self, seq_embed): #[batch, seq_l, embed_dim]
        # seq_embed 转到 attention_vec域（双线性self attention）
        seq_att_embed = torch.tanh(torch.einsum('ijk, kl -> ijl', seq_embed, self.seq_W1)) #[batch, seq_l, attent_dim]
        # 为每一个batch，计算一组 attention score
        attent_score = F.softmax(torch.einsum('ijk, k -> ij', seq_att_embed, self.seq_W2), dim=1) #[batch, seq_l]
        # 利用attention 得分，将每一个batch的seq_embed加权聚合为一个embedding
        behavior_vec = torch.einsum('ijk, ij -> ik', seq_embed, attent_score) #[batch, embed_dim]
        # 计算每个embedding与Candidate_pool中vec的相似度
        Candidate_sim_score = torch.einsum('ij, kj -> ik', behavior_vec, self.Candidate_pool) #[batch, Candidate_num]
        # 对每个batch内的得分，降序排列
        sorted_score, indices = torch.sort(Candidate_sim_score, descending = True)  #[batch, Candidate_num]
        # 取 top k
        sorted_scor_K, indices_K = sorted_score[:, : self.intere_num], indices[:, : self.intere_num]  #[batch, intere_num]
        activated_Candidate = self.Candidate_pool[indices_K] #[batch, intere_num, embed_dim]   
        # 计算 top k 个兴趣的 attention_score
        Candidate_att_score = F.softmax(torch.einsum('ijk, ilk -> ijl', seq_att_embed, activated_Candidate), dim=1) #[batch, seq_l, intere_num]
        return Candidate_att_score, indices_K
    
    def Intere_build(self, seq_embed):  #[batch, seq_l, embed_dim]
        Candidate_att, indices_K = self.Candidate_activation(seq_embed) #[batch, seq_l, intere_num] [batch, intere_num]
        intere_vec = torch.einsum('ijk, ijl -> ikl', Candidate_att, seq_embed) #[batch, intere_num, embed_dim]      
        return intere_vec, indices_K
