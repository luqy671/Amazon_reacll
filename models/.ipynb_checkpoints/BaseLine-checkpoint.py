import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import faiss
import numpy as np  
from models.MultiClass import Multi_Intere_Model

class SINE(Multi_Intere_Model):
    
    def __init__(self, config):
        super(SINE, self).__init__(config)

        self.attent_dim = self.embed_dim
        self.Candidate_num = config.Candidate_num
        
        self.Candidate_pool = nn.Parameter(torch.randn(self.Candidate_num, self.embed_dim))

        self.W1 = nn.Parameter(torch.randn(self.embed_dim, self.attent_dim))
        self.W2= nn.Parameter(torch.randn(self.attent_dim))
        self.W3 = nn.Parameter(torch.randn(self.embed_dim,self.embed_dim))
        self.W_k1 = nn.Parameter(torch.randn(self.embed_dim, self.attent_dim))
        self.W_k2 = nn.Parameter(torch.randn(self.attent_dim, self.intere_num))
        
        self.LayerNorm1 = torch.nn.LayerNorm(self.embed_dim,elementwise_affine==False)
        self.LayerNorm2 = torch.nn.LayerNorm(self.embed_dim,elementwise_affine==False)
        self.LayerNorm3 = torch.nn.LayerNorm(self.embed_dim,elementwise_affine==False)
        
        print("init SINE")
    
    def cal_Cu(self, seq_embed): #[batch, seq_l, embed_dim]
        # seq_embed 转到 attention_vec域（双线性self attention）
        seq_att_embed = torch.tanh(torch.einsum('ijk, kl -> ijl', seq_embed, self.W1)) #[batch, seq_l, attent_dim]
        # 为每一个batch，计算一组 attention score
        attent_score = F.softmax(torch.einsum('ijk, k -> ij', seq_att_embed, self.W2), dim=1) #[batch, seq_l]
        # 利用attention 得分，将每一个batch的seq_embed加权聚合为一个embedding
        behavior_vec = torch.einsum('ijk, ij -> ik', seq_embed, attent_score) #[batch, embed_dim]
        # 计算每个embedding与Candidate_pool中vec的相似度
        Su = torch.einsum('ij, kj -> ik', behavior_vec, self.Candidate_pool) #[batch, Candidate_num]
        # 对每个batch内的得分，降序排列
        sorted_score, indices = torch.sort(Su, descending = True)  #[batch, Candidate_num]
        # 取 top k
        sorted_scor_K, indices_K = sorted_score[:, : self.intere_num], indices[:, : self.intere_num]  #[batch, intere_num]
        #计算 Cu
        C_topK = self.Candidate_pool[indices_K] #[batch, intere_num, embed_dim]
        Su_topK = torch.sigmoid(Su[indices_K]) #[batch, intere_num]
        print("C_top")
        print(C_top.shape)
        print("Su_top")
        print(Su_top.shape)
        Cu = torch.einsum('ijk,ij->ijk', C_topK, Su_topK)  #[batch, intere_num, embed_dim]              
        return Cu
    
    def Intere_build(self, seq_embed):  #[batch, seq_l, embed_dim]
        Cu = self.cal_Cu(seq_embed) #[batch, intere_num, embed_dim]
        
        X_W3_norm = self.LayerNorm1(torch.einsum('ijk,kl->ijl',seq_embed,self.W3)) #[batch, seq_l, embed_dim]
        Cu_norm = self.LayerNorm2(Cu) #[batch, intere_num, embed_dim]
        P_kt = F.softmax(torch.einsum('ijl,ikl->ijk',X_W3_norm,Cu_norm), dim = 2) #[batch, seq_l, intere_num]
        
        P_tk = F.softmax(torch.tanh(torch.einsum('ijl,'seq_embed,self.W_k1)
        
        seq_att_embed =  torch.tanh(torch.einsum('ijk,kl->ijl', seq_embed, self.W_k1)) #[batch, seq_l, attent_dim]
        P_tk = F.softmax(torch.einsum('ijk, kl -> ijl', seq_att_embed, self.W_k2),dim=1) #[batch, seq_l, intere_num]
        intere_vec = LayerNorm3(torch.einsum('ijk,ijl->ikl',torch.mul(P_kt,P_tk),seq_embed))#[batch, intere_num, embed_dim]
        
        return intere_vec, []      

'''**********************************************************************************'''        
class ComiRec(Multi_Intere_Model):
    
    def __init__(self, config):
        super(ComiRec, self).__init__(config)

        self.attent_dim = self.embed_dim
        
        self.W1 = nn.Parameter(torch.randn(self.embed_dim, self.attent_dim))
        self.W2 = nn.Parameter(torch.randn(self.attent_dim, self.intere_num))
        print("**********ComiRec init*********")
    
    def Intere_build(self, seq_embed):  #[batch, seq_l, embed_dim]
        seq_att_embed =  torch.tanh(torch.einsum('ijk,kl->ijl', seq_embed, self.W1)) #[batch, seq_l, attent_dim]
        att_score = F.softmax(torch.einsum('ijk, kl -> ijl', seq_att_embed, self.W2),dim=1)  #[batch, seq_l, intere_num]
        intere_vec = torch.einsum('ijk, ijl -> ilk', seq_embed, att_score) #[batch, intere_num, embed_dim]
        return intere_vec, []
    
'''**********************************************************************************'''

class YouTube_DNN(Multi_Intere_Model):
    
    def __init__(self, config):
        super(YouTube_DNN, self).__init__(config)

        self.MLP = nn.Sequential(
                    nn.Linear(config.embed_dim, int(2*config.embed_dim)),
                    nn.ReLU(),
                    nn.Linear(int(2*config.embed_dim), config.embed_dim))
        print("**********YouTube DNN init*********")
    
    def Intere_build(self, seq_embed):  #[batch, seq_l, embed_dim]
        seq_embed_pooling =  torch.mean(seq_embed,dim=1, keepdim=True) #[[batch, 1, embed_dim]
        intere_vec = self.MLP(seq_embed_pooling) #[batch, 1, embed_dim]
        return intere_vec, []