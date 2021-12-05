import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import faiss
import numpy as np  
from models.BaseClass import Multi_Intere_Model


class Intere_Gene_Model(Multi_Intere_Model):
    
    def __init__(self, config):
        super(Intere_Gene_Model, self).__init__(config)

        self.attent_dim = self.embed_dim
        self.Gene_num = config.Gene_num
        
        self.Gene_pool = nn.Parameter(torch.randn(self.Gene_num, self.embed_dim))
        self.seq_W1 = nn.Parameter(torch.randn(self.embed_dim, self.attent_dim))
        self.seq_W2= nn.Parameter(torch.randn(self.attent_dim))
        
        if config.RNN == 'GRU':
            self.GRU_layer = nn.GRU(input_size=self.embed_dim, hidden_size=config.GRU_hidden, batch_first=True, bidirectional=True)
        else:
            self.GRU_layer = nn.LSTM(input_size=self.embed_dim, hidden_size=config.GRU_hidden, batch_first=True)
        
        self.Context_MLP = nn.Sequential(
                                        nn.Linear(in_features=int(config.GRU_hidden)*2, out_features=int(config.GRU_hidden)),
                                        nn.ReLU(),
                                        nn.Linear(in_features=int(config.GRU_hidden), out_features=1)
                                            )

        
    
    def Gene_activation(self, seq_embed): #[batch, seq_l, embed_dim]
        seq_att_embed = torch.tanh(torch.einsum('ijk, kl -> ijl', seq_embed, self.seq_W1)) #[batch, seq_l, attent_dim]
        attent_score = F.softmax(torch.einsum('ijk, k -> ij', seq_att_embed, self.seq_W2), dim=1) #[batch, seq_l]
        behavior_vec = torch.einsum('ijk, ij -> ik', seq_embed, attent_score) #[batch, embed_dim]
        Gene_sim_score = torch.einsum('ij, kj -> ik', behavior_vec, self.Gene_pool) #[batch, Gene_num]
        sorted_score, indices = torch.sort(Gene_sim_score, descending = False)  #[batch, Gene_num]
        sorted_scor_K, indice_K = sorted_score[:, : self.intere_num], indices[:, : self.intere_num]  #[batch, intere_num]
        activated_Gene = self.Gene_pool[indice_K] #[batch, intere_num, embed_dim]
        weighted_act_Gene = torch.einsum('ijk, ij -> ijk', activated_Gene, sorted_scor_K) #[batch, intere_num, embed_dim]
        Gene_att_score = F.softmax(torch.einsum('ijk, ilk -> ijl', seq_embed, weighted_act_Gene), dim=2) #[batch, seq_l, intere_num]
        return Gene_att_score
        
    def Context_cal(self, seq_embed): #[batch, seq_l, embed_dim]
        output, h_n = self.GRU_layer(seq_embed) #output:[batch, seq_l, D*hiddin_size]
        context_score = self.Context_MLP(output).squeeze(2) #[batch, seq_l]
        return context_score
    
    def Intere_build(self, seq_embed):  #[batch, seq_l, embed_dim]
        Gene_att = self.Gene_activation(seq_embed) #[batch, seq_l, intere_num]
        context_info = self.Context_cal(seq_embed) #[batch, seq_l]
        context_info_repeat = context_info.unsqueeze(2).repeat(1,1,self.intere_num) #[batch, seq_l, intere_num]
        intere_att_weight = torch.einsum('ijk, ijk -> ijk', Gene_att, context_info_repeat) #[batch, seq_l, intere_num]
        intere_vec = torch.einsum('ijk, ijl -> ikl', intere_att_weight, seq_embed) #[batch, intere_num, embed_dim]
        return intere_vec 
