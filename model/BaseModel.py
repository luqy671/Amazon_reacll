import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import faiss
import numpy as np  
from models.BaseClass import Multi_Intere_Model


class YouTube_DNN(Multi_Intere_Model):
    
    def __init__(self, config):
        super(YouTube_DNN, self).__init__(config)

        self.MLP = nn.Sequential(
                    nn.Linear(config.embed_dim, int(2*config.embed_dim)),
                    nn.ReLU(),
                    nn.Linear(int(2*config.embed_dim), config.embed_dim))
        print("This YouTube DNN")
    
    def Intere_build(self, seq_embed):  #[batch, seq_l, embed_dim]
        seq_embed_pooling =  torch.mean(seq_embed,dim=1, keepdim=True) #[[batch, 1, embed_dim]
        intere_vec = self.MLP(seq_embed_pooling) #[batch, 1, embed_dim]
        return intere_vec
        

        
class ComiRec_Model(Multi_Intere_Model):
    
    def __init__(self, config):
        super(ComiRec_Model, self).__init__(config)

        self.attent_dim = self.embed_dim
        
        self.W1 = nn.Parameter(torch.randn(self.embed_dim, self.attent_dim))
        self.W2 = nn.Parameter(torch.randn(self.attent_dim, self.intere_num))
    
    def Intere_build(self, seq_embed):  #[batch, seq_l, embed_dim]
        seq_att_embed =  torch.tanh(torch.einsum('ijk,kl->ijl', seq_embed, self.W1)) #[batch, seq_l, attent_dim]
        att_score = F.softmax(torch.einsum('ijk, kl', seq_att_embed, self.W2),dim=2)  #[batch, seq_l, intere_num]
        att_score = att_score.transpose(1,2)  #[batch_size,intere_num,seq_l]
        intere_vec = torch.einsum('ijk, ilj -> ilk', seq_embed, att_score) #[batch, intere_num, embed_dim]
        return intere_vec
    
