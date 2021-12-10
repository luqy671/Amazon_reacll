import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import faiss
import numpy as np  
import os
from models.MyModel import MyModel_v3
 


class Glove_v3(MyModel_v3):
    
    def __init__(self, config):
        super(Glove_v3, self).__init__(config)
        
        
        self.Glove_window = config.Glove_window
        self.co_max = config.co_max
        # 共现矩阵
        self.co_matrix = np.zeros((self.item_num, self.item_num))
        self.co_bias = nn.Parameter(torch.randn(self.item_num))        
        print("init Glove v3")
    
    def build_co_matrix(self, train_lists):
        print("build Co_occur matrix")
        for train_list in train_lists:
            for seq in train_list:
                for i, item in enumerate(seq):
                    if i>0 :
                        l = min(i, self.Glove_window)
                        for j in range(l):
                            self.co_matrix[item][seq[i-1-j]] += 1
                                               
    def glove_loss(self):
        print("1")
        embed_sim = torch.einsum("ik,jk -> ij",self.item_embed.weight,self.item_embed.weight)#[item_num,item_num]
        print("2")
        bias = self.co_bias.repeat(self.item_num,1) + self.co_bias.repeat(self.item_num,1).transpose(0,1)
        print("3")
        weight = np.clip(self.co_matrix/self.co_max,a_min=0, a_max=1.0)
        print("4")
        loss = torch.tensor(weight) * ((embed_sim + bias - torch.tensor(self.co_matrix)).pow(2))
        print("5")
        loss = torch.sum(loss).cuda()
        return loss
        
        
        '''
        for i in range(self.item_num):
            for j in range(self.item_num):
                if i != j and self.co_matrix[i][j]>0:
                    weight = min(torch.tensor(1.0), self.co_matrix[i][j]/self.co_max)
                    e1 = self.item_embed(torch.tensor(i).cuda())
                    e2 = self.item_embed(torch.tensor(j).cuda())
                    b1 = self.co_bias[i]
                    b2 = self.co_bias[j]
                    square_loss = (torch.einsum("i,i",e1,e2) + b1 + b2 -self.co_matrix[i][j]).pow(2)
                    loss += weight*square_loss
        '''
            
                        
                