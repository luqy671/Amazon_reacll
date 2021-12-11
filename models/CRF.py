import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import faiss
import numpy as np  
import os
from models.MyModel import MyModel_v3 


class CRF_v3(MyModel_v3):
    
    def __init__(self, config):
        super(CRF_v3, self).__init__(config)
        
        self.state_matrix = nn.Parameter(torch.randn(self.Gene_num, self.Gene_num))
        print("init CRF Modle")

        
    def cal_crf_score(self, indicex_K, item_embeds): #[batch, intere_num] [batch, num, embed_dim] 
        batch = item_embeds.shape[0]
        num = item_embeds.shape[1]
        # 计算每个embedding与Gene_pool中vec的相似度
        Gene_sim_score = torch.einsum('ijd, kd -> ijk', item_embeds, self.Gene_pool) #[batch,num,Gene_num]
        # 对每个batch内的得分，降序排列
        sorted_score, indices = torch.sort(Gene_sim_score, descending = True)  #[batch, num,Gene_num]
        # 取 top 1
        indices_top = indices[:, : , 0]  #[batch, num]      
        
        x = indices_top.unsqueeze(-1).repeat(1,1,1,self.intere_num).reshape(batch,-1)
        y = indicex_K.repeat(1,1,num).reshape(batch,-1)
        
        coordinate = [[a,b] for a,b in zip(x, y)]
        crf_score = torch.cat([self.state_matrix[co].reshape(num,-1).sum(dim=-1) for co in coordinate]).reshape(batch,-1) #[batch,num]
        
        return crf_score