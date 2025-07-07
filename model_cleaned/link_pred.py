import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
class L0_Hard_Concrete(nn.Module):  
    def __init__(self, temp, inter_min, inter_max):
        super(L0_Hard_Concrete, self).__init__()
        self.temp = temp
        self.inter_min = inter_min
        self.inter_max = inter_max 
        self.hardtanh = nn.Hardtanh(0, 1) 
        self.pdist = nn.PairwiseDistance(p=1) 

    def perm_distance(self, s):
        index_tensor = torch.tensor(range(s.shape[0]))
        index_comb = torch.combinations(index_tensor)
        perm_s = s[index_comb]
        s_1 = (perm_s[:, 0, :])
        s_2 = (perm_s[:, 1, :])
        return self.pdist(s_1, s_2)

    def forward(self, loc, is_training): 
        if is_training:
            u = torch.rand_like(loc)
            logu = torch.log2(u)
            logmu = torch.log2(1 - u)
            sum_log = loc + logu - logmu
            s = torch.sigmoid(sum_log / self.temp)
            s = s * (self.inter_max - self.inter_min) + self.inter_min 
        else:
            s = torch.sigmoid(loc) * (self.inter_max - self.inter_min) + self.inter_min
        s = torch.clamp(s, min=0, max=1) 
        l0_matrix = torch.sigmoid(loc - self.temp * np.log2(-self.inter_min / self.inter_max)
        l0_penaty = l0_matrix.mean()
        return s, l0_penaty  


class LinkPred(nn.Module):
    def __init__(self, D_in, H, l0_para):
        super(LinkPred, self).__init__()
        self.mlp_forward = nn.Sequential(
                        nn.Linear(D_in * 1, H, bias=False), 
                        nn.Linear(H, 1, bias=False),
                        nn.ReLU()
                        )
        self.sigmoid = nn.Sigmoid()
        self.We = nn.Linear(D_in * 2, D_in,bias=False)
        self.W_1 = nn.Linear(D_in * 1, D_in,bias=True)
        self.W_2 = nn.Linear(D_in * 1, D_in,bias=False)
        self.l0_binary = L0_Hard_Concrete(*l0_para)   
    def forward(self, feature_emb_edge, edge_index, global_edge_weight,avg,batch, is_training, use_forward=True):
        sender_emb4fea = feature_emb_edge[edge_index[0]] 
        receiver_emb4fea = feature_emb_edge[edge_index[1]]  
        input_alter = 3
        if input_alter == 1:
            _input4fea = sender_emb4fea * receiver_emb4fea 
        elif input_alter == 2:
            _input4fea = torch.cat((sender_emb4fea,receiver_emb4fea),dim=-1) 
        elif input_alter == 3:
            _input4fea = self.We(torch.cat((sender_emb4fea,receiver_emb4fea),dim=-1)) 
        elif input_alter == 4: 
            _input4fea = sender_emb4fea
        sender_edge_sessions = batch[edge_index[0]]  
        rece_edge_sessions = batch[edge_index[1]]  
        session_emb4fea = avg[sender_edge_sessions]

        final_input_alter = 2
        if final_input_alter == 1:
            _input4fea = torch.cat((_input4fea, session_emb4fea), dim=-1) 
        elif final_input_alter == 2:
            _input4fea = _input4fea * session_emb4fea                      
        elif final_input_alter == 3:
            _input4fea = self.W_1(_input4fea) + self.W_2(session_emb4fea)  
            _input4fea = F.leaky_relu(_input4fea)
        if use_forward:
            probability4fea = self.mlp_forward(_input4fea)
        else:
            probability4fea = self.mlp_backward(_input4fea)

        s4fea, l0_penaty4fea = self.l0_binary(probability4fea, is_training) 
        return s4fea, l0_penaty4fea, probability4fea   

def construct_pred_edge(fe_index, s, pr_edge_count):
    s = s.squeeze()
    weight = torch.ones_like(s)   #[E,]
    pos_indices = torch.nonzero(s > 0).squeeze(-1)
    sender = fe_index[0][pos_indices]
    receiver = fe_index[1][pos_indices]
    pred_index = torch.stack((sender, receiver), dim=0)       
    use_s_as_weight =False# True#
    use_weight_as_weight = not use_s_as_weight
    if use_s_as_weight:
        pred_weight = s[pos_indices]   
    else:
        pred_weight = pr_edge_count[pos_indices]
    return pred_index,  pred_weight
