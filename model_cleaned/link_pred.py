import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
class L0_Hard_Concrete(nn.Module):  #稀疏化正则化方法。它通过引入一个正则化项来强制某些网络权重趋近于零，从而实现模型的稀疏化
    def __init__(self, temp, inter_min, inter_max):
        super(L0_Hard_Concrete, self).__init__()
        self.temp = temp
        self.inter_min = inter_min
        self.inter_max = inter_max  #定义输出范围的最小值和最大值，用于缩放激活值。
        self.hardtanh = nn.Hardtanh(0, 1) #硬tanh激活函数，将输入值截断到 [0, 1] 范围
        self.pdist = nn.PairwiseDistance(p=1) #计算L1范数的成对距离

    def perm_distance(self, s):
        index_tensor = torch.tensor(range(s.shape[0]))
        index_comb = torch.combinations(index_tensor)
        perm_s = s[index_comb]
        s_1 = (perm_s[:, 0, :])
        s_2 = (perm_s[:, 1, :])
        return self.pdist(s_1, s_2)

    def forward(self, loc, is_training):  #原版本
        if is_training:
            u = torch.rand_like(loc)
            logu = torch.log2(u)
            logmu = torch.log2(1 - u)
            sum_log = loc + logu - logmu
            s = torch.sigmoid(sum_log / self.temp)
            s = s * (self.inter_max - self.inter_min) + self.inter_min  #缩放到 [self.inter_min, self.inter_max] 范围内。
        else:
            s = torch.sigmoid(loc) * (self.inter_max - self.inter_min) + self.inter_min
        # s = torch.where(s < 0, torch.zeros_like(s), s)  # 将负值转换为 0
        # s = torch.where(s > 1, torch.ones_like(s), s)  # 将超过 1 的值转换为 1
        s = torch.clamp(s, min=0, max=1)  #将 s 的值截断在 [0, 1] 范围内
        #s = self.hardtanh(s)
        l0_matrix = torch.sigmoid(loc - self.temp * np.log2(-self.inter_min / self.inter_max))
        # original penalty
        l0_penaty = l0_matrix.mean()
        return s, l0_penaty  #s为稀疏化激活值，l0_penaty：L0正则化项，用于稀疏化正则化

    # def forward(self, loc, miu,logsigam2, is_training):  #DGCD版本
    #     if is_training:
    #         u = torch.rand_like(loc)
    #         logu = torch.log2(u)
    #         logmu = torch.log2(1 - u)
    #         w = torch.sigmoid(loc)
    #         logw = torch.log2(w)
    #         logmw = torch.log2(1 - w)
    #         sum_log = logw - logmw + logu - logmu
    #         s = torch.sigmoid(sum_log / self.temp)
    #         #s = s * (self.inter_max - self.inter_min) + self.inter_min  #缩放到 [self.inter_min, self.inter_max] 范围内。
    #     else:
    #         w = torch.sigmoid(loc)
    #         logw = torch.log2(w)
    #         logmw = torch.log2(1 - w)
    #         sum_log = logw - logmw
    #         s = torch.sigmoid(sum_log)
    #         #s = torch.sigmoid(loc) * (self.inter_max - self.inter_min) + self.inter_min
    #     s = torch.where(s < 0, torch.zeros_like(s), s)  # 将负值转换为 0
    #     s = torch.where(s > 1, torch.ones_like(s), s)  # 将超过 1 的值转换为 1
    #     s = torch.clamp(s, min=0, max=1)  #将 s 的值截断在 [0, 1] 范围内
    #     #s = self.hardtanh(s)
    #     #l0_matrix = torch.sigmoid(loc - self.temp * np.log2(-self.inter_min / self.inter_max))
    #     l0_matrix = (miu**2 + torch.exp(logsigam2) - logsigam2 - 1)/2
    #     # original penalty
    #     l0_penaty = l0_matrix.mean()
    #     return s, l0_penaty  #s为稀疏化激活值，l0_penaty：L0正则化项，用于稀疏化正则化

class LinkPred(nn.Module):
    def __init__(self, D_in, H, l0_para):
        """
        D_in 为hidden_size
        H 为 hidden_size /2
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(LinkPred, self).__init__()
        # self.mlp_forward = nn.Sequential(
        #                 nn.Linear(D_in * 1, H, bias=False), #这里的维度要根据下面的形式进行相应修改！
        #                 nn.ReLU()
        #                 nn.Linear(H, 1, bias=False),
        #                 )
        self.mlp_forward = nn.Linear(D_in * 1, 1, bias=False) #改为只要线性层
        self.W_miu = nn.Linear(D_in * 2, 1, bias=False) #改为只要线性层
        self.W_sigma = nn.Linear(D_in * 2, 1, bias=False) #改为只要线性层
        self.mlp_backward = nn.Sequential(
                        nn.Linear(D_in * 1, H),
                        nn.ReLU(),
                        nn.Linear(H, 1))
        self.sigmoid = nn.Sigmoid()
        self.We = nn.Linear(D_in * 2, D_in,bias=False)
        self.W_1 = nn.Linear(D_in * 1, D_in,bias=True)
        self.W_2 = nn.Linear(D_in * 1, D_in,bias=False)
        self.l0_binary = L0_Hard_Concrete(*l0_para)   #初始化为一个 L0_Hard_Concrete 对象
    def forward(self, feature_emb_edge, edge_index, global_edge_weight,avg,batch, is_training, use_forward=True):
        sender_emb4fea = feature_emb_edge[edge_index[0]] #源节点的embedding
        receiver_emb4fea = feature_emb_edge[edge_index[1]]  #目标节点的embedding  sender_receiver包含每个特征在总特征中的索引
        input_alter = 3
        if input_alter == 1:
            _input4fea = sender_emb4fea * receiver_emb4fea  #①点积后再拼接    改mlp的线性层维度为D_in*2
        elif input_alter == 2:
            _input4fea = torch.cat((sender_emb4fea,receiver_emb4fea),dim=-1)   #②改为拼接，改mlp的线性层维度为D_in*3
        elif input_alter == 3:
            _input4fea = self.We(torch.cat((sender_emb4fea,receiver_emb4fea),dim=-1))   #③拼接后线性变换到d维，改mlp的线性层维度为D_in*2
        elif input_alter == 4:  #直接利用发送节点的表示
            _input4fea = sender_emb4fea
        # 获取每条边的会话编号
        sender_edge_sessions = batch[edge_index[0]]  # 假设边的源节点和目标节点属于同一个会话
        rece_edge_sessions = batch[edge_index[1]]  # 假设边的源节点和目标节点属于同一个会话
        # valid_edges_mask = (sender_edge_sessions == rece_edge_sessions)
        # valid_edge_index = edge_index[:, valid_edges_mask]
        #获取每条边对应的会话向量
        session_emb4fea = avg[sender_edge_sessions]
        #（记得选择和平均会话表示之间的拼接方式！ →改相应的线性层的维度）
        final_input_alter = 2#或者能不能像形成会话表示时计算注意力那样，用各项相加：④[3]
        if final_input_alter == 1:
            _input4fea = torch.cat((_input4fea, session_emb4fea), dim=-1) # [1]拼接会话平均表示
        elif final_input_alter == 2:
            _input4fea = _input4fea * session_emb4fea                      # [2]哈达玛积会话平均表示
        elif final_input_alter == 3:
            _input4fea = self.W_1(_input4fea) + self.W_2(session_emb4fea)   #模仿GNN-GNF之后还要不要leakyrelu一下
            _input4fea = F.leaky_relu(_input4fea)
        if use_forward:
            probability4fea = self.mlp_forward(_input4fea)
            # probability4fea = self.mlp_forward(_input4fea * global_edge_weight)  # 0730预测边时，给边赋予权重
            # probability4fea = self.mlp_forward(torch.cat((_input4fea,global_edge_weight),dim=-1)) #   0730预测边时，给边赋予权重
            # miu = self.W_miu(_input4fea)
            # logsigma2 = self.W_sigma(_input4fea)
            # sigma = torch.exp(logsigma2) ** 0.5
            # epsilon = torch.randn_like(miu)
            # probability4fea = miu + epsilon * sigma
        else:
            probability4fea = self.mlp_backward(_input4fea)

        #probability4fea = self.mlp(_input4fea)   #对输入特征进行变换，得到预测概率 probability4fea
        s4fea, l0_penaty4fea = self.l0_binary(probability4fea, is_training) #原版，对预测概率进行稀疏化，得到稀疏化后的特征s4fea和L0正则化项
        #s4fea, l0_penaty4fea = self.l0_binary(probability4fea, miu,logsigma2,is_training) #DGCD版本
        #通过正则化步骤，模型会强制某些边的概率趋近于零，从而实现边的稀疏化
        return s4fea, l0_penaty4fea, probability4fea   # s4fea：经过 L0 稀疏化后的特征，表示通过 L0 正则化后的边的存在概率。l0_penaty4fea：L0 正则化项。

def construct_pred_edge(fe_index, s, pr_edge_count):
    s = s.squeeze()
    weight = torch.ones_like(s)   #[E,]
    # 找到s中大于0的元素所对应的接收者和发送者
    pos_indices = torch.nonzero(s > 0).squeeze(-1)
    # 根据非零元素的索引获取发送者和接收者
    sender = fe_index[0][pos_indices]
    receiver = fe_index[1][pos_indices]
    pred_index = torch.stack((sender, receiver), dim=0)        # 构建新的边索引
    use_s_as_weight =False# True#
    use_weight_as_weight = not use_s_as_weight
    if use_s_as_weight:
        pred_weight = s[pos_indices]   #s（计算出来的边概率） or weight（采用步数倒数作为边的权重） ！！
    else:
        pred_weight = pr_edge_count[pos_indices]
    return pred_index,  pred_weight