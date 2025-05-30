import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_cleaned.ggnn import InOutGGNN
from model_cleaned.InOutGat import InOutGATConv, InOutGATConv_intra
from torch_geometric.nn import GCNConv, GATConv, SGConv, GatedGraphConv, global_mean_pool
from torch_geometric.utils import softmax, add_self_loops
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from datetime import datetime

class SRGNN(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """
    def __init__(self, hidden_size, n_node, dropout=0.5, negative_slope=0.2, heads=8, item_fusing=False):
        super(SRGNN, self).__init__()
        self.hidden_size, self.n_node = hidden_size, n_node
        self.item_fusing = item_fusing
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gated = GatedGraphConv(self.hidden_size, num_layers=1)  # self.hidden_size：定义了输出特征的维度 （层数问题！）
        self.local_agg = LocalAggregator(self.hidden_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def rebuilt_sess(self, session_embedding, batchs, sess_item_index, seq_lens):
        sections = torch.bincount(batchs)
        split_embs = torch.split(session_embedding, tuple(sections.cpu().numpy()))
        sess_item_index = torch.split(sess_item_index, tuple(seq_lens.cpu().numpy()))
        rebuilt_sess = []
        for embs, index in zip(split_embs, sess_item_index):
            sess = tuple(embs[i].view(1, -1) for i in index)
            sess = torch.cat(sess, dim=0)
            rebuilt_sess.append(sess)
        return tuple(rebuilt_sess)


    def labs(self,edge_index):  #是用来模拟GCE-GNN中的双向边关系
        transposed_edges = edge_index.t()  # 转置 edge_index
        edges_set = {tuple(edge.tolist()) for edge in transposed_edges} # 使用集合来存储所有边
        labels = torch.zeros(transposed_edges.size(0)).to(edge_index.device) # 初始化标签张量
        # 遍历每一条边，检查是否存在其反向边
        for i, (src, dst) in enumerate(transposed_edges):
            if src == dst:
                continue
            # 查找是否存在反向边
            if (dst.item(), src.item()) in edges_set:
                labels[i] = 1
        return labels


    def forward(self,data, hidden, local_edge_index,local_edge_count,mt_batch, local_sess_avg,rebuilt_last_item,is_training, i):
        mt_sess_item_index, seq_len,  mt_sess_masks, local_num_count= \
            data.mt_sess_item_idx, data.sequence_len, data.mt_sess_masks, data.local_num_count

        use_self_loops = True#  False#
        if use_self_loops == True:
            # # 此行直接使用GGNN
            # hidden = self.gated(hidden, local_edge_index)  #此行直接使用GGNN

            #添加自环
            unique_node = torch.unique(local_edge_index).unsqueeze(0)
            edge_index_self = torch.cat((unique_node, unique_node), dim=0)
            edge_index_with_loops = torch.cat([local_edge_index, edge_index_self], dim=1)
            local_edge_index = torch.unique(edge_index_with_loops, dim=1)

            #local_shuangxiang_labs = self.labs(local_edge_index) #放在mt_batch前

            self_loop_weight = torch.ones(local_edge_index.size(1) - local_edge_count.size(0)).to(local_edge_count.device)
            local_edge_count = torch.cat([local_edge_count, self_loop_weight])

            neibor_hidden = self.local_agg(data,hidden, local_edge_index,local_sess_avg, rebuilt_last_item,
                                           mt_sess_masks , i,mt_batch, local_edge_count, local_num_count, is_training)
            hidden = neibor_hidden
        else:
            neibor_hidden = self.local_agg(data,hidden, local_edge_index, local_sess_avg, rebuilt_last_item,
                                           mt_sess_masks, i,  mt_batch,local_edge_count, local_num_count, is_training)  # 局部会话中进行的项目表示更新
            new_hidden = self.gru(neibor_hidden, hidden)     # 考虑其他多种结合方式

            hidden = new_hidden


        sess_embs = self.rebuilt_sess(hidden, mt_batch, mt_sess_item_index, seq_len)
        # #下一行self.rebuilt_sess_and_last函数可以同时返回序列空间的最后一项
        # sess_embs, sess_embs_last = self.rebuilt_sess_and_last(hidden, mt_batch, mt_sess_item_index, seq_len)
        if self.item_fusing:
            return hidden, sess_embs
        else:
            return hidden, self.get_h_s(sess_embs, seq_len)

####################################################################################################


#2*beta 双向邻居，使用不同的线性层  （原版）
class LocalAggregator(MessagePassing):  #beta new_beta
    def __init__(self, dim):
        super(LocalAggregator, self).__init__(aggr='add')   #指定聚合方式为求和
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dim = dim

        self.W_forward = nn.Linear(1*self.dim*1 + 0, 1, bias=False)  # 调整维度
        self.W_backward = nn.Linear(1*self.dim*1 + 0, 1, bias=False) # 调整维度
        self.W_alpha_attention = nn.Linear(1*self.dim, 1, bias=False)


    def forward(self, data,x, edge_index, local_sess_avg,rebuilt_last_item, mt_sess_masks, i , batch=None, edge_weight=None, node_frequency=None, is_training = None,  use_forward=True):
        # Step 1: Source-to-Target message passing
        self.current_i = torch.tensor([i], dtype=torch.float, device=x.device)
        forward_nodes = self.propagate(
            edge_index=edge_index, x=x,data=data,
            local_sess_avg=local_sess_avg,rebuilt_last_item=rebuilt_last_item,
            mt_sess_masks=mt_sess_masks,
            batch=batch,edge_weight=edge_weight,
            node_frequency=node_frequency,
            is_training = is_training,
            flow="source_to_target", use_forward=True)


        # Combine the results (e.g., summing forward and backward messages)
        update_nodes = forward_nodes
        #update_nodes = forward_nodes + backward_nodes
        #update_nodes = torch.cat([forward_nodes, backward_nodes], dim=-1)
        #update_nodes = self.propagate(edge_index=edge_index, x=x, oral_x=x, session_mean=session_mean, last_item_embeddings=last_item_embeddings, batch=batch, edge_weight=edge_weight, node_frequency=node_frequency, flow="source_to_target", use_forward=use_forward)
        return update_nodes
        # forward 方法中调用 self.propagate 时，会依次执行 message、aggregate 和 update 方法

    def message(self, data,x, x_i, x_j, edge_index_i, edge_index_j, local_sess_avg,rebuilt_last_item,mt_sess_masks, batch, edge_weight, node_frequency,is_training,flow, use_forward):
        edge_index = torch.stack((edge_index_j, edge_index_i), dim=0)
        # 找出需要对称扩展的边（发送节点 != 接收节点）
        mask = edge_index[0] != edge_index[1]
        edges_to_expand = edge_index[:, mask]
        # 创建对称边（交换源节点和目标节点）
        symmetric_edges = torch.stack([edges_to_expand[1], edges_to_expand[0]], dim=0)
        # 合并原始边和新创建的对称边
        expanded_edge_index = torch.cat([edge_index, symmetric_edges], dim=1)
        # 获取扩展后边的发送节点特征 (x_j) 和接收节点特征 (x_i)
        expanded_x_j = x[expanded_edge_index[0]]  # 发送节点特征
        expanded_x_i = x[expanded_edge_index[1]]  # 接收节点特征

        reversed_edge_index_j = symmetric_edges[0]
        reversed_edge_index_i = symmetric_edges[1]
        new_edge_index_i = expanded_edge_index[1]
        new_edge_index_j = expanded_edge_index[0]

        # #0509增 (记得和下一段0509同步）
        # index = 3  # 画第几个图
        # i_int = int(self.current_i.item())
        # if not is_training and i_int == 0 :  # 在测试集上画/第几个批次
        #     edge_index_sessions_bianhao = batch[edge_index_i]
        #     counts = torch.bincount(edge_index_sessions_bianhao)
        #     edge_index = torch.stack((edge_index_j, edge_index_i), dim=0)
        #     subgraph_edges = torch.split(edge_index, tuple(counts.cpu().numpy()), dim=1)
        #     new_edge_index_sessions_bianhao = batch[new_edge_index_i] #反向连接后
        #     new_counts = torch.bincount(new_edge_index_sessions_bianhao)
        #     nonzero_counts =  new_counts[new_counts > 0]
        #     sorted_indices = torch.argsort(new_edge_index_sessions_bianhao)
        #     sorted_edges = expanded_edge_index[:, sorted_indices]
        #     new_subgraph_edges = torch.split(sorted_edges, nonzero_counts.tolist(), dim=1)
        #     draw_edges = new_subgraph_edges[index]
        #     sender_x = data.x[draw_edges[0]].squeeze()  # 只画全局图中的跨会话边缘
        #     receiver_x = data.x[draw_edges[1]].squeeze()
        #     draw_edges = torch.stack((sender_x, receiver_x), dim=0)
        #     #创建可视化目录（如果不存在）
        #     os.makedirs('keshihua', exist_ok=True)
        #     np.savetxt('keshihua/local_draw_edges.txt', draw_edges.cpu().numpy().reshape(-1, draw_edges.shape[-1]), fmt='%d')
        # # 0509增(记得和下一段0509同步）

        use_session_mean_as_alpha_query = True#False#
        if use_session_mean_as_alpha_query == True:
            session_mean_i = local_sess_avg[batch[new_edge_index_i]]    # 获取 x_i 所属的会话表示
            alpha_query = session_mean_i
        else:
            last_item_i = rebuilt_last_item[batch[new_edge_index_i]]  # 获取 x_i 所属会话中最后一项的项目表示
            alpha_query =last_item_i

        node_frequency_i = node_frequency[edge_index_i]  # 获取目标节点的频率
        use_alpha = True#False#
        alpha_attention_use_weight =False# True #    #控制是否使用权重（考虑不同的权重定义方法）  ！！
        if use_alpha and alpha_attention_use_weight:
            alpha = F.leaky_relu(self.W_alpha_attention(expanded_x_i * alpha_query * node_frequency_i.unsqueeze(-1)))  #模拟GCE、SUGAR的形式
        elif use_alpha and not alpha_attention_use_weight:
            #alpha = self.W_alpha_attention(F.leaky_relu(self.WW_alpha_attention(new_x_i * alpha_query)))  #模拟GCE、SUGAR的形式
            alpha = F.leaky_relu(self.W_alpha_attention(expanded_x_i * alpha_query))  #模拟GCE、SUGAR的形式
            #alpha = F.leaky_relu(self.W_alpha_attention(new_x_j * alpha_query))  #用new_x_j 替换 i
            #alpha = softmax(alpha, new_edge_index_i)  # 1108可选
        beta = (F.leaky_relu(self.W_forward(x_i * x_j)))
        reversed_beta = (F.leaky_relu(self.W_backward(x[reversed_edge_index_j] * x[reversed_edge_index_i])))
        new_beta = torch.cat([beta,reversed_beta])
        # #下一行1108可选
        # new_beta = softmax(new_beta, new_edge_index_i) # 1108可选
        final_E = new_beta  + alpha    #直接相加
        #final_E = torch.min(new_beta,alpha)    #min处理
        #final_E1 = (new_beta  + alpha)/2 #两者平均
        #final_E = F.leaky_relu(final_E) # 1108可选
        aij = softmax(final_E, new_edge_index_i)  # [Ei,1]   #这里直接用new_edge_index_i


        # # 0509增(记得和上一段0509同步）
        # if not is_training and i_int == 0 :  # 在测试集上画/第几个批次
        #     aij = final_E  # 0526 不softmax以观察权重差别
        #     sorted_weights = aij[sorted_indices]
        #     new_subgraph_weights = torch.split(sorted_weights, nonzero_counts.tolist())
        #     weights_tensor = new_subgraph_weights[index]
        #     # 获取当前时间，格式化为 年-月-日 时:分
        #     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 保留前3位毫秒
        #     with open('keshihua/local_edges_weights_tensor.txt', 'a') as f:
        #         f.write(f"epoch: {current_time}\n")
        #         np.savetxt(f, weights_tensor.cpu().numpy(), fmt='%.4f')
        #         f.write("\n")
        # # 0509增(记得和上一段0509同步）

        pairwise_analysis = (aij * (expanded_x_j))
        return pairwise_analysis, reversed_edge_index_i

    def aggregate(self, inputs, edge_index_i, edge_index_j):
        input, reversed_edge_index_i = inputs
        # 聚合来自邻居的消息
        index = torch.cat([edge_index_i, reversed_edge_index_i], dim=-1).unsqueeze(-1)
        #aggr_out = scatter(inputs, index, dim=0, reduce=self.aggr, dim_size=len(torch.unique(index)))
        unique_value, new_index = torch.unique(index, return_inverse=True)
        aggr_out = scatter(input, new_index, dim=0, reduce=self.aggr)
        return  aggr_out

    def update(self, aggr_out,x, mt_sess_masks,data):   #更新节点特征
        # aggr_out has shape [N, dim]   aggr_out 是聚合后的消息
        new_x = torch.zeros_like(x)
        new_x[mt_sess_masks==1] = aggr_out
        new_x[mt_sess_masks==0] = x[mt_sess_masks==0]
        # local_x =  x[mt_sess_masks == 1]
        # x[mt_sess_masks==1] = aggr_out
        # gate = self.W_gated(torch.sigmoid(self.W_gated_q(x) + self.W_gated_p(new_x)))  #模仿DISEN-GNN中的残差连接
        # new_x = gate * x + (1-gate) * new_x     #模仿DISEN-GNN中的残差连接，直接return new_x可以了 不用再+x
        return new_x + x   #若不+x 则为邻居信息的加和？ #加x有点类似残差？(注意这里的是否加x！）
