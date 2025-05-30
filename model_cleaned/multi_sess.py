import numpy as np
import torch
import math
import os
from torch import nn
from torch.nn import Module, Parameter
from model_cleaned.ggnn import InOutGGNN
from model_cleaned.InOutGat import InOutGATConv
from torch_geometric.nn.conv import GATConv, GCNConv, SGConv
from torch_geometric.nn import global_mean_pool
from model_cleaned.link_pred import L0_Hard_Concrete, LinkPred, construct_pred_edge
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_scatter import scatter
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

def draw_graph(nodes_tensor, edges_tensor, weights_tensor=None):
    nodes = nodes_tensor.cpu().flatten().tolist()
    edges = list(zip(edges_tensor[0].cpu().tolist(), edges_tensor[1].cpu().tolist()))
    if weights_tensor is None:
        weights = [0] * len(edges)
    else:
        weights = weights_tensor.cpu().flatten().tolist()

    G = nx.DiGraph()  # 创建一个有向图
    G.add_nodes_from(nodes)  # 添加节点
    # 添加带权重的边
    for i, (u, v) in enumerate(edges):
        G.add_edge(u, v, weight=weights[i])
    # 绘制图
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    # 绘制边的权重
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # 显示图形
    plt.title("Connection Graph")
    plt.axis('off')
    plt.show()

class LightGCNConv(MessagePassing):
    def __init__(self):
        super(LightGCNConv, self).__init__(aggr='add') 
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index) 
        row, col = edge_index  
        deg = degree(col, x.size(0), dtype=x.dtype)  
        #eps = 1e-10
        #deg_inv_sqrt = (deg + eps).pow(-0.5)
        deg_inv_sqrt = deg.pow(-0.5) 
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j   

class LightGCN(torch.nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(LightGCN, self).__init__()
        self.convs = torch.nn.ModuleList([LightGCNConv() for _ in range(num_layers)])
        self.hidden_size = hidden_size
    def forward(self, x, edge_index):
        all_embeddings = [x]# 存储每层的输出
        for conv in self.convs:
            x = conv(x, edge_index)  # 每一层的前向传播
            all_embeddings.append(x)  # 保存每层的输出
        return torch.mean(torch.stack(all_embeddings), dim=0)# 对所有层的嵌入进行平均



class GroupGraph(Module):
    def __init__(self, n_node, hidden_size, dropout=0.5, negative_slope=0.2, heads=8, item_fusing=False, l0_para=None):
        super(GroupGraph, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_edge = nn.Embedding(n_node, self.hidden_size)  
        self.item_fusing = item_fusing
        self.global_dropout = dropout
        self.l0_para = eval(l0_para) 
        self.linkp = LinkPred(self.hidden_size, self.hidden_size//2, self.l0_para)
        self.sgcn = SGConv(in_channels=hidden_size, out_channels=hidden_size, K=2, add_self_loops=False)
        self.lightgcn = LightGCN(hidden_size=hidden_size, num_layers=1)
        self.gated = GatedGraphConv(self.hidden_size, num_layers=1)  
        self.gru = nn.GRUCell(1 * self.hidden_size, self.hidden_size)


    def rebuilt_sess(self, session_embedding, node_num, sess_item_index, seq_lens):
        split_embs = torch.split(session_embedding, tuple(node_num))  
        sess_item_index = torch.split(sess_item_index, tuple(seq_lens.cpu().numpy()))
        rebuilt_sess = []
        for embs, index in zip(split_embs, sess_item_index):
            sess = tuple(embs[i].view(1, -1) for i in index)
            sess = torch.cat(sess, dim=0)
            rebuilt_sess.append(sess)
        return tuple(rebuilt_sess)

    def forward(self, i,epoch, hidden, data, local_sess_avg, rebuilt_last_item ,is_training=True): 

        mt_edge_index,mt_edge_count, mt_batch, node_num,  mt_sess_item_index, all_edge_label, mt_sess_masks, seq_lens= \
            data.edge_index, data.mt_edge_count, data.batch, data.mt_node_num, data.mt_sess_item_idx, data.all_edge_label, data.mt_sess_masks, data.sequence_len
        l0_penalty_ = 0
        positive_edge_index = mt_edge_index[:, all_edge_label == 1] 
        positive_edge_count = mt_edge_count[all_edge_label == 1]
        pr_edge_index = mt_edge_index[:, all_edge_label == 0]
        pr_edge_count = mt_edge_count[all_edge_label == 0]
        global_forward_weight = torch.ones(pr_edge_index.shape[1]).unsqueeze(-1)  # mt_edge_count.unsqueeze(-1) 
        s, l0_penalty_, probability4fea = self.linkp(hidden,  pr_edge_index, global_forward_weight, local_sess_avg, 
                                    mt_batch, is_training,use_forward=True) 
        pred_edge_index, pred_edge_weight = construct_pred_edge(pr_edge_index, s, pr_edge_count)  
        new_edge_index = torch.cat((positive_edge_index, pred_edge_index), dim=1) 
        new_edge_count = torch.cat((positive_edge_count,pred_edge_weight))
        new_edge_index = torch.unique(new_edge_index, dim=1)
        use_undirected = False  # True #  
        if use_undirected:
            reverse_edge_index = torch.flip(new_edge_index, [0])
            undirected_edge_index = torch.cat([new_edge_index, reverse_edge_index], dim=1)
            mt_edge_index = undirected_edge_index
        else:
            mt_edge_index = new_edge_index


        alternative = 2  #选择全局图中信息聚合的方式
        if alternative == 1:
            hidden = self.sgcn(hidden, mt_edge_index)
        elif alternative == 2:
            neibor_hidden = self.lightgcn(hidden, mt_edge_index)   
            use_self_loops = True#False#
            if use_self_loops == True:
                hidden = neibor_hidden
            else:
                hidden = self.W_4(torch.cat([hidden,neibor_hidden],dim=-1))
        use_dropout = True#False# 
        if use_dropout == True:
            hidden = F.dropout(hidden, self.global_dropout, training=self.training) 
        sess_hidden = self.rebuilt_sess(hidden, node_num, mt_sess_item_index, seq_lens)

        if self.item_fusing:
            return hidden, sess_hidden, l0_penalty_
        else:
            return hidden, self.get_h_group(sess_hidden, seq_lens), l0_penalty_
