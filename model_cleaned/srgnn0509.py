import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
        self.gated = GatedGraphConv(self.hidden_size, num_layers=1)
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

    def forward(self,data, hidden, local_edge_index,local_edge_count,mt_batch, local_sess_avg,rebuilt_last_item,is_training, i):
        mt_sess_item_index, seq_len,  mt_sess_masks, local_num_count= \
            data.mt_sess_item_idx, data.sequence_len, data.mt_sess_masks, data.local_num_count

        use_self_loops = True#  False#
        if use_self_loops == True:
            # hidden = self.gated(hidden, local_edge_index)  #此行直接使用GGNN
            unique_node = torch.unique(local_edge_index).unsqueeze(0)
            edge_index_self = torch.cat((unique_node, unique_node), dim=0)
            edge_index_with_loops = torch.cat([local_edge_index, edge_index_self], dim=1)
            local_edge_index = torch.unique(edge_index_with_loops, dim=1)
            self_loop_weight = torch.ones(local_edge_index.size(1) - local_edge_count.size(0)).to(local_edge_count.device)
            local_edge_count = torch.cat([local_edge_count, self_loop_weight])

            neibor_hidden = self.local_agg(data,hidden, local_edge_index,local_sess_avg, rebuilt_last_item,
                                           mt_sess_masks , i,mt_batch, local_edge_count, local_num_count, is_training)
            hidden = neibor_hidden
        else:
            neibor_hidden = self.local_agg(data,hidden, local_edge_index, local_sess_avg, rebuilt_last_item,
                                           mt_sess_masks, i,  mt_batch,local_edge_count, local_num_count, is_training) 
            new_hidden = self.gru(neibor_hidden, hidden)    

            hidden = new_hidden

        sess_embs = self.rebuilt_sess(hidden, mt_batch, mt_sess_item_index, seq_len)
        if self.item_fusing:
            return hidden, sess_embs
        else:
            return hidden, self.get_h_s(sess_embs, seq_len)

####################################################################################################
class LocalAggregator(MessagePassing):  
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

        update_nodes = forward_nodes
        return update_nodes

    def message(self, data,x, x_i, x_j, edge_index_i, edge_index_j, local_sess_avg,rebuilt_last_item,mt_sess_masks, batch, edge_weight, node_frequency,is_training,flow, use_forward):
        edge_index = torch.stack((edge_index_j, edge_index_i), dim=0)
        mask = edge_index[0] != edge_index[1]
        edges_to_expand = edge_index[:, mask]
        symmetric_edges = torch.stack([edges_to_expand[1], edges_to_expand[0]], dim=0)
        expanded_edge_index = torch.cat([edge_index, symmetric_edges], dim=1)
        expanded_x_j = x[expanded_edge_index[0]]  
        expanded_x_i = x[expanded_edge_index[1]] 
        reversed_edge_index_j = symmetric_edges[0]
        reversed_edge_index_i = symmetric_edges[1]
        new_edge_index_i = expanded_edge_index[1]
        new_edge_index_j = expanded_edge_index[0]
        use_session_mean_as_alpha_query = True#False#
        session_mean_i = local_sess_avg[batch[new_edge_index_i]]   
        alpha_query = session_mean_i
        alpha = F.leaky_relu(self.W_alpha_attention(expanded_x_i * alpha_query)) 
        beta = (F.leaky_relu(self.W_forward(x_i * x_j)))
        reversed_beta = (F.leaky_relu(self.W_backward(x[reversed_edge_index_j] * x[reversed_edge_index_i])))
        new_beta = torch.cat([beta,reversed_beta])
        final_E = new_beta  + alpha  
        aij = softmax(final_E, new_edge_index_i)  # [Ei,1]  
        pairwise_analysis = (aij * (expanded_x_j))
        return pairwise_analysis, reversed_edge_index_i

    def aggregate(self, inputs, edge_index_i, edge_index_j):
        input, reversed_edge_index_i = inputs
        index = torch.cat([edge_index_i, reversed_edge_index_i], dim=-1).unsqueeze(-1)
        unique_value, new_index = torch.unique(index, return_inverse=True)
        aggr_out = scatter(input, new_index, dim=0, reduce=self.aggr)
        return  aggr_out

    def update(self, aggr_out,x, mt_sess_masks,data):   #更新节点特征
        new_x = torch.zeros_like(x)
        new_x[mt_sess_masks==1] = aggr_out
        new_x[mt_sess_masks==0] = x[mt_sess_masks==0]
        return new_x + x
