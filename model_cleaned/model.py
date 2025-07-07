from model_cleaned.multi_sess import GroupGraph
from model_cleaned.srgnn import SRGNN

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class Embedding2Score(nn.Module):
    def __init__(self, hidden_size, n_node, using_represent, item_fusing, scale):
        super(Embedding2Score, self).__init__()
        self.hidden_size = hidden_size
        self.n_node = n_node
        self.using_represent = using_represent
        self.item_fusing = item_fusing
        self.scale = scale

    def forward(self, h_s, h_group, final_s, item_embedding_table):
        emb = item_embedding_table.weight.transpose(1, 0)
        if self.item_fusing:
            emb =  F.normalize(emb, dim=0,p=2)
            final_s = self.scale * F.normalize(final_s, dim=-1,p=2)
            scores = torch.matmul(final_s, emb)
            z_i_hat = scores
        else:
            gate = F.sigmoid(self.W_2(h_s) + self.W_3(h_group))
            sess_rep = h_s * gate + h_group * (1 - gate)
            if self.using_represent == 'comb':
                z_i_hat = torch.mm(sess_rep, emb)
            elif self.using_represent == 'h_s':
                z_i_hat = torch.mm(h_s, emb)
            elif self.using_represent == 'h_group':
                z_i_hat = torch.mm(h_group, emb)
            else:
                raise NotImplementedError
        return z_i_hat

class CNNFusing(nn.Module):
    def __init__(self, hidden_size, num_filters, max_len):
        super(CNNFusing, self).__init__()
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_1_inter = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2_inter = nn.Linear(self.hidden_size, self.hidden_size)
        self.q_inter = nn.Linear(self.hidden_size, 1)
        self.position_emb = nn.Embedding(max_len+1, self.hidden_size)
        self.W_pos_inter = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, intra_item_emb, inter_item_emb, seq_len, reverse_pos, session_features):

        final_s = self.get_final_s_GCE_GNN(final_emb, seq_len, reverse_pos)
        return final_s


    def get_final_s_GCE_GNN(self, hidden, seq_len,reverse_pos): 
        v_mean_repeat = tuple(nodes.mean(dim=0, keepdim=True).repeat(nodes.shape[0], 1) for nodes in hidden)
        v_mean_repeat_concat = torch.cat(v_mean_repeat, dim=0)
        hidden = torch.cat(hidden, dim=0)
        distinguish = True   
        pos_emb = self.position_emb(reverse_pos)  
        pos_hidden = torch.tanh(self.W_pos_inter(torch.cat([hidden, pos_emb], -1)))  
        alpha = self.q_inter(torch.sigmoid(self.W_1_inter(v_mean_repeat_concat) + self.W_2_inter(pos_hidden))) 
        s_g_whole = alpha * hidden
        s_g_split = torch.split(s_g_whole, tuple(seq_len.cpu().numpy())) 
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)
        h_s = torch.cat(s_g, dim=0)
        return h_s


class GraphModel(nn.Module):
    def __init__(self, opt, n_node,max_len):
        super(GraphModel, self).__init__()
        #定义属性变量
        self.hidden_size, self.n_node, self.max_len = opt.hidden_size, n_node, max_len
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        #self.embedding_edge = nn.Embedding(self.n_node, self.hidden_size)  
        self.length_emb = nn.Embedding(self.max_len+1, self.hidden_size) 
        self.dropout = opt.global_dropout
        self.negative_slope = opt.negative_slope
        self.heads = opt.heads
        self.item_fusing = opt.item_fusing
        self.num_filters = opt.num_filters
        self.use_alternative_coo = opt.use_alternative_coo
        self.use_alternative_final_s = opt.use_alternative_final_s
        #定义线性层 or 自学习参数
        self.W_coo_15 = nn.Linear(2 * self.hidden_size, self.hidden_size)  # （注意输出维度的设置 1 or self.hidden_size）
        self.W_concat = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.W_sum_1 = nn.Linear(self.hidden_size, 1) # （注意输出维度的设置 1 or self.hidden_size）
        self.W_sum_2 = nn.Linear(self.hidden_size, 1) # （注意输出维度的设置 1 or self.hidden_size）
        self.W_Q = nn.Linear(self.hidden_size , self.hidden_size,bias=False)
        self.W_K = nn.Linear(self.hidden_size * 1, self.hidden_size,bias=False)
        self.W_con = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.W_self = nn.Linear(1 * self.hidden_size, self.hidden_size,bias=False)
        self.W_neibor = nn.Linear(1 * self.hidden_size, self.hidden_size,bias=False)
        self.W_gated_concat = nn.Linear(2 * self.hidden_size, 1,bias=False)  
        self.W_gated_self = nn.Linear(1 * self.hidden_size, 1,bias=False)  
        self.W_gated_neibor = nn.Linear(1 * self.hidden_size, 1,bias=False) 
        self.a1 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.a2 = nn.Parameter(torch.randn(1), requires_grad=True)
        #定义模型相关模块
        self.srgnn = SRGNN(self.hidden_size, n_node=n_node, item_fusing=opt.item_fusing)
        self.group_graph = GroupGraph( self.n_node, self.hidden_size, dropout=self.dropout, negative_slope=self.negative_slope,
                                      heads=self.heads, item_fusing=opt.item_fusing, l0_para=opt.l0_para)
        self.cnn_fusing = CNNFusing(self.hidden_size, self.num_filters, self.max_len)
        self.e2s = Embedding2Score(self.hidden_size, n_node, opt.using_represent, opt.item_fusing, opt.scale)
        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):  
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def rebuilt_sess(self, session_embedding, node_num, sess_item_index, seq_lens):
        split_embs = torch.split(session_embedding, tuple(node_num))
        sess_item_index = torch.split(sess_item_index, tuple(seq_lens.cpu().numpy()))
        rebuilt_sess = []
        for embs, index in zip(split_embs, sess_item_index):
            sess = tuple(embs[i].view(1, -1) for i in index)
            sess = torch.cat(sess, dim=0)
            rebuilt_sess.append(sess)
        return tuple(rebuilt_sess)

    def forward(self, data, i, epoch, is_training=True):
        if self.item_fusing:
            mt_x = data.mt_x - 1  
            embedding = self.embedding(mt_x)
            embedding = embedding.squeeze()
            mt_edge_index, mt_edge_count, mt_batch, node_num, mt_sess_item_index, \
                all_edge_label, mt_sess_masks, seq_lens,num_count = \
                data.edge_index, data.mt_edge_count, data.batch, data.mt_node_num, data.mt_sess_item_idx, \
                    data.all_edge_label, data.mt_sess_masks, data.sequence_len, data.num_count
            reverse_pos = data.reverse_pos
            v_i = torch.split(embedding, tuple(node_num.cpu().numpy()))  
            local_xulie_hidden = []
            rebuilt_last_item = []
            local_mt_sess_item_index = torch.split(mt_sess_item_index, tuple(seq_lens.cpu().numpy()))
            for node_indices, v in zip(local_mt_sess_item_index, v_i):
                local_xulie_hidden.append(v[node_indices]) 
                sess = v[node_indices[-1]]
                rebuilt_last_item.append(sess)
            local_xulie_hidden = torch.cat(local_xulie_hidden, dim=0) 
            seq_lens_device = seq_lens.to(local_xulie_hidden.device)
            seq_batch = torch.repeat_interleave(torch.arange(len(seq_lens_device), device=local_xulie_hidden.device),seq_lens_device)
            local_sess_avg = global_mean_pool(local_xulie_hidden, seq_batch)
            rebuilt_last_item = torch.stack(rebuilt_last_item, dim=0)
            #（局部）
            local_edge_index = mt_edge_index[:, all_edge_label == 1]   
            edge_count = data.edge_count  
            intra_hidden, intra_item_emb = self.srgnn(data, embedding,local_edge_index,edge_count,mt_batch, local_sess_avg,rebuilt_last_item,is_training, i)          
            l0_penalty_ = 0  # 全局边预测过程得到的正则化数值
            #（全局）
            inter_hidden, inter_item_emb, l0_penalty_ = self.group_graph.forward(i,epoch,embedding, data, local_sess_avg,rebuilt_last_item ,is_training=is_training)
            final_s_intra = self.cnn_fusing.get_final_s_GCE_GNN(intra_item_emb,data.sequence_len, data.reverse_pos) # tensor(B,dim)
            final_s_inter = self.cnn_fusing.get_final_s_GCE_GNN(inter_item_emb, data.sequence_len, data.reverse_pos) # tensor(B,dim)
            #得到会话个性化的session_features
            use_fusion = True #  
            use_alternative_coo = self.use_alternative_coo#5 # 控制 coo 生成方式的参数
            if use_fusion:
                v_last = torch.stack([session[-1] for session in intra_item_emb])  # (100, 100)
                v_mean = torch.stack([torch.mean(session, dim=0) for session in intra_item_emb])  # (100, 100)
                if use_alternative_coo == 15:
                    session_lengths = torch.tensor([len(session) for session in intra_item_emb], dtype=torch.long,
                                                   device=v_last.device)  # (B,)
                    coo_input_1 = v_mean - v_last
                    coo_input = torch.cat([coo_input_1, self.length_emb(session_lengths)],dim=-1)  # (B, 2*dim )
                    coo = torch.sigmoid(self.W_coo_15(coo_input))#.view(-1, 1) 
            use_alternative_final_s = self.use_alternative_final_s#2  #final_s的拼接方式选择
            if use_alternative_final_s == 1:
                final_s = session_features * final_s_inter + (1 - session_features) * final_s_intra
            elif use_alternative_final_s == 2:  #倒置
                final_s = session_features * final_s_intra + (1 - session_features) * final_s_inter
            elif use_alternative_final_s == 3: 
                final_s = self.W_concat(torch.cat([final_s_intra,final_s_inter],dim=-1))
            elif use_alternative_final_s == 4:
                final_s = final_s_intra + final_s_inter
            elif use_alternative_final_s == 5:
                final_s = torch.max(final_s_intra,final_s_inter)
           

            ####################################################################################
            #计算预测分数
            scores = self.e2s(h_s=None, h_group=None, final_s=final_s, item_embedding_table=self.embedding)

        else:
            x = data.x - 1
            embedding = self.embedding(x)
            embedding = embedding.squeeze()
            h_s = self.srgnn(data, embedding)
            mt_x = data.mt_x - 1
            embedding = self.embedding(mt_x)
            embedding = embedding.squeeze()
            h_group = self.group_graph.forward(embedding, data)
            scores = self.e2s(h_s=h_s, h_group=h_group, final_s=None, item_embedding_table=self.embedding)

        return scores,  l0_penalty_

