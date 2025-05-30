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
    # 将边 Tensor 转换为元组列表
    edges = list(zip(edges_tensor[0].cpu().tolist(), edges_tensor[1].cpu().tolist()))
    if weights_tensor is None:
        # 如果未提供权重张量，默认权重为 0
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
        super(LightGCNConv, self).__init__(aggr='add')  # 聚合操作使用 "加法"
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index)  #如果已有自环，还会添加吗？/ 不会，会自动跳过 /注意：若索引不连续，这个函数有bug
        #未添加自环时的情况：如果某些节点没有边连接（即没有邻居），这些节点的度数为零。 可能会发生报错：NaN or Inf found in input tensor.！
        row, col = edge_index  # 计算每个节点的度数
        deg = degree(col, x.size(0), dtype=x.dtype)  #计算每个节点的入度，即每个节点被多少条边指向。
        #eps = 1e-10
        #deg_inv_sqrt = (deg + eps).pow(-0.5)
        deg_inv_sqrt = deg.pow(-0.5) # 计算邻居节点的度数倒数并进行归一化
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)# 调用 propagate 执行消息传递，进行邻居聚合
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j   # 消息传递过程中的消息，邻居节点特征乘以归一化的度

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
        self.embedding_edge = nn.Embedding(n_node, self.hidden_size)  # 单独为边预测时给一个嵌入表
        self.item_fusing = item_fusing
        self.global_dropout = dropout
        self.l0_para = eval(l0_para)  # l0正则化参数 /使用了 eval 函数来将字符串表示的列表转换为实际的列表对象
        self.linkp = LinkPred(self.hidden_size, self.hidden_size//2, self.l0_para)
        self.sgcn = SGConv(in_channels=hidden_size, out_channels=hidden_size, K=2, add_self_loops=False)
        self.lightgcn = LightGCN(hidden_size=hidden_size, num_layers=1)
        self.gated = GatedGraphConv(self.hidden_size, num_layers=1)  # self.hidden_size：定义了输出特征的维度
        self.gru = nn.GRUCell(1 * self.hidden_size, self.hidden_size)


    def rebuilt_sess(self, session_embedding, node_num, sess_item_index, seq_lens):
        split_embs = torch.split(session_embedding, tuple(node_num))  #节点特征分布到各个全局图中
        sess_item_index = torch.split(sess_item_index, tuple(seq_lens.cpu().numpy()))
        rebuilt_sess = []
        for embs, index in zip(split_embs, sess_item_index):
            sess = tuple(embs[i].view(1, -1) for i in index)
            sess = torch.cat(sess, dim=0)
            rebuilt_sess.append(sess)
        return tuple(rebuilt_sess)

    def forward(self, i,epoch, hidden, data, local_sess_avg, rebuilt_last_item ,is_training=True):  #0508新增 i 表示第几个batch !

        mt_edge_index,mt_edge_count, mt_batch, node_num,  mt_sess_item_index, all_edge_label, mt_sess_masks, seq_lens= \
            data.edge_index, data.mt_edge_count, data.batch, data.mt_node_num, data.mt_sess_item_idx, data.all_edge_label, data.mt_sess_masks, data.sequence_len
        #mt_sess_item_index是自身会话在全局图节点空间的序列索引
        #mt_sess_mask是自身节点在全局图节点空间的掩码

        l0_penalty_ = 0
        #new_edge_count = mt_edge_count

        # #（全局图直接统一删边！！）
        # global_forward_weight = torch.ones(mt_edge_index.shape[1]).unsqueeze(-1)  #mt_edge_count.unsqueeze(-1) # （边数量，1）全局图中边权重的设定方式
        # s, l0_penalty_ = self.linkp(hidden, mt_edge_index, global_forward_weight, local_sess_avg,
        #                             mt_batch, is_training,use_forward=True)  # 边预测   此时边的权重值为自学习出来的？
        # pred_edge_index, pred_edge_weight = construct_pred_edge(mt_edge_index, s, mt_edge_count)  # 提取保留的边/交互
        #
        # use_undirected = False #True #
        # if use_undirected:
        #     reverse_edge_index = torch.flip(pred_edge_index, [0])  # 反转有向边
        #     undirected_edge_index = torch.cat([pred_edge_index, reverse_edge_index], dim=1)
        #     mt_edge_index = undirected_edge_index
        # else:
        #     mt_edge_index = pred_edge_index
        #     new_edge_count = pred_edge_weight
        # #（全局图直接统一删边）

        ###########################################################################################


        # （过滤掉局部边之后再进行删边！！)
        positive_edge_index = mt_edge_index[:, all_edge_label == 1]  #自身会话的边
        positive_edge_count = mt_edge_count[all_edge_label == 1]
        pr_edge_index = mt_edge_index[:, all_edge_label == 0]
        pr_edge_count = mt_edge_count[all_edge_label == 0]

        # #0508增 (记得和下一段0508同步）
        # index = 45 #画第几个图
        # if not is_training and i == 0: #在测试集上画
        #     sections = torch.bincount(mt_batch)
        #     split_x = torch.split(data.x, tuple(sections.cpu().numpy()))
        #     mt_edge_sessions_bianhao = mt_batch[mt_edge_index[0]]
        #     positive_edge_sessions_bianhao = mt_batch[positive_edge_index[0]]
        #     pr_edge_sessions_bianhao = mt_batch[pr_edge_index[0]]
        #
        #     sender_x = data.x[pr_edge_index[0]].squeeze() #只画全局图中的跨会话边缘
        #     receiver_x = data.x[pr_edge_index[1]].squeeze()
        #     draw_edges = torch.stack((sender_x, receiver_x), dim=0)
        #     pr_edge_sections = torch.bincount(pr_edge_sessions_bianhao)
        #     split_draw_edge_index = torch.split(draw_edges, tuple(pr_edge_sections.cpu().numpy()), dim=1)
        #     g_draw_nodes = split_x[index]
        #     g_draw_edges = split_draw_edge_index[index]
        #     # draw_graph(g_draw_nodes, g_draw_edges)
        #
        #     l_sender_x = data.x[positive_edge_index[0]].squeeze() #只画全局图中的当前会话边缘
        #     l_receiver_x = data.x[positive_edge_index[1]].squeeze()
        #     l_draw_edges = torch.stack((l_sender_x, l_receiver_x), dim=0)
        #     l_edge_sections = torch.bincount(positive_edge_sessions_bianhao)
        #     l_split_draw_edge_index = torch.split(l_draw_edges, tuple(l_edge_sections.cpu().numpy()), dim=1)
        #     l_draw_nodes = split_x[index] #和g_draw_nodes相同
        #     l_draw_edges = l_split_draw_edge_index[index]
        #     #draw_graph(l_draw_nodes, l_draw_edges)
        #
        #     # mt_sender_x = data.x[mt_edge_index[0]].squeeze() #画原始全局图
        #     # mt_receiver_x = data.x[mt_edge_index[1]].squeeze()
        #     # mt_draw_edges = torch.stack((mt_sender_x, mt_receiver_x), dim=0)
        #     # mt_edge_sections = torch.bincount(mt_edge_sessions_bianhao)
        #     # mt_split_draw_edge_index = torch.split(mt_draw_edges, tuple(mt_edge_sections.cpu().numpy()), dim=1)
        #     # mt_draw_nodes = split_x[index]
        #     # mt_draw_edges = mt_split_draw_edge_index[index]
        #     # draw_graph(mt_draw_nodes, mt_draw_edges)
        #
        #     # 创建可视化目录（如果不存在）
        #     os.makedirs('keshihua', exist_ok=True)
        #     # 保存g_draw_nodes和g_draw_edges
        #     np.savetxt('keshihua/g_draw_nodes.txt', g_draw_nodes.cpu().numpy(), fmt='%d')
        #     np.savetxt('keshihua/g_draw_edges.txt', g_draw_edges.cpu().numpy().reshape(-1, g_draw_edges.shape[-1]), fmt='%d')
        #
        #     # 保存l_draw_nodes和l_draw_edges
        #     np.savetxt('keshihua/l_draw_nodes.txt', l_draw_nodes.cpu().numpy(), fmt='%d')
        #     np.savetxt('keshihua/l_draw_edges.txt', l_draw_edges.cpu().numpy().reshape(-1, l_draw_edges.shape[-1]), fmt='%d')
        # #0508增 (记得和下一段0508同步）

        global_forward_weight = torch.ones(pr_edge_index.shape[1]).unsqueeze(-1)  # mt_edge_count.unsqueeze(-1) # （边数量，1）全局图中边权重的设定方式
        # # 1102将hidden改为edge_emb
        # mt_x = data.x - 1
        # edge_embedding = self.embedding_edge(mt_x)
        # edge_embedding = edge_embedding.squeeze()
        # # 1102将hidden改为edge_emb
        s, l0_penalty_, probability4fea = self.linkp(hidden,  pr_edge_index, global_forward_weight, local_sess_avg,  #1102将hidden改为edge_emb
                                    mt_batch, is_training,use_forward=True)  # 边预测   此时边的权重值为自学习出来的？

        # # 0508增 (记得和上一段0508同步）
        # if not is_training and i == 0:  # 在测试集上画
        #     split_s = torch.split(s, tuple(pr_edge_sections.cpu().numpy()))
        #     weights_tensor = split_s[index]
        #     split_p = torch.split(probability4fea, tuple(pr_edge_sections.cpu().numpy()))
        #     p_tensor = split_p[index]
        #     # 合并两个张量为一个二维数组
        #     combined_array = np.column_stack((weights_tensor.cpu().numpy(),p_tensor.cpu().numpy()))
        #     # 获取当前时间，格式化为 年-月-日 时:分:秒
        #     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #     # 写入合并后的张量数据
        #     with open('keshihua/g_edges_weights_tensor.txt', 'a') as f:
        #         f.write(f"epoch: {epoch} {current_time}\n")
        #         np.savetxt(f, combined_array, fmt='%.4f', delimiter='\t')
        #         f.write("\n")
        #
        #         # # 获取当前时间，格式化为 年-月-日 时:分
        #     # current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        #     # with open('keshihua/g_edges_weights_tensor.txt', 'a') as f:
        #     #     f.write(f"epoch: {epoch} {current_time}\n")
        #     #     np.savetxt(f, weights_tensor.cpu().numpy(), fmt='%.4f')
        #     #     f.write("\n")
        # # 0508增 (记得和上一段0508同步）

        pred_edge_index, pred_edge_weight = construct_pred_edge(pr_edge_index, s, pr_edge_count)  # 提取保留的边/交互
        new_edge_index = torch.cat((positive_edge_index, pred_edge_index), dim=1)  #这样给边拼接在后面会有影响吗？？
        new_edge_count = torch.cat((positive_edge_count,pred_edge_weight))
        # 如果需要确保没有重复边，可以去重
        new_edge_index = torch.unique(new_edge_index, dim=1)
        use_undirected = False  # True #  反转双向的话，得考虑边权重的问题了
        if use_undirected:
            reverse_edge_index = torch.flip(new_edge_index, [0])  # 反转有向边
            undirected_edge_index = torch.cat([new_edge_index, reverse_edge_index], dim=1)
            mt_edge_index = undirected_edge_index
        else:
            mt_edge_index = new_edge_index
        # （过滤掉局部边之后再进行删边）

        ###########################################################################################

        alternative = 2  #选择全局图中信息聚合的方式
        if alternative == 1:
            hidden = self.sgcn(hidden, mt_edge_index)
        elif alternative == 2:
            neibor_hidden = self.lightgcn(hidden, mt_edge_index)   #考虑class LightGCNConv中的是否添加自环 →记得去修改代码
            # 是否要放在这里，类似于GCL，对neibor_hidden进一步个性化提取处理
            use_self_loops = True#False#
            if use_self_loops == True:
                hidden = neibor_hidden
            else:
                hidden = self.W_4(torch.cat([hidden,neibor_hidden],dim=-1))
            # #类似于GCL，对neighbor_vector进一步个性化提取处理
            # session_emb4fea = local_sess_avg[mt_batch]
            # gating_score_a = torch.sigmoid(self.WG1(hidden) +self.WG2(session_emb4fea))
            # hidden = hidden * gating_score_a
            # #类似于GCL，对neighbor_vector进一步个性化提取处理
        use_dropout = True#False#  #（可能在形成该通道会话表示之前先dropout一下，但注意和model里面的hidden的混淆使用！）
        if use_dropout == True:
            hidden = F.dropout(hidden, self.global_dropout, training=self.training)  # 0923增
        # hidden = F.normalize(hidden, p=2, dim=-1)   #0923增

        sess_hidden = self.rebuilt_sess(hidden, node_num, mt_sess_item_index, seq_lens)

        if self.item_fusing:
            return hidden, sess_hidden, l0_penalty_
        else:
            return hidden, self.get_h_group(sess_hidden, seq_lens), l0_penalty_
