from model_cleaned.multi_sess import GroupGraph
from model_cleaned.srgnn0509 import SRGNN

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

            # #NISER中的归一化
            emb =  F.normalize(emb, dim=0,p=2)
            final_s = self.scale * F.normalize(final_s, dim=-1,p=2)
            scores = torch.matmul(final_s, emb)
            z_i_hat = scores#torch.mm(final_s, emb)  #会发生 NaN or Inf found in input tensor.

            # # 常规内积
            # z_i_hat = torch.mm(final_s, emb)

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

        return z_i_hat#, #单个值后面加一个逗号（如 return z_i_hat,）会使得返回值变成一个元组（tuple）

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
        self.position_emb = nn.Embedding(max_len+1, self.hidden_size) #注意大小，否则可能发生CUDA设备端的断言错误
        self.W_pos_inter = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, intra_item_emb, inter_item_emb, seq_len, reverse_pos, session_features):

        final_emb = self.features_fusing(intra_item_emb, inter_item_emb, session_features, seq_len)
        # final_emb = intra_item_emb
        final_s = self.get_final_s_GCE_GNN(final_emb, seq_len, reverse_pos) #要打开GCE_GNN中的第一行了
        return final_s


    def get_final_s_GCE_GNN(self, hidden, seq_len,reverse_pos):  #原版
        #hidden = torch.split(hidden, tuple(seq_len.cpu().numpy()))  #注释此行
        #v_n = tuple(nodes[-1].view(1, -1) for nodes in hidden)
        # v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in hidden)
        # v_n_repeat = torch.cat(v_n_repeat, dim=0)  #要得到此形状大小
        # #0926增平均表示查询项
        v_mean_repeat = tuple(nodes.mean(dim=0, keepdim=True).repeat(nodes.shape[0], 1) for nodes in hidden)
        v_mean_repeat_concat = torch.cat(v_mean_repeat, dim=0)
        # #0926增平均表示查询项
        hidden = torch.cat(hidden, dim=0)

        distinguish = True#False#       #用于判断是否区分重复项目的位置编号
        if distinguish == True:         #重复项目的位置编号是相同的
            pos_emb = self.position_emb(reverse_pos)  #reverse_pos区分重复出现项目
            # # Posrec双向学习
            # forward_pos_emb = self.rotary_emb(forward_pos)
            # reverse_pos_emb = self.rotary_emb(reverse_pos)
            # pos_emb = torch.cat((forward_pos_emb, reverse_pos_emb), dim=-1)
        else:
            reverse_positions = []# 初始化一个空列表用于存储结果/GCE-GNN论文中好像直接按序列倒序，不区分重复出现项目
            for length in seq_len:
                reverse_positions.extend(reversed(range(length.item())))  # 根据当前的会话长度生成倒序位置
            reverse_positions = torch.tensor(reverse_positions).to(self.position_emb.weight.device)  # 不区分重复出现项目
            pos_emb = self.position_emb(reverse_positions)

        jiehe_alter = 1  #用于选择位置嵌入和项目嵌入两者之间的结合方式
        if jiehe_alter == 1:
            pos_hidden = torch.tanh(self.W_pos_inter(torch.cat([hidden, pos_emb], -1)))  # 拼接之后转换回d维，再加入一个tanh函数 GCE-GNN
        elif jiehe_alter == 2:
            pos_hidden = self.W_pos_inter(torch.cat([hidden, pos_emb], -1))  # 拼接之后转换回d维 HG-GNN
        elif jiehe_alter == 3:
            pos_hidden = hidden + pos_emb  # 直接相加 SGNN-HN

        #alpha = self.q_inter(torch.sigmoid(self.W_1_inter(v_n_repeat) + self.W_2_inter(hidden)))  # |V|_i * 1
        alpha = self.q_inter(torch.sigmoid(self.W_1_inter(v_mean_repeat_concat) + self.W_2_inter(pos_hidden)))  # (是否可以使用其他非线性激活如relu!)
        #alpha = self.q_inter(torch.sigmoid(self.W_1_inter(v_n_repeat) + self.W_11_inter(v_mean_repeat_concat) + self.W_2_inter(pos_hidden)))
        # 是否可再模仿SGRec的式子（7）来计算注意力
        #alpha = self.q_inter(torch.tanh(self.W_1_inter(v_n_repeat) + self.W_2_inter(hidden) + self.W_q_inter(pos_emb)))  #是否可能用于全局图通道的会话形成表示？
        alpha_softamax = False#True#
        if alpha_softamax:
            split_alpha = torch.split(alpha, seq_len.tolist())  # 根据 seq_len 切分 alpha
            normalized_alpha = [torch.nn.functional.softmax(session_alpha, dim=0) for session_alpha in split_alpha]# 对每个会话的 alpha 进行 softmax
            alpha = torch.cat(normalized_alpha)
        use_hidden_agg = True  #用于选择哪种嵌入(是否含有位置信息)用于聚合成最终的会话表示 #记得此处是选择hidden还是pos_hidden来聚合形成会话！！
        if use_hidden_agg:
            s_g_whole = alpha * hidden
        else:
            s_g_whole = alpha * pos_hidden
        s_g_split = torch.split(s_g_whole, tuple(seq_len.cpu().numpy()))  # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        # Eq(7)
        # h_s = torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1)  #注意：返回2d维的会话表示
        #h_s = self.W_3_inter(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        h_s = torch.cat(s_g, dim=0)
        return h_s

    # copy版
    def get_final_s_GCE_GNN_copy(self, hidden, seq_len,reverse_pos):          #copy版
        #hidden = torch.split(hidden, tuple(seq_len.cpu().numpy()))  #注释此行
        v_n = tuple(nodes[-1].view(1, -1) for nodes in hidden)
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in hidden)
        v_n_repeat = torch.cat(v_n_repeat, dim=0)  #要得到此形状大小
        # # # #0926增平均表示查询项
        # v_mean_repeat = tuple(nodes.mean(dim=0, keepdim=True).repeat(nodes.shape[0], 1) for nodes in hidden)
        # v_mean_repeat_concat = torch.cat(v_mean_repeat, dim=0)
        # # #0926增平均表示查询项
        hidden = torch.cat(hidden, dim=0)

        distinguish =  True# False#     #用于判断是否区分重复项目的位置编号
        if distinguish == True:         #重复项目的位置编号是相同的
            pos_emb = self.position_emb(reverse_pos)  #reverse_pos区分重复出现项目
            # # Posrec双向学习
            # forward_pos_emb = self.rotary_emb(forward_pos)
            # reverse_pos_emb = self.rotary_emb(reverse_pos)
            # pos_emb = torch.cat((forward_pos_emb, reverse_pos_emb), dim=-1)
        else:
            reverse_positions = []# 初始化一个空列表用于存储结果/GCE-GNN论文中好像直接按序列倒序，不区分重复出现项目
            for length in seq_len:
                reverse_positions.extend(reversed(range(length.item())))  # 根据当前的会话长度生成倒序位置
            reverse_positions = torch.tensor(reverse_positions).to(self.position_emb.weight.device)  # 不区分重复出现项目
            pos_emb = self.position_emb(reverse_positions)

        jiehe_alter = 1  #用于选择位置嵌入和项目嵌入两者之间的结合方式
        if jiehe_alter == 1:
            pos_hidden = torch.tanh(self.W_pos(torch.cat([hidden, pos_emb], -1)))  # 拼接之后转换回d维，再加入一个tanh函数 GCE-GNN
        elif jiehe_alter == 2:
            pos_hidden = self.W_pos(torch.cat([hidden, pos_emb], -1))  # 拼接之后转换回d维 HG-GNN
        elif jiehe_alter == 3:
            pos_hidden = hidden + pos_emb  # 直接相加 SGNN-HN

        #alpha = self.q_inter(torch.sigmoid(self.W_1_inter(v_n_repeat) + self.W_2_inter(hidden)))  # |V|_i * 1
        alpha = self.q(torch.sigmoid(self.W_1(v_n_repeat) + self.W_2(pos_hidden)))  # (是否可以使用其他非线性激活如relu!)
        #alpha = self.q_inter(torch.sigmoid(self.W_1_inter(v_n_repeat) + self.W_11_inter(v_mean_repeat_concat) + self.W_2_inter(pos_hidden)))
        # 是否可再模仿SGRec的式子（7）来计算注意力
        #alpha = self.q_inter(torch.tanh(self.W_1_inter(v_n_repeat) + self.W_2_inter(hidden) + self.W_q_inter(pos_emb)))  #是否可能用于全局图通道的会话形成表示？
        alpha_softamax = False#True#
        if alpha_softamax:
            split_alpha = torch.split(alpha, seq_len.tolist())  # 根据 seq_len 切分 alpha
            normalized_alpha = [torch.nn.functional.softmax(session_alpha, dim=0) for session_alpha in split_alpha]# 对每个会话的 alpha 进行 softmax
            alpha = torch.cat(normalized_alpha)
        use_hidden_agg = True  #用于选择哪种嵌入用于聚合成最终的会话表示 #记得此处是选择hidden还是pos_hidden来聚合形成会话！！
        if use_hidden_agg:
            s_g_whole = alpha * hidden
        else:
            s_g_whole = alpha * pos_hidden
        s_g_split = torch.split(s_g_whole, tuple(seq_len.cpu().numpy()))  # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)
        h_s = torch.cat(s_g, dim=0)
        return h_s



class GraphModel(nn.Module):
    def __init__(self, opt, n_node,max_len):
        super(GraphModel, self).__init__()
        #定义属性变量
        self.hidden_size, self.n_node, self.max_len = opt.hidden_size, n_node, max_len
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        #self.embedding_edge = nn.Embedding(self.n_node, self.hidden_size)  #将局部和全局的emb表分开对待？？
        self.length_emb = nn.Embedding(self.max_len+1, self.hidden_size)  #注意大小，否则可能发生CUDA设备端的断言错误
        self.dropout = opt.global_dropout
        self.negative_slope = opt.negative_slope
        self.heads = opt.heads
        self.item_fusing = opt.item_fusing
        self.num_filters = opt.num_filters
        self.use_alternative_coo = opt.use_alternative_coo
        self.use_alternative_final_s = opt.use_alternative_final_s
        #定义线性层 or 自学习参数
        self.W_coo_15 = nn.Linear(2 * self.hidden_size, self.hidden_size)  # （注意输出维度的设置 1 or self.hidden_size）
        # 16中 定义一致性向量和长度向量的全连接层
        self.fc_coo = nn.Linear(self.hidden_size, self.hidden_size)  # 一致性输入
        self.fc_length = nn.Linear(self.hidden_size, self.hidden_size)  # 长度输入
        self.fc_out = nn.Linear(self.hidden_size * 1, 1)  # 输出融合系数
        # 16中 定义一致性向量和长度向量的全连接层
        self.W_concat = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.W_sum_1 = nn.Linear(self.hidden_size, 1) # （注意输出维度的设置 1 or self.hidden_size）
        self.W_sum_2 = nn.Linear(self.hidden_size, 1) # （注意输出维度的设置 1 or self.hidden_size）
        self.W_Q = nn.Linear(self.hidden_size , self.hidden_size,bias=False)
        self.W_K = nn.Linear(self.hidden_size * 1, self.hidden_size,bias=False)
        self.W_con = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.W_self = nn.Linear(1 * self.hidden_size, self.hidden_size,bias=False)
        self.W_neibor = nn.Linear(1 * self.hidden_size, self.hidden_size,bias=False)
        self.W_gated_concat = nn.Linear(2 * self.hidden_size, 1,bias=False)  #2d,d 还是 2d,1
        self.W_gated_self = nn.Linear(1 * self.hidden_size, 1,bias=False)  #dd 还是 d1
        self.W_gated_neibor = nn.Linear(1 * self.hidden_size, 1,bias=False)  #dd 还是 d1
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

    def reset_parameters(self):  #均匀分布，好像实际代码中都是说采用这种
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    # rebuilt_sess重构会话序列表示大小为tuple：100 包括tensor（2,100).(3,100)......
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
            mt_x = data.mt_x - 1  #此处有进行减一操作！！
            embedding = self.embedding(mt_x)
            embedding = embedding.squeeze()
            mt_edge_index, mt_edge_count, mt_batch, node_num, mt_sess_item_index, \
                all_edge_label, mt_sess_masks, seq_lens,num_count = \
                data.edge_index, data.mt_edge_count, data.batch, data.mt_node_num, data.mt_sess_item_idx, \
                    data.all_edge_label, data.mt_sess_masks, data.sequence_len, data.num_count
            reverse_pos = data.reverse_pos #此反向位置重复项目统一相同使用最后一次出现的倒位置
            #0927同时生成（当前自身会话）的 （平均表示） 和 （最后一项）
            # 根据local_mt_sess_item_index获得local_xulie_hidden；根据seq_lens生成seq_batch
            v_i = torch.split(embedding, tuple(node_num.cpu().numpy()))  # 将整个 x 切分回图 G_i
            local_xulie_hidden = []
            rebuilt_last_item = []
            local_mt_sess_item_index = torch.split(mt_sess_item_index, tuple(seq_lens.cpu().numpy()))
            for node_indices, v in zip(local_mt_sess_item_index, v_i):
                local_xulie_hidden.append(v[node_indices])  # 提取每个图的节点嵌入
                sess = v[node_indices[-1]]# 提取每个图的最后一个节点的嵌入表示
                rebuilt_last_item.append(sess)
            # 拼接所有图的节点嵌入
            local_xulie_hidden = torch.cat(local_xulie_hidden, dim=0)  # 新增，以便直接使用 global_mean_pool 方法
            # 生成 seq_batch
            seq_lens_device = seq_lens.to(local_xulie_hidden.device)
            seq_batch = torch.repeat_interleave(torch.arange(len(seq_lens_device), device=local_xulie_hidden.device),seq_lens_device)
            local_sess_avg = global_mean_pool(local_xulie_hidden, seq_batch)# 计算序列的平均表示
            rebuilt_last_item = torch.stack(rebuilt_last_item, dim=0)# 将最后一个节点的嵌入表示拼接成一个张量
            #0927同时生成当前会话的平均表示和最后一项

            ###########################################################################################

            #（局部）
            #获取intra_item_emb；返回的是各会话序列空间的表示 大小为tuple：100 包括tensor（2,100).(3,100)......
            local_edge_index = mt_edge_index[:, all_edge_label == 1]    #局部边是否也需要删除？？
            #local_edge_count = mt_edge_count[all_edge_label == 1] #这是个局部边在全局图中的频率
            edge_count = data.edge_count  #这是个局部边在局部图中的频率
            intra_hidden, intra_item_emb = self.srgnn(data, embedding,local_edge_index,edge_count,mt_batch, local_sess_avg,rebuilt_last_item,is_training, i)  #是否可以直接这样？
            # （局部）

            # #获取传播后的本地会话的平均表示 or 最后一项 or 直接会话表示  以用于全局图删边的个性化
            # pred_intra_hidden = torch.cat(intra_item_emb, dim=0)
            # pred_local_sess_avg = global_mean_pool(pred_intra_hidden, seq_batch)  # 计算传播后的序列的平均表示
            # pred_local_sess = self.cnn_fusing.get_final_s_GCE_GNN(intra_item_emb, data.sequence_len, data.reverse_pos)
            # # #若用最后一项，则可以从self.srgnn返回得到
            # local_sess_avg_alter = 3
            # if local_sess_avg_alter == 1:  #使用正常平均项
            #     local_sess_avg = local_sess_avg
            # elif local_sess_avg_alter == 2: #使用经过局部GGNN或GAT聚合后的(平均项)
            #     local_sess_avg = pred_local_sess_avg
            # elif local_sess_avg_alter == 3: #使用经过局部GGNN或GAT聚合后的(局部会话表示)（模仿GNN-GNF）
            #     local_sess_avg = pred_local_sess
            # # 获取传播后的本地会话的平均表示 or 最后一项 or 直接会话表示  以用于全局图删边的个性化

            ###########################################################################################

            l0_penalty_ = 0  # 全局边预测过程得到的正则化数值
            #（全局）
            #获取inter_item_emb；返回的是各会话序列空间的表示 大小为tuple：100 包括tensor（2,100).(3,100)......
            inter_hidden, inter_item_emb, l0_penalty_ = self.group_graph.forward(i,epoch,embedding, data, local_sess_avg,rebuilt_last_item ,is_training=is_training)
            dropout_global = False# True#   #控制是否对全局项目表示进行dropout！！！（注意：和group_graph类里面的dropout的混淆使用）
            if dropout_global:
                inter_hidden = F.dropout(inter_hidden, 0.6, training=self.training)  # 控制是否对全局图中得到的项目表示进行dropout
            #（全局）


            # # 10.28(验证单局部图/单全局图时用下一行)
            # final_s = self.cnn_fusing.get_final_s_GCE_GNN(inter_item_emb, data.sequence_len, data.reverse_pos) # HG_GNN,用此行时，记得去注释get_final_s的第一行

            ####################################################################################
            ####################################################################################

            #[1]会话层面表示融合，得到两通道各自的会话表示后，两个会话表示再融合为一个最终的会话表示 （判断两通道是否使用不同的会话表示方法？？查询选择、位置正反及聚合、拼接last）

            final_s_intra = self.cnn_fusing.get_final_s_GCE_GNN(intra_item_emb,data.sequence_len, data.reverse_pos) # tensor(B,dim)
            final_s_inter = self.cnn_fusing.get_final_s_GCE_GNN(inter_item_emb, data.sequence_len, data.reverse_pos) # tensor(B,dim)

            #得到会话个性化的session_features
            use_fusion = True #  False #   如果为 False(5/6)，直接使用全0或全1 →即只使用某一通道的会话来直接预测
            use_alternative_coo = self.use_alternative_coo#5 # 控制 coo 生成方式的参数
            if use_fusion:
                v_last = torch.stack([session[-1] for session in intra_item_emb])  # (100, 100)
                v_mean = torch.stack([torch.mean(session, dim=0) for session in intra_item_emb])  # (100, 100)
                if use_alternative_coo == 15:
                    session_lengths = torch.tensor([len(session) for session in intra_item_emb], dtype=torch.long,
                                                   device=v_last.device)  # (B,)
                    coo_input_1 = v_mean - v_last
                    coo_input = torch.cat([coo_input_1, self.length_emb(session_lengths)],dim=-1)  # (B, 2*dim )
                    #coo_input = coo_input_1 + self.length_emb(session_lengths)  # (B, dim)
                    #coo = torch.sigmoid(self.W_coo_15(coo_input)).view(-1, 1) #输出（100，1） #不同的输入，使用不同维度的线性层,去改self.W_coo_15
                    coo = torch.sigmoid(self.W_coo_15(coo_input))#.view(-1, 1) #输出（100，100） #不同的输入，使用不同维度的线性层,去改self.W_coo_15
                elif use_alternative_coo == 16:
                    session_lengths = torch.tensor([len(session) for session in intra_item_emb], dtype=torch.long,
                                                   device=final_s_intra.device)  # (B,)
                    coo_input_1 = v_mean - v_last
                    coo_out = torch.tanh(self.fc_coo(coo_input_1))
                    length_out = torch.tanh(self.fc_length(self.length_emb(session_lengths)))
                    #相加 or 拼接
                    alpha_input = coo_out + length_out #相加  相加时是否可以不要tanh函数
                    #alpha_input = torch.cat([coo_out, length_out], dim=-1) #拼接→需改self.fc_out线性层的维度
                    coo = torch.sigmoid(self.fc_out(alpha_input)).view(-1,1)

                elif use_alternative_coo == 5:  #1111replace
                    coo = torch.ones(len(intra_item_emb), 1).to(final_s_intra.device)  # 全 1
                elif use_alternative_coo == 6:
                    coo = torch.zeros(len(intra_item_emb), 1).to(final_s_intra.device)  # 全 0 #1111replace

                session_features = coo #[B,1]  # 所有 session 的特征批量计算后直接返回

            else:  #1111replaced
                # 如果不需要融合计算，直接将 coo 设为全 1 或全 0
                if use_alternative_coo == 5:
                    session_features = torch.ones(len(intra_item_emb), 1).to(final_s_intra.device)  # 全 1
                elif use_alternative_coo == 6:
                    session_features = torch.zeros(len(intra_item_emb), 1).to(final_s_intra.device)  # 全 0  #1111replaced

            ###########################
            #利用以上得到的session_features进行拼接
            use_alternative_final_s = self.use_alternative_final_s#2  #final_s的拼接方式选择
            if use_alternative_final_s == 1:
                final_s = session_features * final_s_inter + (1 - session_features) * final_s_intra
            elif use_alternative_final_s == 2:  #倒置
                final_s = session_features * final_s_intra + (1 - session_features) * final_s_inter
            elif use_alternative_final_s == 3:  #HG-GNN式（17）
                alpha = self.W_concat(torch.cat([final_s_intra,final_s_inter],dim=-1))
                alpha_concat = torch.sigmoid(alpha)
                final_s = alpha_concat * final_s_intra + (1 - alpha_concat) * final_s_inter
            elif use_alternative_final_s == 4:  #I3GN式（12）
                alpha = (self.W_sum_1(final_s_intra) + self.W_sum_2(final_s_inter))
                alpha_sum = torch.sigmoid(alpha)
                final_s = alpha_sum * final_s_intra + (1 - alpha_sum) * final_s_inter
            elif use_alternative_final_s == 5:  # GNRRW式（7） 自学习出来
                final_s = F.sigmoid(self.a1) * final_s_intra + F.sigmoid(self.a2) * final_s_inter
            elif use_alternative_final_s == 6:  #直接拼接后转换为d维
                final_s = self.W_concat(torch.cat([final_s_intra,final_s_inter],dim=-1))
            elif use_alternative_final_s == 7: #直接相加，模仿CAR中的No Gating Network无消融
                final_s = final_s_intra + final_s_inter
            elif use_alternative_final_s == 8:
                final_s = torch.max(final_s_intra,final_s_inter)
            #[1]会话层面表示融合，得到两通道各自的会话表示后，两个会话表示再融合为一个最终的会话表示 （判断两通道是否使用不同的会话表示方法？？查询选择、位置正反及聚合、拼接last）

            ####################################################################################

            # #（2）在项目层面进行动态权重融合
            # # （注意）：要打开会话层面融合的代码，以利用以上得到的session_features；并需设置use_alternative_final_s = 0；
            # # 可暂时不用算出final_s_intra和final_s_inter，以提高运行效率
            # final_s = self.cnn_fusing.forward(intra_item_emb, inter_item_emb, data.sequence_len, data.reverse_pos, session_features)

            #计算预测分数
            scores = self.e2s(h_s=None, h_group=None, final_s=final_s, item_embedding_table=self.embedding)

            # #0921辅助任务 ： LDGC-SR
            # intra_scores = self.e2s(h_s=None, h_group=None, final_s=final_s_intra, item_embedding_table=self.embedding)
            # inter_scores = self.e2s(h_s=None, h_group=None, final_s=final_s_inter, item_embedding_table=self.embedding)
            # use_k_fusion = False #True #
            # if use_k_fusion:
            #     k=session_features   #[B,1]
            #     #k = 1-k
            #     scores = k *intra_scores + (1-k) * inter_scores
            # else:
            #     scores = (intra_scores + inter_scores) / 2
            # # 0921辅助任务
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
        #return scores, intra_scores, inter_scores, l0_penalty_

