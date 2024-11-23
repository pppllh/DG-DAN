# 专门用于跑NEGTN

import os
import argparse
import logging
import time
import torch
from tqdm import tqdm
from data.multi_sess_graph import *
#from data.multi_sess_graph0924 import *
from torch_geometric.loader import DataLoader #用于加载图数据集的 DataLoader
from model_10.model import GraphModel
from train import forward
from tensorboardX import SummaryWriter



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: Retailrocket/tmall/nowplaying/yoochoose1_64/sample/diginetica/Tmall')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=15, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default= 0.0005, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-6, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--top_k', type=int, default=20, help='top K indicator for evaluation')
parser.add_argument('--negative_slope', type=float, default=0.2, help='negative_slope')
parser.add_argument('--gat_dropout', type=float, default=0.6, help='dropout rate in gat')
parser.add_argument('--heads', type=int, default=8, help='gat heads number')
parser.add_argument('--num_filters', type=int, default=2, help='gat heads number')
parser.add_argument('--using_represent', type=str, default='comb', help='comb, h_s, h_group')
parser.add_argument('--predict', type=bool, default=False, help='gat heads number')
parser.add_argument('--item_fusing', type=bool, default=True, help='gat heads number')
parser.add_argument('--random_seed', type=int, default=24, help='input batch size')
parser.add_argument('--id', type=int, default=70, help='id')
parser.add_argument('--l0_para', nargs='?', default='[0.66, -0.1, 1.1]',
                        help="l0 parameters, which are beta (temprature), zeta (interval_min) and gama (interval_max).")
opt = parser.parse_args()
#logging.warning(opt)


def main():

    # torch.manual_seed(opt.random_seed)
    # torch.cuda.manual_seed(opt.random_seed)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    cur_dir = os.getcwd()
    #适用multi_sess_graph ： 邻居会话构成全局图 ATU_GTN_
    train_dataset = MultiSessionsGraph(cur_dir + '/datasets/' + opt.dataset, phrase='train', knn_phrase='ATU_GTN_neigh_data_'+str(opt.id))
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataset = MultiSessionsGraph(cur_dir + '/datasets/' + opt.dataset, phrase='test', knn_phrase='ATU_GTN_neigh_data_'+str(opt.id))
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    # #适用multi_sess_graph ： 邻居会话构成全局图 DGTN
    # train_dataset = MultiSessionsGraph(cur_dir + '/datasets/' + opt.dataset, phrase='train', knn_phrase='new_03_neigh_data_'+str(opt.id))
    # train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    # test_dataset = MultiSessionsGraph(cur_dir + '/datasets/' + opt.dataset, phrase='test', knn_phrase='new_03_neigh_data_'+str(opt.id))
    # test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    # #适用multi_sess_graph0924 ： GCE-GNN构成全局图
    # train_dataset = MultiSessionsGraph(root=cur_dir + '/datasets/' + opt.dataset, phrase='train')
    # train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    # test_dataset = MultiSessionsGraph(root=cur_dir + '/datasets/' + opt.dataset, phrase='test')
    # test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    log_dir = cur_dir + '/log/' + str(opt.dataset) + '/' + time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.warning('logging to {}'.format(log_dir))
    writer = SummaryWriter(log_dir)

    if opt.dataset == 'new_cikm16':
        n_node = 43097
    elif opt.dataset == 'diginetica':
        n_node = 39055#43097
    elif opt.dataset == 'yoochoose1_64':
        n_node = 17597#17400
    elif opt.dataset == 'tmall':
        n_node = 40727  #40727+1
    elif opt.dataset == 'Tmall':
        n_node = 40727  #40727+1
    elif opt.dataset == 'nowplaying':
        n_node = 42626  #60416+1
    elif opt.dataset == 'Retailrocket':
        n_node = 36968  # 36968+1 /对的，项目数确实为36968
    else:
        n_node = 309

    model = GraphModel(opt, n_node=n_node).to(device)
    # #源代码
    # multigraph_parameters = list(map(id, model.group_graph.parameters()))
    # srgnn_parameters = (p for p in model.parameters() if id(p) not in multigraph_parameters)
    # parameters = [{"params": model.group_graph.parameters(), "lr": 0.001}, {"params": srgnn_parameters}]
    # # best 0.1
    # lambda1 = lambda epoch: 0.1 ** (epoch // 3)
    # lambda2 = lambda epoch: 0.1 ** (epoch // 3)
    # optimizer = torch.optim.Adam(parameters, lr=opt.lr, weight_decay=opt.l2)
    # #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
    # # 源代码

    #常规
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
    #常规

    #logging.warning(model)
    if not opt.predict:
        best_result20 = [0, 0]
        best_epoch20 = [0, 0]

        best_result10 = [0, 0]
        best_epoch10 = [0, 0]

        best_result5 = [0, 0]
        best_epoch5 = [0, 0]
        for epoch in range(opt.epoch):
            start_time = time.time()
            #scheduler.step(epoch)
            print("Epoch ", epoch)
            forward(model, train_loader, device, writer, epoch, top_k=opt.top_k, optimizer=optimizer, train_flag=True)
            end_time = time.time()
            #print(end_time - start_time)
            scheduler.step()
            with torch.no_grad():
                mrr20, hit20, mrr10, hit10, mrr5, hit5 = forward(model, test_loader, device, writer, epoch, top_k=opt.top_k, train_flag=False)

            if hit20 >= best_result20[0]:
                best_result20[0] = hit20
                best_epoch20[0] = epoch
                # torch.save(model.state_dict(), log_dir+'/best_recall_params.pkl')
            if mrr20 >= best_result20[1]:
                best_result20[1] = mrr20
                best_epoch20[1] = epoch

            if hit10 >= best_result10[0]:
                best_result10[0] = hit10
                best_epoch10[0] = epoch
                # torch.save(model.state_dict(), log_dir+'/best_recall_params.pkl')
            if mrr10 >= best_result10[1]:
                best_result10[1] = mrr10
                best_epoch10[1] = epoch
                # torch.save(model.state_dict(), log_dir+'/best_mrr_params.pkl')

            if hit5 >= best_result5[0]:
                best_result5[0] = hit5
                best_epoch5[0] = epoch
                # torch.save(model.state_dict(), log_dir+'/best_recall_params.pkl')
            if mrr5 >= best_result5[1]:
                best_result5[1] = mrr5
                best_epoch5[1] = epoch

            print('Best Result:')
            print('\tMrr@%d:\t%.4f\tEpoch:\t%d' % (20, best_result20[1], best_epoch20[1]))
            print('\tRecall@%d:\t%.4f\tEpoch:\t%d\n' % (20, best_result20[0], best_epoch20[0]))
            # print('\tMrr@%d:\t%.4f\tEpoch:\t%d' % (opt.top_k, best_result10[1], best_epoch10[1]))
            # print('\tRecall@%d:\t%.4f\tEpoch:\t%d\n' % (opt.top_k, best_result10[0], best_epoch10[0]))
            # print('\tMrr@%d:\t%.4f\tEpoch:\t%d' % (5, best_result5[1], best_epoch5[1]))
            # print('\tRecall@%d:\t%.4f\tEpoch:\t%d' % (5, best_result5[0], best_epoch5[0]))
            print("-"*20)
        # print_txt(log_dir, opt, best_result, best_epoch, opt.top_k, note, save_config=True)
    else:
        log_dir = 'log/cikm16/2019-08-19 14:27:33'
        model.load_state_dict(torch.load(log_dir+'/best_mrr_params.pkl'))
        mrr, hit = forward(model, test_loader, device, writer, 0, top_k=opt.top_k, train_flag=False)
        best_result = [hit, mrr]
        best_epoch = [0, 0]
        # print_txt(log_dir, opt, best_result, best_epoch, opt.top_k, save_config=False)

# if __name__ == '__main__':
main()
