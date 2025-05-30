import numpy as np
import logging
import time

def forward(model, loader, device, writer, epoch, l0=0.3,top_k=20, optimizer=None, train_flag=True):
    start = time.time()
    if train_flag:
        model.train()
    else:
        model.eval()
        hit10, mrr10 = [], []
        hit5, mrr5 = [], []
        hit20, mrr20 = [], []

    mean_loss = 0.0
    updates_per_epoch = len(loader)
    test_dict = {}
    for i, batch in enumerate(loader):
        if train_flag:
            optimizer.zero_grad()
        scores, l0_penalty_ = model(batch.to(device), i, epoch, is_training=train_flag)
        targets = batch.y - 1
        loss = model.loss_function(scores, targets) + l0*l0_penalty 
        if train_flag:
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/train_batch_loss', loss.item(), epoch * updates_per_epoch + i)
        else:
            sub_scores = scores.topk(20)[1]    # batch * top_k
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit20.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr20.append(0)
                else:
                    mrr20.append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(top_k)[1]    # batch * top_k
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit10.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr10.append(0)
                else:
                    mrr10.append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(5)[1]    # batch * top_k
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit5.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr5.append(0)
                else:
                    mrr5.append(1 / (np.where(score == target)[0][0] + 1))


        mean_loss += loss / batch.num_graphs
        end = time.time()
        print("\rProcess: [%d/%d]   %.2f   usetime: %fs" % (i, updates_per_epoch, i/updates_per_epoch * 100, end - start),
              end='', flush=True)
    print('\n')

    if train_flag:
        writer.add_scalar('loss/train_loss', mean_loss.item(), epoch)
        print("Train_loss: ", mean_loss.item())
    else:
        writer.add_scalar('loss/test_loss', mean_loss.item(), epoch)
        hit20 = np.mean(hit20) * 100
        mrr20 = np.mean(mrr20) * 100

        hit10 = np.mean(hit10) * 100
        mrr10 = np.mean(mrr10) * 100

        hit5 = np.mean(hit5) * 100
        mrr5 = np.mean(mrr5) * 100
        print("Result:")
        print("\tMrr@", 20, ": ", mrr20)
        print("\tRecall@", 20, ": ", hit20)

        return mrr20, hit20, mrr10, hit10, mrr5, hit5



