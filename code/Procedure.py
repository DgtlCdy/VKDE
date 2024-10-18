'''
Created on Aug 22, 2022

Design training and test process
'''
import world
import numpy as np
import torch
from torch import nn
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
# from sklearn.metrics import roc_auc_score
import math
import joblib


CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    utils.print_log(f'start of BPR_train_original') # testonly
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    utils.print_log(f'end of BPR_train_original') # testonly
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    utils.print_log(f'start of test_one_batch') # testonly
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    utils.print_log(f'end of test_one_batch') # testonly
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    utils.print_log(f'start of Test, epoch-{epoch}') # testonly
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    if world.simple_model == None:
        Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        print('***** test users: %s'%len(users))
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = math.ceil(len(users) / u_batch_size)
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            if(world.simple_model != 'none'):
                rating = Recmodel.getUsersRating(batch_users, world.dataset)
                rating = torch.from_numpy(rating)
                rating = rating.to('cuda')
            else:
                rating = Recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        utils.print_log(f'end of Test, epoch-{epoch}') # testonly
        utils.write_test_result(f'{epoch}-th epoch, {results}', 'VKDE_yelp2018_fix_interest_sample.txt')
        return results


def Test_sim(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict

    # eval mode with no dropout
    if world.simple_model == None:
        Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        print('***** test users: %s'%len(users))
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []

        total_batch = math.ceil(len(users) / u_batch_size)
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
        
            groundTrue_list.append(groundTrue)

            user_idxes, item_idxes = Recmodel.dataset.UserItemNet[batch_users_gpu.cpu()].nonzero()
            #print("user_idxes, item_idxes:", user_idxes, item_idxes)
            num_user_inters = np.array(Recmodel.dataset.UserItemNet[user_idxes].sum(axis=1)).flatten()
            inter_items = Recmodel.ii_sim_idx_mat[item_idxes].flatten()
            #print("inter_items.shape:",inter_items.shape)

            start= 0
            end = 0
            lastNum = 0
            temp = []
            rating= []
            old_user = user_idxes[0]
            for index, user in enumerate(user_idxes):
                if user == old_user:
                    temp.extend(inter_items[index*Recmodel.topk: index*Recmodel.topk+Recmodel.topk])
                else:
                    rating.append(temp)
                    temp = []
                    old_user = user
            rating.append(temp)              
            rating_list.append(rating)

        # print("groundTrue_list len:", len(groundTrue_list))
        # print("rating_list len:", len(rating_list))
        # input("Next?")

        all_correct_rate = []
        all_recall_rate = []
        for index, rating_batch in enumerate(rating_list):
            # print("groundTrue_list[index] len:", len(groundTrue_list[index]))
            # print("rating_list[index] len:", len(rating_batch))
            # input("Next?")
            for jndex, rating in enumerate(rating_batch):
                candidates = list(set(rating))
                #print("len:",len(candidates))
                correct_count = 0
                for candidate in candidates:
                    if int(candidate) in groundTrue_list[index][jndex]:
                        #print("correct:",int(candidate))
                        correct_count += 1
                
                if len(candidates) != 0:
                    #print("correct_count:", correct_count, "real_count:", len(groundTrue_list[index][jndex]))
                    all_correct_rate.append(correct_count*1.0/len(candidates))
                    all_recall_rate.append(correct_count*1.0/len(groundTrue_list[index][jndex]))
                    #print(correct_count*1.0/len(candidates))
        print("mean:", np.array(all_correct_rate).mean(), "recall mean:", np.array(all_recall_rate).mean())
        
        return 0

def Test_Diff_Users(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    if isinstance(Recmodel, nn.Module):
        Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    
    user2group = []
    user_n_items = [len(dataset.UserItemNet[i].nonzero()[1]) for i in range(dataset.n_users)]
    min_n_items = min(user_n_items)
    max_n_items = max(user_n_items)
    if max_n_items < 16 * 5:
        interval = 8
    else:
        interval = 16
    group_line = [min_n_items + i * interval for i in range(5)]
    for num in user_n_items:
        if num >= group_line[0] and num < group_line[1]:
            user2group.append(0)
        elif num >= group_line[1] and num < group_line[2]:
            user2group.append(1)
        elif num >= group_line[2] and num < group_line[3]:
            user2group.append(2)
        elif num >= group_line[3] and num < group_line[4]:
            user2group.append(3)
        else:
            user2group.append(4)
    results_groups = {}
    for i in range(5):
        results_groups[i] = {'precision': np.zeros(len(world.topks)),
                             'recall': np.zeros(len(world.topks)),
                             'ndcg': np.zeros(len(world.topks))}
    test_users = list(testDict.keys())
    groups = [[] for _ in range(5)]
    for uid in test_users:
        groups[user2group[uid]].append(uid)

    with torch.no_grad():
        for i in range(len(groups)):
            users = groups[i]
            print(f'Group {i} user num: {len(users)}')

            users_list = []
            rating_list = []
            groundTrue_list = []
            # auc_record = []
            # ratings = []
            total_batch = math.ceil(len(users) / u_batch_size)
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosItems(batch_users)
                groundTrue = [testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)
                if(world.simple_model != 'none'):
                    # rating = lm.getUsersRating(batch_users, world.dataset)
                    rating = Recmodel.getUsersRating(batch_users, world.dataset)
                    rating = torch.from_numpy(rating)
                    rating = rating.to(world.device)
                    ## Copy data to GPU and back introduces latency, just to fit the functions in VKDE
                else:
                    rating = Recmodel.getUsersRating(batch_users_gpu)
                # rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1<<10)
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                # aucs = [ 
                #         utils.AUC(rating[i],
                #                   dataset, 
                #                   test_data) for i, test_data in enumerate(groundTrue)
                #     ]
                # auc_record.extend(aucs)
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            assert total_batch == len(users_list)
            X = zip(rating_list, groundTrue_list)
            if multicore == 1:
                pre_results = pool.map(test_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(test_one_batch(x))
            scale = float(u_batch_size/len(users))
            for result in pre_results:
                results_groups[i]['recall'] += result['recall']
                results_groups[i]['precision'] += result['precision']
                results_groups[i]['ndcg'] += result['ndcg']
            results_groups[i]['recall'] /= float(len(users))
            results_groups[i]['precision'] /= float(len(users))
            results_groups[i]['ndcg'] /= float(len(users))
            # results['auc'] = np.mean(auc_record)
            if world.tensorboard:
                w.add_scalars(f'Test/Recall@{world.topks}',
                              {str(world.topks[j]): results_groups[i]['recall'][j] for j in range(len(world.topks))}, epoch)
                w.add_scalars(f'Test/Precision@{world.topks}',
                              {str(world.topks[j]): results_groups[i]['precision'][j] for j in range(len(world.topks))}, epoch)
                w.add_scalars(f'Test/NDCG@{world.topks}',
                              {str(world.topks[j]): results_groups[i]['ndcg'][j] for j in range(len(world.topks))}, epoch)
            if multicore == 1:
                pool.close()

        return results_groups, group_line

def Test_Embeddings(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict

    # eval mode with no dropout
    if world.simple_model == None:
        Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        print('***** test users: %s'%len(users))
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        embedding_list = []
        # auc_record = []
        # ratings = []
        total_batch = math.ceil(len(users) / u_batch_size)
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            if(world.simple_model != 'none'):
                # rating = lm.getUsersRating(batch_users, world.dataset)
                rating = Recmodel.getUsersRating(batch_users, world.dataset)
                rating = torch.from_numpy(rating)
                rating = rating.to('cuda')
                ## Copy data to GPU and back introduces latency, just to fit the functions in VKDE
            else:
                rating = Recmodel.getUsersRating(batch_users_gpu)
                embedding = Recmodel.getEmbeddings(batch_users_gpu)
            # rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
            embedding_list.append(embedding.cpu().numpy())
        assert total_batch == len(users_list)
        # joblib.dump(rating_list, './pakdd/plot_data/{}_rec_items.pkl'.format(world.dataset), compress=3)
        joblib.dump(embedding_list, './pakdd/plot_data/{}_{}_{}_embeddings.pkl'.format(world.model_name, world.dataset, world.config['enc_dims']), compress=3)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results