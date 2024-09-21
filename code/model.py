"""
Created on Aug. 22, 2022

Define models here
"""
from world import cprint
import world
import torch
from torch.autograd import Variable, backward
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.utils.data import DataLoader
import torch.utils.data as data
import time
from dataloader import BasicDataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import numpy as np
# from sparsesvd import sparsesvd
import math
import os
import joblib
from tqdm import tqdm
import pickle

import random
import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s- %(filename)s[line:%(lineno)d]-%(levelname)s: %(message)s')

logging.warning("Launch Logging")
class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError


class MultVAE(BasicModel):
    """
    Implementation of Mult-VAE
    """
    def __init__(self, config, dataset):
        super(MultVAE, self).__init__()
        self.config = config
        self.dataset = dataset

        R = self.dataset.UserItemNet.A
        self.R = torch.tensor(R).float()

        self.lam = self.config['vae_reg_param']
        self.anneal_ph = self.config['kl_anneal']
        self.act = world.config['act_vae']
        self.dropout = self.config['dropout_multvae']

        self.total_anneal_steps = 200000
        self.update_count = 0.0

        enc_dims = self.config['enc_dims']
        if isinstance(enc_dims, str):
            enc_dims = eval(enc_dims)
        dec_dims = enc_dims[::-1]

        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        enc_dims = [self.num_items] + enc_dims
        dec_dims = dec_dims + [self.num_items]

        self.encoder = nn.Sequential()
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            if i == len(enc_dims) - 2:
                out_dim = out_dim * 2
            self.encoder.add_module(name='Encoder_Linear_%s'%i, module=nn.Linear(in_dim, out_dim))
            if i != len(enc_dims) - 2:
                self.encoder.add_module(name='Encoder_Activation_%s'%i, module=self.act)
            
        self.decoder = nn.Sequential()
        for i, (in_dim, out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            self.decoder.add_module(name='Decoder_Linear_%s'%i, module=nn.Linear(in_dim, out_dim))
            if i != len(dec_dims) - 2:
                self.decoder.add_module(name='Decoder_Activation_%s'%i, module=self.act)

        self.init_param()
        self.init_optim()
    
    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.trunc_normal_(m.bias, std=0.001)
    
    def init_optim(self):
        self.optim = optim.Adam([param for param in self.parameters() if param.requires_grad], self.config['vae_lr'])
    
    def forward(self, rating_matrix_batch):
        batch_input = F.normalize(rating_matrix_batch, p=2, dim=1)
        batch_input = F.dropout(batch_input, p=self.dropout, training=self.training)

        x = self.encoder(batch_input)
        mean, logvar = x[:, :(len(x[0])//2)], x[:, (len(x[0])//2):]
        stddev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(stddev)
        if self.training:
            # epsilon = torch.randn_like(stddev)
            z = mean + epsilon * stddev
        else:
            z = mean
        out = self.decoder(z)
        # KL divergence
        var_square = torch.exp(logvar)
        kl = 0.5 * torch.mean(torch.sum(mean ** 2 + var_square - 1. - logvar, dim=-1))
        return out, kl

    def getUsersRating(self, users):
        self.eval()
        users = users.cpu().numpy()
        rating_matrix_batch = self.R[users].to(world.device)
        predict_out, _ = self.forward(rating_matrix_batch)
        return predict_out
    
    def getEmbeddings(self, users):
        self.eval()
        users = users.cpu()
        rating_matrix_batch = self.R[users].to(world.device)
        batch_input = rating_matrix_batch
        x = self.encoder(batch_input)
        mean, logvar = x[:, :(len(x[0])//2)], x[:, (len(x[0])//2):]
        return mean

    @staticmethod
    def calculate_mult_log_likelihood(prediction, label):
        log_softmax_output = torch.nn.LogSoftmax(dim=-1)(prediction)
        log_likelihood = -torch.mean(torch.sum(log_softmax_output * label, 1))
        return log_likelihood
    
    def reg_loss(self):
        """
        Return the L2 regularization of weights. Code is implemented according to tensorflow.
        """
        reg_list = [0.5 * torch.sum(m.weight ** 2) for m in self.modules() if isinstance(m, nn.Linear)]
        reg = 0.
        for val in reg_list:
            reg += val
        return reg
    
    def train_one_epoch(self):
        self.train()
        users = np.arange(self.num_users)
        np.random.shuffle(users)
        batch_size = self.config['vae_batch_size']
        n_batch = math.ceil(self.num_users / batch_size)
        loss_dict = {}
        neg_ll_list = []
        kl_list = []
        reg_list = []
        for idx in range(n_batch):
            start_idx = idx * batch_size
            end_idx = min(start_idx + batch_size, self.num_users)
            batch_users = users[start_idx:end_idx]
            rating_matrix_batch = self.R[batch_users].to(world.device)
            predict_out, kl = self.forward(rating_matrix_batch)
            neg_ll = self.calculate_mult_log_likelihood(predict_out, rating_matrix_batch)

            if self.total_anneal_steps > 0:
                self.anneal_ph = min(self.anneal_ph, 1. * self.update_count / self.total_anneal_steps)
            else:
                pass

            loss = neg_ll + self.anneal_ph * kl + self.lam * self.reg_loss()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            neg_ll_list.append(neg_ll.item())
            kl_list.append(kl.item())
            reg_list.append(self.reg_loss().item())
            self.update_count += 1
        loss_dict['neg_ll'] = neg_ll_list
        loss_dict['kl'] = kl_list
        loss_dict['reg'] =  reg_list
        return loss_dict


class VKDE(nn.Module):
    """
    Implementation of VKDE
    """
    def __init__(self, config, dataset):
        super(VKDE, self).__init__()
        self.config = config
        self.dataset = dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items

        R = self.dataset.getBipartiteGraph().toarray()
        logging.warning(type(R))
        self.R = torch.tensor(R).float()

        self.R2 = np.random.choice(range(self.num_items), self.num_users)

        self.lam = self.config['vae_reg_param']
        self.anneal_ph = self.config['kl_anneal']
        self.act = world.config['act_vae']
        self.tau = config['tau_model2']
        self.dropout = config['dropout_model2']
        self.normalize = config['normalize_model2']
        self.topk = config['topK_model3']

        enc_dims = self.config['enc_dims']
        if isinstance(enc_dims, str):
            enc_dims = eval(enc_dims)
        dec_dims = enc_dims[::-1]

        enc_dims = [self.num_items] + enc_dims
        dec_dims = dec_dims + [self.num_items]

        self.encoder = nn.Sequential()
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            if i == len(enc_dims) - 2:
                out_dim = out_dim * 2
            self.encoder.add_module(name='Encoder_Linear_%s'%i, module=nn.Linear(in_dim, out_dim))
            if i != len(enc_dims) - 2:
                self.encoder.add_module(name='Encoder_Activation_%s'%i, module=self.act)
                pass

        self.mapper = nn.Linear(enc_dims[-1], 1)  

        self.items = nn.parameter.Parameter(torch.randn(self.num_items, enc_dims[-1]))
        self.items_weight = nn.parameter.Parameter(torch.randn(self.num_items, enc_dims[-1]))
 
        self.init_param()
        self.init_optim()

        self.get_topk_ii()

        if world.dataset in ['amazon-book', 'ml-20m']:
            self.gram_matrix = torch.tensor(self.gram_matrix).float()
        else:
            self.gram_matrix = torch.tensor(self.gram_matrix).float().to(world.device)
        self.epoch = 0
    
    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if isinstance(m.bias, torch.Tensor):
                    nn.init.trunc_normal_(m.bias, std=0.001)
        nn.init.xavier_normal_(self.items)
        
    
    def init_optim(self):
        self.optim = optim.Adam([param for param in self.parameters() if param.requires_grad], self.config['vae_lr'])
    
    
    #ideology：calculate local interaction，forward learning，combine local distribution
    def forward_kernel_1226(self, rating_matrix_batch, rating_matrix_batch2=None):
        batch_input0 = F.normalize(rating_matrix_batch, p=2, dim=1).cpu()
        batch_input0 = F.dropout(batch_input0, p=self.dropout, training=self.training)

        if world.dataset in ['amazon-book', 'ml-20m']:
            zeros = torch.zeros(rating_matrix_batch.shape[1]) 
            ones = torch.ones(rating_matrix_batch.shape[1])
        else:
            batch_input0 = batch_input0.to(world.device)
            zeros = torch.zeros(rating_matrix_batch.shape[1]).to(world.device)
            ones = torch.ones(rating_matrix_batch.shape[1]).to(world.device)

        batch_input01 = torch.where(batch_input0>0, ones, zeros)  #dropout rating_matrix_batch
        batch_input_arr = []   #record multiple local interactions
        batch_input_num = []   #record number

        for user in range(batch_input01.shape[0]):
            user_input = batch_input01[user]
            if rating_matrix_batch2!=None:
                items = torch.nonzero(rating_matrix_batch2[user]).cpu().numpy()  
            else:
                items = torch.nonzero(user_input).cpu().numpy()
            items = items.reshape(-1)

            # deal with no interaction
            if len(items) == 0:
                items = np.random.choice(range(self.num_items), 30)
           
            logging.warning('user:{0}, items:{1},{2}'.format(user, items, batch_input01.shape[0]))
            item_similars_sampled = self.gram_matrix[items] 
            input_item_sampled = item_similars_sampled * user_input
            input_item_sampled = F.normalize(input_item_sampled, p=1, dim=1)
            batch_input_arr.append(input_item_sampled)
            
            batch_input_num.append(len(items))
        
        new_input = torch.cat(batch_input_arr, dim =0).to(world.device)
        batch_input0 = F.normalize(new_input, p=2, dim=1)

        #encoder and decoder
        x = self.encoder(batch_input0)
        mean, logvar = x[:, :(len(x[0])//2)], x[:, (len(x[0])//2):]
        stddev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(stddev)
        if self.training:
            z = mean + epsilon * stddev
        else:
            z = mean

        logging.warning(z.shape)
        dot_product = z @ self.items.T

        try:
            if self.normalize:
                out = F.normalize(z)@ (F.normalize(self.items).T)  / self.tau #- self.popularity
            
            else:
                out = z @ self.items.T 

        except Exception as e :
            print(e)
            print("new_input.shape:", new_input.shape)

        #combine z
        zeros = torch.zeros(rating_matrix_batch.shape[1]).to(world.device)
        _index = 0
        new_output = []
        out = torch.exp(out)
        for inner_num in batch_input_num: 
            if inner_num!= 0:
                start_index = _index
                end_index = _index + inner_num   #
                inner_out = torch.mean(out[start_index:end_index, :], dim = 0)  #average for out
                inner_out = torch.log(inner_out+1.0)   
                new_output.append(inner_out.unsqueeze(0))
                _index = end_index
            else:    #deal with exception
                new_output.append(zeros.unsqueeze(0))

        new_output = torch.cat(new_output, dim=0)

        var_square = torch.exp(logvar)
        kl = 0.5 * torch.mean(torch.sum(mean ** 2 + var_square - 1. - logvar, dim=-1))

        return z, new_output, kl, batch_input0 

    #Sample one interest
    def forward_kernel(self, rating_matrix_batch, rating_matrix_batch2=None):
        batch_input0 = F.normalize(rating_matrix_batch, p=2, dim=1)
        batch_input0 = F.dropout(batch_input0, p=self.dropout, training=self.training)

        if world.dataset in ['amazon-book', 'ml-20m']:
            zeros = torch.zeros(rating_matrix_batch.shape[1]).to(world.device)
            ones = torch.ones(rating_matrix_batch.shape[1]).to(world.device)
        else:
            zeros = torch.zeros(rating_matrix_batch.shape[1]).to(world.device)
            ones = torch.ones(rating_matrix_batch.shape[1]).to(world.device)

        batch_input01 = torch.where(batch_input0>0, ones, zeros) 

        logging.debug(('{0}, {1}').format(rating_matrix_batch2.shape, rating_matrix_batch2))
        item_similars_sampled = torch.Tensor(self.gram_matrix[rating_matrix_batch2]).to(world.device)
        logging.debug(('{0}, {1}').format(item_similars_sampled.shape, batch_input01.shape))
        input_item_sampled = item_similars_sampled * batch_input01
        input_item_sampled = F.normalize(input_item_sampled, p=1, dim=1)

        batch_input0 = F.normalize(input_item_sampled, p=2, dim=1)

        #encoder 和 decoder
        x = self.encoder(batch_input0)
        mean, logvar = x[:, :(len(x[0])//2)], x[:, (len(x[0])//2):]
        stddev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(stddev)
        if self.training:
            z = mean + epsilon * stddev
        else:
            z = mean
        dot_product = z @ self.items.T

        try:
            if self.normalize:
                out = F.normalize(z)@ (F.normalize(self.items).T)  / self.tau 
            
            else:
                out = z @ self.items.T 

        except Exception as e :
            print(e)
            print("new_input.shape:", new_input.shape)

        new_output = out

        var_square = torch.exp(logvar)
        kl = 0.5 * torch.mean(torch.sum(mean ** 2 + var_square - 1. - logvar, dim=-1))

        return z, new_output, kl, batch_input0 

    def getUsersRating(self, users):
        self.eval()
        users = users.cpu()
        test_batch_size = users.shape[0]
        

        if world.dataset in ['amazon-book', 'ml-20m']:  #hard to load into GPU
            rating_matrix_batch = self.R[users]

            batch_size = int(self.config['vae_batch_size']/4)
            num_users = len(rating_matrix_batch)
            n_batch = math.ceil(num_users / batch_size)
            predict_out_arr = []

            torch.autograd.set_detect_anomaly(True)
            for idx in range(n_batch):

                start_idx = idx * batch_size
                end_idx = min(start_idx + batch_size, num_users)
                batch_users = users[start_idx:end_idx]

                rating_matrix_batch = self.R[batch_users]

                _, predict_out, kl, _ = self.forward_kernel_1226(rating_matrix_batch)
                
                predict_out_arr.append(predict_out)

            predict_out = torch.cat(predict_out_arr, dim =0).to(world.device)
            rating_matrix_batch = self.R[users].to(world.device) 

        else:
            rating_matrix_batch = self.R[users].to(world.device)
            _, predict_out, _, _ = self.forward_kernel_1226(rating_matrix_batch)

        sample_num = 1
        for user in users:
            tmp = predict_out[user%test_batch_size] * rating_matrix_batch[user%test_batch_size] 

            items_prob_index = torch.nonzero(tmp).cpu()
            items_prob = tmp[items_prob_index]

            items_prob = F.normalize(items_prob, p=1.0, dim=0)  
            items_prob = items_prob.reshape(-1)
            
            
            items = torch.nonzero(rating_matrix_batch[user%test_batch_size]).cpu().numpy()
            items = items.reshape(-1)

            tmp = random.choices(items, weights=items_prob, k=sample_num)

            if len(tmp) == 0:
                tmp = np.random.choice(range(self.num_items), 1)

            self.R2[user] = tmp[0]

        return predict_out


    @staticmethod
    def calculate_mult_log_likelihood(prediction, label, users, items, ii_sim):
        log_softmax_output = torch.nn.LogSoftmax(dim=-1)(prediction)
        log_likelihood_O = -torch.mean(torch.sum(log_softmax_output * label, 1))
        log_likelihood = log_likelihood_O
        log_likelihood_I = -torch.sum(log_softmax_output[(users, items)] * ii_sim) / prediction.shape[0]
        log_likelihood += log_likelihood_I
        return log_likelihood, log_likelihood_O, log_likelihood_I

    @staticmethod
    def calculate_mult_log_likelihood_simple(prediction, label):        
        
        log_softmax_output = torch.nn.LogSoftmax(dim=-1)(prediction)        
        log_likelihood = -torch.mean(torch.sum(log_softmax_output * label, 1))
    
        return log_likelihood, log_likelihood, log_likelihood*0
    
    
    def reg_loss(self):
        """
        Return the L2 regularization of weights. Code is implemented according to tensorflow.
        """
        reg_list = [0.5 * torch.sum(m.weight ** 2) for m in self.modules() if isinstance(m, nn.Linear)]
        reg_list = reg_list + [0.5 * torch.sum(self.items ** 2)] 
        reg = 0.
        for val in reg_list:
            reg += val
    
        return reg

    def train_one_epoch(self):
        self.train()
        users = np.arange(self.num_users)
        batch_size = self.config['vae_batch_size']
        n_batch = math.ceil(self.num_users / batch_size)
        loss_dict = {}
        neg_ll_list = []
        kl_list = []
        reg_list = []

        self.epoch += 1

        torch.autograd.set_detect_anomaly(True)
        for idx in range(n_batch):

            start_idx = idx * batch_size
            end_idx = min(start_idx + batch_size, self.num_users)
            batch_users = users[start_idx:end_idx]

            if world.dataset in ['amazon-book', 'ml-20m']: #especially deal with the two large datasets
                rating_matrix_batch = self.R[batch_users].to(world.device)
                num_users = len(rating_matrix_batch)
            else:
                rating_matrix_batch = self.R[batch_users].to(world.device)

            
            if self.config['sampling'] == 1:
                samplingEpoch = 0 
            else:
                samplingEpoch = self.config['epochs']

            if self.epoch >=  samplingEpoch:  #choose sampling or not
                rating_matrix_batch2 = torch.LongTensor(self.R2[batch_users]).to(world.device)
                _, predict_out, kl, _ = self.forward_kernel(rating_matrix_batch, rating_matrix_batch2)
            else:
                _, predict_out, kl, _ = self.forward_kernel_1226(rating_matrix_batch)


            if world.dataset in ['amazon-book', 'ml-20m']:  #especially deal with the two large datasets
                rating_matrix_batch = self.R[batch_users].to(world.device)
            
            neg_ll, log_likelihood_O, log_likelihood_I = self.calculate_mult_log_likelihood_simple(predict_out, rating_matrix_batch)  #self.PRINT 传递epoch信息
            
            #############################################################################
            loss = neg_ll + self.anneal_ph * kl + self.lam * self.reg_loss()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if idx == n_batch - 1:
                print('log_likelihood_O: %.4f'%(log_likelihood_O.item()-log_likelihood_I.item()), end=", ")
                print('log_likelihood_I: %.4f'%log_likelihood_I.item(), end=", ")
                print("anneal_ph *KL: %.4f"%(self.anneal_ph *kl.item()), end=", ")
                print("lam * reg_loss: %.4f"%(self.lam *self.reg_loss().item()))

            neg_ll_list.append(neg_ll.item())
            kl_list.append(kl.item())
            reg_list.append(self.reg_loss().item())

        loss_dict['neg_ll'] = neg_ll_list
        loss_dict['kl'] = kl_list
        loss_dict['reg'] =  reg_list
        return loss_dict

    
    def get_topk_ii(self):
        """
        For every item, get its topk similar items according to the co-occurrent matrix.
        """
        save_path = f'./pretrained/{world.dataset}/{world.model_name}'
        ii_sim_mat_path = save_path + '/ii_sim_mat_'+ str(self.topk) +'.pkl'
        ii_sim_idx_mat_path = save_path + '/ii_sim_idx_mat_'+ str(self.topk) +'.pkl'
        gram_matrix_path = save_path + '/gram_matrix.pkl'
        if not os.path.exists(gram_matrix_path):
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            adj_mat = self.dataset.UserItemNet
            row_sum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(row_sum, -0.5).flatten()
            d_inv[np.isposinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_mat = d_mat.dot(adj_mat)
            col_sum = np.array(adj_mat.sum(axis=0))
            d_inv = np.power(col_sum, -0.5).flatten()
            d_inv[np.isposinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_mat = norm_mat.dot(d_mat).astype(np.float32)

            gram_matrix = norm_mat.T.dot(norm_mat).toarray()

            print("Successfully created the co-occurrence matrix!")
            self.ii_sim_mat = torch.zeros(self.num_items, self.topk)
            self.ii_sim_idx_mat = torch.zeros(self.num_items, self.topk)
            for iid in range(self.num_items):
                row = torch.from_numpy(gram_matrix[iid])
                sim, idx = torch.topk(row, self.topk)
                self.ii_sim_mat[iid] = sim
                self.ii_sim_idx_mat[iid] = idx
                if iid % 15000 == 0:
                    print('Getting {} items\' topk done'.format(iid))
            self.ii_sim_mat = self.ii_sim_mat
            self.ii_sim_idx_mat = self.ii_sim_idx_mat.numpy()
            self.gram_matrix = gram_matrix
            joblib.dump(self.ii_sim_mat, ii_sim_mat_path, compress=3)
            joblib.dump(self.ii_sim_idx_mat, ii_sim_idx_mat_path, compress=3)
            joblib.dump(gram_matrix, gram_matrix_path, compress=3)
        else:
            self.ii_sim_mat = joblib.load(ii_sim_mat_path)
            self.ii_sim_idx_mat = joblib.load(ii_sim_idx_mat_path)
            self.gram_matrix = joblib.load(gram_matrix_path)

        if world.LOAD ==1:  
            weight_len = self.encoder[0].weight.shape[0]//2
            print(weight_len)
            gram_matrix = self.encoder[0].weight[:weight_len,:].T.mm(self.encoder[0].weight[:weight_len,:])   

            try:
                f = open("items_embedding_VKDE.pkl",'rb+')
                item_embedding = pickle.load(f)
                gram_matrix = torch.from_numpy(item_embedding.dot(item_embedding.T))
        
                self.gram_matrix = gram_matrix

                print(gram_matrix.cpu().shape, self.items.shape)

            except: 
                print("EOFError!")