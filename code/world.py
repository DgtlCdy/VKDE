'''
Created on Aug 22, 2022
'''

import os
from os.path import join
import torch
from torch import nn
from enum import Enum
from parse import parse_args
import multiprocessing
import platform

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
torch.cuda.device_count()
args = parse_args()

# ROOT_PATH = os.path.dirname(os.path.dirname(__file__)) # 这个命令，Linux下获取不到
if platform.system() == 'Windows':
    ROOT_PATH = 'C:/codes/VKDE'
else:
    ROOT_PATH = '/data0/dengchao/VKDE'

CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
LOG_PATH = join(ROOT_PATH, 'log')
TEST_RESULT_PATH = join(ROOT_PATH, 'test_result')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')

import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book',  'video_games']
all_models  = ['mf', 'lgn', 'MultVAE', 'lgn_listwise', 'MultDAE', 'lgn_vae', 'RecVAE', 
               'mfmodel', 'jova', 'locavae', 'evcf', 'MacridVAE', 'elfm', 'BUIR_ID', 
               'BUIR_NB', 'mf_listwise', 'simplex', 'ultragcn',
               'VAEplus', 'VAE_Graph', 'VKDE']

config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False

# configurations of MultVAE
if args.model in ['MultVAE', 'MultDAE', 'lgn_vae', 'lgn_listwise']:
    config['vae_batch_size'] = 256 # default 256
    config['vae_reg_param'] = 0.0 # default 0.001
    config['kl_anneal'] = args.kl_anneal # default 0.2
    config['enc_dims'] = args.enc_dims # default [64]
    config['vae_lr'] = 0.001 # default 0.001
    config['dropout_multvae'] = args.dropout_multvae
    if args.act_vae == 'tanh':
        config['act_vae'] = nn.Tanh()
    elif args.act_vae == 'relu':
        config['act_vae'] = nn.ReLU()
    elif args.act_vae == 'sigmoid':
        config['act_vae'] = nn.Sigmoid()
    elif args.act_vae == 'leakyrelu':
        config['act_vae'] = nn.LeakyReLU()
    else:
        raise NotImplementedError
    

    config['tau_macridvae'] = args.tau_model2
    config['tau_model2'] = args.tau_model2
    config['alpha_model2'] = args.alpha_model2
    config['dropout_model2'] = args.dropout_model2
    config['vae_lr'] = args.lr # default 0.001
    config['vae_reg_param'] = args.reg_model2 # default 0.001
    config['FP'] = args.FP
    config['FN'] = args.FN
    config['weight_scale'] = args.weight_scale
    config['topK'] = args.topK

# configurations of RecVAE
if args.model == 'RecVAE':
    config['hidden_dim_recvae'] = 128
    config['latent_dim_recvae'] = 64
    config['not_alternating_recvae'] = args.not_alternating_recvae
    config['lr_recvae'] = 5e-4
    config['batch_size_recvae'] = 500
    config['beta_recvae'] = None
    config['gamma_recvae'] = 0.005
    config['n_enc_epochs_recvae'] = 3
    config['n_dec_epochs_recvae'] = 1

# configurations of MFModel
if args.model == 'mfmodel':
    config['reg_mf'] = 0.01
    config['latent_dim_mf'] = 16
    config['stddev_mf'] = 0.1
    config['lr_mf'] = 0.007
    config['num_neg_mf'] = 10

if args.model == 'jova':
    config['lr_jova'] = 0.003
    config['using_hinge_jova'] = 1
    config['beta_jova'] = 0.01
    config['batch_size_jova'] = 1500
    config['inter_dim_jova'] = 320
    config['latent_dim_jova'] = 80

# configurations of LOCA_VAE
if args.model == 'locavae':
    config['num_local_locavae'] = 300
    config['anchor_selection_locavae'] = 'coverage'
    config['dist_type_locavae'] = 'arccos'
    config['kernel_type_locavae'] = 'epanechnikov'
    config['train_h_locavae'] = 1
    config['test_h_locavae'] = 0.8
    config['embedding_locavae'] = 'MultVAE'
    config['num_local_threads_locavae'] = 5
    config['batch_size_locavae'] = 512 # default 512
    config['test_batch_size_locavae'] = 1024
    config['lr_locavae'] = 0.001
    
    config['num_epochs_locavae'] = 200
    config['print_step_locavae'] = 1
    config['test_step_locavae'] = 10
    config['test_from_locavae'] = 1

    config['enc_dims_locavae'] = "[64]"
    config['total_anneal_steps_locavae'] = 200000
    config['anneal_cap_locavae'] = 0.2
    config['dropout_locavae'] = 0.5 # default 0.2
    config['reg_locavae'] = 0.0 # default 0
    config['verbose_locavae'] = 0

    config['latent_dim_locavae'] = eval(config['enc_dims_locavae'])[-1]

if args.model == 'evcf':
    config['hidden_size_evcf'] = 600
    config['gated_evcf'] = True
    config['z1_size_evcf'] = 200
    config['z2_size_evcf'] = 200
    config['num_layers_evcf'] = 2
    config['number_components_evcf'] = 1000
    config['use_training_data_init_evcf'] = False
    config['pseudoinputs_mean_evcf'] = 0.05
    config['pseudoinputs_std_evcf'] = 0.01
    config['cuda_evcf'] = True
    config['input_type_evcf'] = "multinomial"
    config['lr_evcf'] = 0.0005
    config['batch_size_evcf'] = 200
    config['warmup_evcf'] = 100
    config['max_beta_evcf'] = 0.3
    config['latent_dim_evcf'] = config['z1_size_evcf']

if args.model == 'MacridVAE':
    config['kfac_macridvae'] = 7 # default 7 
    config['dfac_macridvae'] = args.dfac_macridvae
    config['lam_macridvae'] = 0.0
    config['lr_macridvae'] = 1e-3
    config['tau_macridvae'] = 0.1
    config['nogb_macridvae'] = True # default False
    config['std_macridvae'] = 0.075 # default 0.075
    config['batch_size_macridvae'] = 256
    config['beta_macridvae'] = 0.2
    config['keep_prob_macridvae'] = 0.5

    config['latent_dim_macridvae'] = config['dfac_macridvae'] # for generating the model saving path in utils.py


    config['tau_macridvae'] = args.tau_model2
    config['tau_model2'] = args.tau_model2
    config['alpha_model2'] = args.alpha_model2
    config['dropout_model2'] = args.dropout_model2
    config['vae_lr'] = args.lr # default 0.001
    config['vae_reg_param'] = args.reg_model2 # default 0.001
    config['FP'] = args.FP
    config['FN'] = args.FN
    config['weight_scale'] = args.weight_scale
    config['topK'] = args.topK

if args.model == 'elfm':
    config['num_negs_ELFM'] = 1000
    config['batch_size_ELFM'] = 256
    config['lr_ELFM'] = 1e-3
    config['lr_decay_ELFM'] = 0.1
    config['lr_decay_steps_ELFM'] = 50
    config['alpha_ELFM'] = 0.1 # {0.1, 0.05}
    config['lam_ELFM'] = 1e-4 # {1e-6, 1e-4}
    config['num_clusters_ELFM'] = 256 # {512, 1024, 2048}
    config['latent_dim_ELFM'] = 64

if args.model in ['BUIR_ID', 'BUIR_NB']:
    config['latent_dim_BUIR_ID'] = 64
    config['momentum_BUIR_ID'] = 0.995
    config['batch_size_BUIR_ID'] = 1024
    config['lr_BUIR_ID'] = 0.001
    config['weight_decay_BUIR_ID'] = 1e-4

    # Additional hyperparameter for BUIR_NB
    config['n_layers_BUIR_NB'] = 3
    config['drop_flag_BUIR_NB'] = True

if args.model == 'mf_listwise':
    config['lr_mf_listwise'] = 0.001
    config['reg_mf_listwise'] = 0.001
    config['latent_dim_mf_listwise'] = 64
    config['batch_size_mf_listwise'] = 256

if args.model == 'simplex':
    config['latent_dim_simplex'] = args.recdim
    config['history_max_len_simplex'] = 500
    config['truncating_simplex'] = 'pre'
    config['padding_simplex'] = 'pre'
    config['num_negs_simplex'] = 1000 # 1000 for Yelp2018
    config['sample_probs_simplex'] = None
    config['ignore_pos_items_simplex'] = False
    config['gamma_simplex'] = args.gamma_simplex
    config['aggregator_simplex'] = 'mean'
    config['attention_dropout_simplex'] = 0
    config['net_dropout_simplex'] = 0.1
    config['lr_simplex'] = 1.e-4
    config['margin_simplex'] = 0.9 # 0.9
    config['negative_weight_simplex'] = 150 # 150 for Yelp2018
    config['batch_size_simplex'] = 512
    config['similarity_score_simplex'] = 'cosine'
    config['embedding_regularizer_simplex'] = [1.e-8]
    config['net_regularizer_simplex'] = None
    config['embedding_initializer_simplex'] = "lambda w: nn.init.normal_(w, std=1e-4)"
    config['max_gradient_norm_simplex'] = 10.

if args.model == 'ultragcn':
    config['lr_ultragcn'] = 1e-4
    config['batch_size_ultragcn'] = 512
    config['w1_ultragcn'] = 1e-6
    config['w2_ultragcn'] = 1
    config['w3_ultragcn'] = 1e-6
    config['w4_ultragcn'] = 1
    config['num_negs_ultragcn'] = 1500
    config['neg_weight_ultragcn'] = 300
    config['gamma_ultragcn'] = 1e-4
    config['lambda_ultragcn'] = 5e-4
    config['sampling_sift_pos_ultragcn'] = False
    config['latent_dim_ultragcn'] = 64
    config['ii_neighbor_num_ultragcn'] = 10

if args.model in ['VAEplus','VAE_Graph','VKDE','VAEKernelPlus','VAEKernelEmb']:
    config['tau_model2'] = args.tau_model2
    config['alpha_model2'] = args.alpha_model2
    config['dropout_model2'] = args.dropout_model2
    config['normalize_model2'] = args.normalize_model2
    config['encoder_ctrl_model3'] = args.encoder_ctrl_model3
    config['decoder_ctrl_model3'] = args.decoder_ctrl_model3
    config['topK_model3'] = args.topK_model3
    if args.dataset == "amazon-book":
        config['vae_batch_size'] = 64 # default 256
    else:
        config['vae_batch_size'] = 1024 # default 256
    config['vae_reg_param'] = args.reg_model2 # default 0.001
    config['kl_anneal'] = args.kl_anneal # default 0.2
    config['enc_dims'] = args.enc_dims # default [64]
    config['vae_lr'] = args.lr # default 0.001
    config['latent_dim_mymodel'] = eval(args.enc_dims)[-1]
    config['FP'] = args.FP
    config['FN'] = args.FN
    config['weight_scale'] = args.weight_scale
    config['topK'] = args.topK
    config['kernel'] = args.kernel
    config['sampling'] = args.sampling # if sampling interest from all
    config['epochs'] = 100 # 新增的，作用待定

    if args.act_vae == 'tanh':
        config['act_vae'] = nn.Tanh()
    elif args.act_vae == 'relu':
        config['act_vae'] = nn.ReLU()
    elif args.act_vae == 'sigmoid':
        config['act_vae'] = nn.Sigmoid()
    elif args.act_vae == 'leakyrelu':
        config['act_vae'] = nn.LeakyReLU()
    else:
        raise NotImplementedError



GPU = torch.cuda.is_available()
if args.cuda!= -1 and GPU:
    device = torch.device('cuda:'+ str(args.cuda))
else:
    device = torch.device("cpu")

CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
simple_model = args.simple_model
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")


TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗   ██╗██╗  ██╗██████╗ ███████╗
██║   ██║██║ ██╔╝██╔══██╗██╔════╝
██║   ██║█████╔╝ ██║  ██║█████╗  
╚██╗ ██╔╝██╔═██╗ ██║  ██║██╔══╝  
 ╚████╔╝ ██║  ██╗██████╔╝███████╗
  ╚═══╝  ╚═╝  ╚═╝╚═════╝ ╚══════╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=VKDE%0A
# print(logo)
