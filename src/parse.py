'''
Created on Aug 22, 2022
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go VKDE")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of VKDE")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of VKDE")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='gowalla',
                        help="available datasets: [yelp2018, amazon-book, video-games]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=0,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model')
    parser.add_argument('--simple_model', type=str, default='none', help='simple-rec-model, support [none, lgn-ide, gf-cf]')
    parser.add_argument('--enc_dims', type=str, default="[64]", help="layers dimensions of MultVAE")
    parser.add_argument('--kl_anneal', type=float, default=0.2, help="kl annealing, beta for the kl term in the ELBO of MultVAE")
    parser.add_argument('--act_vae', type=str, default='tanh', choices=['tanh', 'sigmoid', 'relu', 'leakyrelu'], help="activation function of MultVAE")
    parser.add_argument('--not_alternating_recvae', action='store_true', help="Whether not to apply alternating training")
    parser.add_argument('--gamma_simplex', type=float, default=1, help="Used to balance the constributions of embeddings of id and neighbors.")
    parser.add_argument('--tau_model2', type=float, default=0.1, help="A temperature hyperparameter for training model2.")
    parser.add_argument('--reg_model2', type=float, default=0.001, help="Regularization weight for training model2.")
    parser.add_argument('--dropout_model2', type=float, default=0.5, help="Dropout rate for training model2.")
    parser.add_argument('--alpha_model2', type=float, default=1e-4, help="Weight of item-item similarity for training model2.")
    parser.add_argument('--normalize_model2', type=int, default=1, help="Whether normalize the user and item embeddings for training model2.")
    parser.add_argument('--dropout_multvae', type=float, default=0.5, help="Dropout rate for training MultVAE.")
    parser.add_argument('--encoder_ctrl_model3', type=int, default=2, choices=[0, 1, 2], help="Choose linear, non-linear or hybrid for encoder.")
    parser.add_argument('--decoder_ctrl_model3', type=int, default=0, choices=[0, 1], help="Choose linear, non-linear for deocder.")
    parser.add_argument('--topK_model3', type=int, default=10, help="Choose top M most similar items for each interacted items.")
    parser.add_argument('--cuda', type=int,default=0)
    parser.add_argument('--dfac_macridvae', type=int,default=64,
                        help="the embedding size of MacridVAE")
    parser.add_argument('--FP', type=int,default=0,
                        help="consider false positive or not")
    parser.add_argument('--FN', type=int,default=0,
                        help="consider False Negative or not")
    parser.add_argument('--weight_scale', type=int, default=40,
                        help="the scale of softmax_output-softmax_output_mean")
    parser.add_argument('--topK', type=int, default=-1,
                        help="select topK items to  calculate softmax")
    parser.add_argument('--kernel', type=str, default='gram',
                        help="kernel function, support [gram, cos, emb]")
    parser.add_argument('--sampling', type=int, default=0)
                    
    return parser.parse_args()
