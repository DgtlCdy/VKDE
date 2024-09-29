import os

from torch.serialization import save
import world
import utils
from world import cprint
import torch
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sn
sn.set()
import joblib
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register as register
from register import dataset
import model


if __name__ == '__main__':
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    if isinstance(Recmodel, nn.Module):
        Recmodel = Recmodel.to(world.device)

    weight_file = utils.getFileName()
    print(f"load and save to {weight_file}")
    if world.LOAD:
        try:
            Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
            # Recmodel.get_topk_ii() # 这里是不是重复了？
            world.cprint(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    Neg_k = 1

    # init tensorboard
    if world.tensorboard:
        w : SummaryWriter = SummaryWriter(
                                        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                        )
    else:
        w = None
        world.cprint("not enable tensorflowboard")

    best_results = {'precision': np.zeros(len(world.topks)),
                    'recall': np.zeros(len(world.topks)),
                    'ndcg': np.zeros(len(world.topks))}
    best_epoch = 0

    try:
        if(world.simple_model != 'none'):   #for lgn-ide, gf-cf
            epoch = 0
            cprint("[TEST]")
            adj_mat = dataset.UserItemNet.tolil()
            if(world.simple_model == 'lgn-ide'):
                lm = model.LGCN_IDE(adj_mat)
                lm.train()
            elif(world.simple_model == 'gf-cf'):
                lm = model.GF_CF(adj_mat)
                lm.train()
            def ensure_dirs(path):
                if not os.path.exists(path):
                    os.makedirs(path)
            Procedure.Test(dataset, lm, epoch, w, world.config['multicore'])
        else:  
            if world.model_name == 'lgn':
                for epoch in range(world.TRAIN_epochs):
                    if epoch %10 == 0:
                        cprint("[TEST]")
                        results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                        if results['recall'][0] > best_results['recall'][0]:
                            best_results = results
                    output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
                    print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
                    torch.save(Recmodel.state_dict(), weight_file)    
                print('best results: %s'%(best_results))
            else:
                # 开始训练
                for epoch in range(world.TRAIN_epochs):
                    utils.print_log(f'start train in epoch-{epoch}') # testonly
                    t0 = time.time()
                    batch_loss: dict = Recmodel.train_one_epoch()  
                    utils.print_log(f'end train in epoch-{epoch}') # testonly
                    elapsed_time = time.time() - t0
                    if (epoch % 50 == 0) or (epoch == world.TRAIN_epochs - 1):
                        cprint("[TEST]")
                        utils.print_log(f'start Test in epoch-{epoch}') # testonly
                        results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                        utils.print_log(f'end Test in epoch-{epoch}') # testonly
                        if results['recall'][0] > best_results['recall'][0]:
                            best_results = results
                            best_epoch = epoch
                            if isinstance(Recmodel, nn.Module):
                                torch.save(Recmodel.state_dict(), weight_file)
                    if batch_loss:
                        if batch_loss.get('total_loss'):
                            print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] Elapsed time: {elapsed_time:<.1f} Total_loss: {np.sum(batch_loss["total_loss"]):<.2f}')
                        elif batch_loss.get('neg_ll'):
                            print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] Elapsed time: {elapsed_time:<.1f} Neg_ll: {np.sum(batch_loss["neg_ll"]):<.2f}')
                        else:
                            pass
                    else:
                        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] Elapsed time: {elapsed_time}')
                    utils.print_log(f'end for in epoch-{epoch}') # testonly

                print('best results: %s'%(best_results))
                print('best epoch is: %s'%(best_epoch+1))

                def mplot(data, name):
                    plt.xlabel('epoch')
                    plt.ylabel(f'{name}')
                    plt.title(f'{world.model_name}-{world.dataset}-{name}')
                    plt.plot(data)
                    if len(eval(world.args.enc_dims)) > 1:
                        save_name = f'{world.model_name}-{world.dataset}-{name}-{world.args.enc_dims}-{world.args.act_vae}.jpg'
                    else:
                        save_name = f'{world.model_name}-{world.dataset}-{name}-{world.args.enc_dims}.jpg'
                    plt.savefig(save_name)
                    plt.clf()
                def save_data(data, path, name):
                    save_name = path + f'{world.model_name}-{world.dataset}-{name}-{world.args.enc_dims}.pkl'
                    joblib.dump(data, save_name, compress=3)

                def ensure_dirs(path):
                    if not os.path.exists(path):
                        os.makedirs(path)

    finally:
        if world.tensorboard:
            w.close()
