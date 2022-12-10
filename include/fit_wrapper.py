from torch.autograd import Variable
import torch
import torch.optim
import copy
import numpy as np
from scipy.linalg import hadamard

from .helpers import *


dtype = torch.cuda.FloatTensor
#dtype = torch.FloatTensor
           

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=500):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.65**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer



def fit(net,
        img_noisy_var,
        num_channels,
        img_clean_var,
        num_iter = 5000,
        LR = 0.01,
        OPTIMIZER='LBFGS',
        opt_input = False,
        reg_noise_std = 0,
        reg_noise_decayevery = 100000,
        mask_var = None,
        apply_f = None,
        lr_decay_epoch = 0,
        net_input = None,
        net_input_gen = "random",
        device=None,
        find_best=False,
        weight_decay=0,
       ):

    if net_input is not None:
        print("input provided")
    else:
        # feed uniform noise into the network 
        totalupsample = 2**len(num_channels)
        # width = 5
        # height = 10
        # width and height for cropped Marmousi Model (70, 70)
        
        # print(img_clean_var.data[0].shape)
        print("img_clean_var", img_clean_var.data.shape)
       
        width = int(img_clean_var.data.shape[2]/32)
        height = int(img_clean_var.data.shape[3]/32)
        shape = [1,num_channels[0], width, height]
        print("Uniform Noise Shape: ", shape)
        net_input = Variable(torch.zeros(shape))
        net_input.data.uniform_(1000,5000)

    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()
    print(net.parameters())
    
    p = [x for x in net.parameters() ]

    if(opt_input == True): # optimizer over the input as well
        net_input.requires_grad = True
        p += [net_input]

    mse_wrt_noisy = np.zeros(num_iter)
    mse_wrt_truth = np.zeros(num_iter)

    if OPTIMIZER == 'SGD':
        print("optimize with SGD", LR)
        optimizer = torch.optim.SGD(p, lr=LR,momentum=0.9,weight_decay=weight_decay)
    elif OPTIMIZER == 'adam':
        print("optimize with adam", LR)
        optimizer = torch.optim.Adam(p, lr=LR,weight_decay=weight_decay)
    elif OPTIMIZER == 'LBFGS':
        print("optimize with LBFGS", LR)
        optimizer = torch.optim.LBFGS(p, lr=LR)

    mse = torch.nn.MSELoss()  
    
    print("img_noisy_var", img_noisy_var.shape)
    noise_energy = mse(img_noisy_var, img_clean_var)

    if find_best:
        best_net = copy.deepcopy(net)
        best_mse = 1000000.0
    
    epoch_loss_list = []
    for i in range(num_iter):
        
        if lr_decay_epoch is not 0:
            optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch)
        if reg_noise_std > 0:
            if i % reg_noise_decayevery == 0:
                reg_noise_std *= 0.7
            net_input = Variable(net_input_saved + (noise.normal_() * reg_noise_std))
        
        def closure():
            
            optimizer.zero_grad()
            out = net(net_input.type(dtype)).to(device)
            
            # print("out",out.shape)
            # print(img_noisy_var.shape)

            # training loss 
            if mask_var is not None:
                loss = mse( out * mask_var , img_noisy_var * mask_var )
            elif apply_f:
                loss = mse( apply_f(out) , img_noisy_var )
            else:
                
                loss = mse(out, img_noisy_var)
                
            loss.backward(retain_graph=True)
            
            mse_wrt_noisy[i] = loss.data.cpu().numpy()
            
            
            # the actual loss 
            true_loss = mse(Variable(out.data, requires_grad=False), img_clean_var)
            mse_wrt_truth[i] = true_loss.data.cpu().numpy()
            
            
            if i%50 == 0:
                iter_out = net(net_input.type(dtype)).to(device)
                iter_loss = mse(iter_out, img_noisy_var)
                epoch_loss_list.append(iter_loss.data)
                print('Iteration %f     Train loss %f' %(i, iter_loss.data))
                
            
            return loss

        
        loss = optimizer.step(closure)
        
         
        if find_best:
            # if training loss improves by at least one percent, we found a new best net
            if best_mse > 1.005*loss.data:
                best_mse = loss.data
                best_net = copy.deepcopy(net)
                 
        
    if find_best:
        net = best_net
    return mse_wrt_noisy, mse_wrt_truth, net_input_saved, net, epoch_loss_list

