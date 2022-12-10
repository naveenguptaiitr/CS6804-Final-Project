import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Custom Dataloader for enumerating using PyTorch dataloader
class DatasetLoader(Dataset):
    
    def __init__(self, amps, vel):
        self.amps = amps
        self.vel = vel
    
    
    def __getitem__(self, idx):
        return (self.amps[idx], self.vel[idx])
    
    def __len__(self):
        return len(self.amps)
    

class FeedForwardNetwork(nn.Module):
    
    def __init__(self):
        super(FeedForwardNetwork, self).__init__()
        
        self.layer1 = nn.Linear(200, 150)
        self.layer2 = nn.Linear(150, 120)
        self.layer3 = nn.Linear(120, 100)
        self.layer4 = nn.Linear(100, 70)
        
    def forward(self, x):
        
        x = self.layer1(x)
        x = F.relu(x)
        
        x = self.layer2(x)
        x = F.relu(x)
        
        x = self.layer3(x)
        x = F.relu(x)
        
        x = self.layer4(x)
        x = F.relu(x)
        
        return x
    

class ConvolutionalNetwork(nn.Module):
    
    def __init__(self, num_channels):
        super(ConvolutionalNetwork, self).__init__()
        
        layers = []
        
        self.layer1 = nn.Sequential(nn.Conv2d(num_channels[0], num_channels[1],
                                                kernel_size=(5,5), stride=(1,1), padding=(0,0)),
                                     nn.BatchNorm2d(num_channels[1]),
                                     nn.ReLU())
        
        self.layer2 = nn.Sequential(nn.Conv2d(num_channels[1], num_channels[2],
                                                kernel_size=(5,5), stride=(1,1), padding=(0,0)),
                                     nn.BatchNorm2d(num_channels[2]),
                                     nn.ReLU())
        
        self.layer3 = nn.Sequential(nn.Conv2d(num_channels[2], num_channels[3],
                                                kernel_size=(5,5), stride=(1,1), padding=(0,0)),
                                     nn.BatchNorm2d(num_channels[3]),
                                     nn.ReLU())
        
        self.layer4 = nn.Sequential(nn.Conv2d(num_channels[3], num_channels[4],
                                                kernel_size=(5,5), stride=(1,1), padding=(0,0)),
                                     nn.BatchNorm2d(num_channels[4]),
                                     nn.ReLU())
        
        self.layer5 = nn.Sequential(nn.Conv2d(num_channels[4], num_channels[5],
                                                kernel_size=(3,3), stride=(1,1), padding=(0,0)),
                                     nn.BatchNorm2d(num_channels[5]),
                                     nn.ReLU())
        
        self.layer6 = nn.Sequential(nn.Conv2d(num_channels[5], num_channels[6],
                                                kernel_size=(3,3), stride=(1,1), padding=(0,0)),
                                     nn.BatchNorm2d(num_channels[6]),
                                     nn.ReLU())
        
        layers.append(self.layer1)
        layers.append(self.layer2)
        layers.append(self.layer3)
        layers.append(self.layer4)
        layers.append(self.layer5)
        layers.append(self.layer6)
        
        self.cnn_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.cnn_layers(x)

        return x
    
    


class DecoderNetwork(nn.Module):
    
    def __init__(self, nums):
        super(DecoderNetwork, self).__init__()
        
        layers = []
        
        self.layer1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                    self.conv(nums[0], nums[1], kernel_size=1, stride=1, 
                                              pad='reflection'),
                                   nn.BatchNorm2d(nums[1]),
                                   nn.ReLU())
        
        self.layer2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                    self.conv(nums[1], nums[2], kernel_size=1, stride=1, 
                                              pad='reflection'),
                                   nn.BatchNorm2d(nums[2]),
                                   nn.ReLU())
        
        self.layer3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                    self.conv(nums[2], nums[3], kernel_size=1, stride=1, 
                                              pad='reflection'),
                                   nn.BatchNorm2d(nums[3]),
                                   nn.ReLU())
        
        self.layer4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                    self.conv(nums[3], nums[4], kernel_size=1, stride=1, 
                                              pad='reflection'),
                                   nn.BatchNorm2d(nums[4]),
                                   nn.ReLU())
        
        layers.append(self.layer1)
        layers.append(self.layer2)
        layers.append(self.layer3)
        layers.append(self.layer4)
        
        self.cnn_layers = nn.Sequential(*layers)
    
    def forward(self, x):
        
        x = self.cnn_layers(x)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)

        return x
    
    def conv(self, in_f, out_f, kernel_size, stride=1, pad='zero'):
        padder = None
        to_pad = int((kernel_size - 1) / 2)
        if pad == 'reflection':
            padder = nn.ReflectionPad2d(to_pad)
            to_pad = 0

        convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=False)

        layers = filter(lambda x: x is not None, [padder, convolver])
        return nn.Sequential(*layers)
        
        
def oldtrain(net, img_noisy_var, img_clean_var, num_iter,
         LR=1e-4, OPTIMIZER='ADAM',device=None):
    
    # if using feed-forward network
    width = 70
    height = 200

    shape = [width, height]

    net_input = Variable(torch.zeros(shape))
    net_input.data.uniform_(0, 1)
        
    net_input_saved = net_input.data.clone()
    
    p = [x for x in net.parameters()]
    
    mse_wrt_noisy = np.zeros(num_iter)
    mse_wrt_clean = np.zeros(num_iter)
    
    if OPTIMIZER=='SGD':
        optimizer = torch.optim.SGD(p, lr=LR, momentum=0.9)
        
    elif OPTIMIZER=='ADAM':
        optimizer = torch.optim.Adam(p, lr=LR)
    
    else:
        optimizer = torch.optim.LBFGS(p, lr=LR)
        
    loss_fn = torch.nn.MSELoss()  
    
    dtype = torch.cuda.FloatTensor
    net_input = net_input.type(dtype).to(device)
    
    
    loss_curve = []

    for i in range(num_iter):
        optimizer.zero_grad()

        out = net(net_input)

        loss = loss_fn(out, img_noisy_var)
        true_loss = loss_fn(out, img_clean_var)

        loss.backward() 
        optimizer.step()

        mse_wrt_noisy[i] = loss.data.cpu().numpy()
        mse_wrt_clean[i] =true_loss.data.cpu().numpy()

        if i%50 == 0:
            curr_output = net(net_input)
            curr_loss = loss_fn(curr_output, img_noisy_var)
            loss_curve.append(curr_loss.data)
 
            
    return mse_wrt_noisy, mse_wrt_clean, net_input_saved,  net
        
    
def train(net, img_noisy_var, img_clean_var, num_iter,
         LR=1e-4, OPTIMIZER='ADAM', batch_size=1, model_type='cnn', device=None):
    
        
    if model_type == 'ffn':
        # if using feed-forward network
        width = 70
        height = 200
        
        shape = [width, height]

    else:
        # if using convolutional network
        width = 90
        height = 90
        
        total_vp = img_noisy_var.shape[0]
        
        shape = [total_vp, 1, width, height]

    net_input = Variable(torch.zeros(shape))
    net_input.data.uniform_(0, 1)
        
    net_input_saved = net_input.data.clone()
    
    p = [x for x in net.parameters()]
    
    mse_wrt_noisy = np.zeros(num_iter)
    mse_wrt_clean = np.zeros(num_iter)
    
    if OPTIMIZER=='SGD':
        optimizer = torch.optim.SGD(p, lr=LR, momentum=0.9)
        
    elif OPTIMIZER=='ADAM':
        optimizer = torch.optim.Adam(p, lr=LR)
    
    else:
        optimizer = torch.optim.LBFGS(p, lr=LR)
        
    loss_fn = torch.nn.MSELoss()  
    
    dtype = torch.cuda.FloatTensor
    net_input = net_input.type(dtype).to(device)
    
    if(total_vp == 1):
        loss_curve = []

        for i in range(num_iter):
            optimizer.zero_grad()

            out = net(net_input)
            
            # print(out.shape)

            loss = loss_fn(out, img_noisy_var)
            true_loss = loss_fn(out, img_clean_var)

            loss.backward() 
            optimizer.step()

            mse_wrt_noisy[i] = loss.data.cpu().numpy()
            mse_wrt_clean[i] =true_loss.data.cpu().numpy()

            if i%50 == 0:
                curr_output = net(net_input)
                curr_loss = loss_fn(curr_output, img_noisy_var)
                loss_curve.append(curr_loss.data)
    
    else:
        
        data = DatasetLoader(net_input, img_noisy_var)
        trainloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_iter):
            train_loss_noisy = 0
            for batch_id, (ni, vel) in enumerate(trainloader):
                optimizer.zero_grad()
                                
                out = net(ni)
                
                loss = loss_fn(out, img_noisy_var)
                
                train_loss_noisy += loss.item()
                
                loss.backward()
                optimizer.step()
                
                train_loss_noisy += loss.item()
                
            mse_wrt_noisy[epoch] = train_loss_noisy
            
    return mse_wrt_noisy, mse_wrt_clean, net_input_saved,  net
        
        
                
            
                
                
                
        
    
    