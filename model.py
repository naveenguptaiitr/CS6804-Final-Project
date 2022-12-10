import torch
import torch.nn.functional as F
import torch.nn as nn
from math import ceil


def conv_block(input_channels, output_channels, kernel_size=3, stride=1, pad=1):
    
    return nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad),
                         nn.BatchNorm2d(output_channels),
                         nn.LeakyReLU(0.2,inplace=True))

def conv_block_last(input_channels, output_channels, kernel_size=3, stride=1, pad=1):
    
    return nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad),
                         nn.BatchNorm2d(output_channels),
                         nn.ReLU(0.2))


def convtranspose_block(input_channels, output_channels, kernel_size=2, stride=2, pad=0, out=False):
    if out == False:
        return nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels,
                                                kernel_size=kernel_size, stride=stride, padding=pad),
                             nn.BatchNorm2d(output_channels),
                             nn.Tanh())
    else:
        return nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels, 
                                                kernel_size, stride, pad),
                             nn.BatchNorm2d(output_channels),
                             nn.LeakyReLU(0.2, inplace=True))

def maxpool_block(kernel_size, stride):
    return nn.Sequential(nn.MaxPool2d(kernel_size, stride))

def resize_conv_with_bn(input_channels, output_channels, scale_factor=2, mode='nearest'):
    
    return nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode=mode),
                         nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2,
                                   padding=1),
                         nn.BatchNorm2d(num_features=output_channels),
                         nn.LeakyReLU(0.2, inplace=True))


class InversionNetB(nn.Module):
    
    def __init__(self, input_channel, encoder_channels, decoder_channels, sample_spatial=1.0):
        
        super(InversionNetB, self).__init__()
        
        self.encoder_channels = [input_channel, *encoder_channels]
        self.decoder_channels = [encoder_channels[-1], *decoder_channels]
        
        encoder_layers = []
        decoder_layers = []
        
        for i in range(len(self.encoder_channels)-1):
            if i == 0:
                encoder_layers.append(conv_block(self.encoder_channels[i], self.encoder_channels[i+1], 
                                                 kernel_size=(7,1), stride=(2,1), pad=(3,0)))
            elif i<7:
                if(i%2!=0):
                    encoder_layers.append(conv_block(self.encoder_channels[i], self.encoder_channels[i+1], 
                                                 kernel_size=(3,1), stride=(2,1), pad=(1,0)))
                else:
                    encoder_layers.append(conv_block(self.encoder_channels[i], self.encoder_channels[i+1], 
                                                 kernel_size=(3,1), pad=(1,0)))
                    
            elif i<len(self.encoder_channels)-2:
                if(i%2!=0):
                    encoder_layers.append(conv_block(self.encoder_channels[i], self.encoder_channels[i+1],stride=2))
                else:
                    encoder_layers.append(conv_block(self.encoder_channels[i], self.encoder_channels[i+1]))
            else:
                encoder_layers.append(conv_block(self.encoder_channels[i], self.encoder_channels[i+1],
                                                 kernel_size=(8, ceil(70 * sample_spatial / 8)), pad=0))
        
        for i in range(len(self.decoder_channels)-1):
            
            if i == 0:
                decoder_layers.append(convtranspose_block(self.decoder_channels[i], self.decoder_channels[i+1],
                                                         kernel_size=5, out=False))
                    
                decoder_layers.append(conv_block(self.decoder_channels[i+1], self.decoder_channels[i+1]))
            
            else:
                decoder_layers.append(convtranspose_block(self.decoder_channels[i], self.decoder_channels[i+1],
                                                         kernel_size=4, stride=2, pad=1, out=False))
                                     
                decoder_layers.append(conv_block(self.decoder_channels[i+1], self.decoder_channels[i+1]))
        
        
        self.conv_block_last = conv_block_last(self.decoder_channels[-1], 1)
                                     
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.decoder_layers = nn.Sequential(*decoder_layers)
                
    
    def forward(self, x):
        x = self.encoder_layers(x)
        x = self.decoder_layers(x)
        
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)
        x = self.conv_block_last(x)                       
        return x
    
                
        
        
            
        
        
        
        
        
        
        
        
        