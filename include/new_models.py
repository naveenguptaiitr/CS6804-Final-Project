import torch
import torch.nn as nn

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module


def conv(in_f, out_f, kernel_size, stride=1, pad='zero'):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=False)

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)


def new_models(
        num_output_channels=3, 
        num_channels_up=[128]*5,
        layers=5,
        filter_size_up=1,
        need_sigmoid=True, 
        pad ='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(),
        bn_before_act = False,
        bn_affine = True,
        upsample_first = True,
        ):
    
    num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
    n_scales = len(num_channels_up) 
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales
    
    
    model = nn.Sequential()
    
    curr_layer = 0
    for i in range(len(num_channels_up)-1):
        
        if curr_layer<layers:
            
            if upsample_first:
                model.add(conv( num_channels_up[i], num_channels_up[i+1],  filter_size_up[i], 1, pad=pad))
                if upsample_mode!='none' and i != len(num_channels_up)-2:

                    model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
                    model.add(conv(num_channels_up[i+1], num_channels_up[i+1],1, pad=pad))

                    #model.add(nn.Conv2d(num_channels_up[i+1], num_channels_up[i+1], kernel_size=3, padding=1))

                    # model.add(nn.PixelShuffle(upscale_factor=2)) 
                    # model.add(conv(num_channels_up[i+1]//4, num_channels_up[i+1],5, pad=pad))
                    # model.add(nn.Conv2d(num_channels_up[i+1]//4, num_channels_up[i+1], kernel_size=3, padding=1))
            else:
                if upsample_mode!='none' and i!=0:

                    model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
                    model.add(conv(num_channels_up[i], num_channels_up[i+1],1, pad=pad))

                model.add(conv( num_channels_up[i], num_channels_up[i+1],  filter_size_up[i], 1, pad=pad))
        else:
            model.add(nn.Conv2d(num_channels_up[i+1], num_channels_up[i+1], kernel_size=3, padding=1))
            model.add(act_fun)
            
        curr_layer += 1
        
        if i != len(num_channels_up)-1:	
            if(bn_before_act): 
                model.add(nn.BatchNorm2d( num_channels_up[i+1] ,affine=bn_affine))
            model.add(act_fun)
            if(not bn_before_act): 
                model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))
      
    model.add(conv(num_channels_up[-1], num_output_channels, 1, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    
    return model




