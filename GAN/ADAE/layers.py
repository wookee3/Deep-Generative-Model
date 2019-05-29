import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _single():
    pass


def _init_weight(m):
    class_name = m.__class__
    if "Linear" in class_name:
        pass
    elif "Conv" in class_name:
        pass
    else:
        pass


# conv layer with SAME padding in tensorflow
# TODO: padding == 'valid
class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=1, dilation=1, padding='SAME'):
        super(Conv2d, self).__init__()

        if padding.lower() == 'same':
            if isinstance(k_size, int):
                pad_len = int((k_size - 1) * dilation / 2)
                pad = nn.ConstantPad2d(pad_len, 0)
            else:
                pad_len_h = int((k_size[0] - 1) * dilation / 2)
                pad_len_w = int((k_size[1] - 1) * dilation / 2)
                pad = nn.ConstantPad2d((pad_len_h, pad_len_h, pad_len_w, pad_len_w), 0)
        else:
            pass
        conv = nn.Conv2d(in_ch, out_ch, k_size, stride, dilation=dilation)
        self.sequence = nn.Sequential(pad, conv)

    def forward(self, data):
        return self.sequence(data)


class Conv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=1, dilation=1, padding='SAME', causal=False):
        super(Conv1d, self).__init__()
        if padding.lower() == 'same':
            if causal:
                pad_len = (k_size - 1) * dilation
                pad = nn.ConstantPad1d((pad_len, 0), 0)
            else:
                pad_len = int((k_size-1)*dilation/2)
                pad = nn.ConstantPad1d(pad_len, 0)
        else:
            pass    
        conv = nn.Conv1d(in_ch, out_ch, k_size, stride, dilation=dilation)
        self.sequence = nn.Sequential(pad, conv)

    def forward(self, data):
        return self.sequence(data)
        

# TODO: causal deconv layers
class Deconv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=1, dilation=1, padding='SAME', out_padding=0):
        super(Deconv2d, self).__init__()
        assert type(k_size) == type(stride), 'k_size and stride should have same size'
       
        if padding.lower() == 'same':
            if isinstance(k_size, int):
                pad_len = int(((k_size - 1) * dilation + 1 - stride)/2)
            else:
                pad_len_h = int(((k_size[0] - 1) * dilation + 1 - stride[0])/2)
                pad_len_w = int(((k_size[1] - 1) * dilation + 1 - stride[1]) / 2)
                pad_len = (pad_len_h, pad_len_w)
        else:
            pass
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, k_size, stride, pad_len, dilation=dilation)
        
    def forward(self, data):
        return self.deconv(data)
        

class Deconv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=1, dilation=1, padding='SAME', out_padding=0):
        super(Deconv1d, self).__init__()
        
    def forward(self, data):
        pass


if __name__ == '__main__':
    import numpy as np
    # make tensor for dilated causal conv1d
    # batch_size(2) * time_len(5) * dimension(3)
    # a = np.array([1,1,1])
    # a = np.array([(i+1)*a for i in range(5)])
    # a = np.array([a,a]).astype(np.float32)

    # a = torch.from_numpy(a)
    # print(a.dtype)
    # print(a.size())

    # conv layer
    conv2d = Conv2d(in_ch=1, out_ch=3, k_size=3, stride=1, dilation=2)
    conv1d = Conv1d(1, 3, 3, 1, 1, causal=False)
    def_conv2d = nn.Conv2d(1, 3, 3, 1, 2)
    
    # input
    input2d = torch.randn(4, 1, 4, 4)
    input1d = torch.randn(4, 1, 28)
    print("input")
    print(input2d.size())
    print(input1d.size())

    mid2d = conv2d(input2d)
    mid1d = conv1d(input1d)
    print("mid")
    print(mid2d.size())
    print(mid1d.size())
    
    # deconv layer
    deconv2d = Deconv2d(3, 1, 4, 2, 1)
    # deconv1d = Deconv1d(3, 1, 3, 1, 1, causal=False)
    print("output")
    print(deconv2d(mid2d).size())
    # print(deconv1d.size())

    # in_ch, out_ch, k_size, stride=1, dilation=1