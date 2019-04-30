import torch
import torch.nn as nn
import torch.nn.functional as F


class conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride, dilation, causal=False):
        super().__init__()
        
    def forward(self, data):
        pass

class conv1d(nn.Module):
    pass


class deconv2d(nn.Module):
    pass


class deconv1d(nn.Module):
    pass


if __name__ == '__main__':
    import numpy as np
    # make tensor for dilated causal conv1d
    # batch_size(2) * time_len(5) * dimension(3)
    a = np.array([1,1,1])
    a = np.array([(i+1)*a for i in range(5)])
    a = np.array([a,a]).astype(np.float32)

    a = torch.from_numpy(a)
    print(a.dtype)
    print(a.size())


    # a = np.array([1,2,3,4,5,6,7,8,9,10])
    # a = np.array([a,a])
    # a = torch.randn(32,28,28,3)
    # print(a.size())
    # a = F.pad(a, )
    # print(a.size())