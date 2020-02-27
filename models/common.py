import torch
import torch.nn as nn
import time
import sys
import math
import os
from collections import OrderedDict

def print_dict(dict_):
    for k,v in dict_.items():
        print('{}: {}'.format(k, v))

#def create_wt_map(module):
#    i = 0
#    dict_ = {}
#    for k,v in module.named_modules():
#        if isinstance(v, nn.Conv2d) or isinstance(v, nn.Linear) or \
#                isinstance(v, nn.BatchNorm2d):
#            dict_[i] = v
#            i += 1
#    return dict_
#
#
#def copy_weights(dest, src):
#    src_dict = create_wt_map(src)
#    dest_dict = create_wt_map(dest)
#
#    for k,v in src_dict.items():
#        dest_dict[k].weight.data = v.weight.data.clone()
#
#        if v.bias is not None:
#            dest_dict[k].bias.data = v.bias.data.clone()
#
#        ## NOTE: how to copy the BN layer parameters
#        if isinstance(v, nn.BatchNorm2d):
#            dest_dict[k].running_mean = v.running_mean
#            dest_dict[k].running_var = v.running_var
#
#    for k,v in src_dict.items():
#        assert torch.all(dest_dict[k].weight.data == v.weight.data)
#
#    return dest

def multi_maxmin(data, axes, max_=True, keepdim=False):
    '''
    performs `torch.max` over multiple dimensions of `input`
    '''
    axes = sorted(axes)
    output = data
    for axis in reversed(axes):
        if max_ == True:
            output, _ = output.max(axis, keepdim)
        else:
            output, _ = output.min(axis, keepdim)
    return output

def mid_tensor(mid, H, W, sz, dev):
    mid1 = torch.zeros(sz, device=torch.device(dev))
    for i in range(H):
        for j in range(W):
            mid1[:,:,i,j] = mid[:,:,0,0]
    return mid1


def _find_mid_(in_, H, W, axes, dev):
    max_vals = multi_maxmin(in_, axes, 1, keepdim=True)
    min_vals = multi_maxmin(in_, axes, 0, keepdim=True)
    mid = (max_vals + min_vals) / 2
    in_cap = mid_tensor(mid, H, W, in_.size(), dev)
    delta = in_ - in_cap
    delta_mag = torch.abs(delta)
    delta_sgn = torch.where(delta >= 0, \
            torch.tensor(1.0, device=torch.device(dev)), \
            torch.tensor(-1.0, device=torch.device(dev)))
    return (in_cap, delta_mag, delta_sgn)


def find_mid(data, H, W, order, dev):
    " finds the mid of a tensor window of size HxW "
    axes = (2, 3)
    out_data = torch.zeros(data.size(), device=torch.device(dev))
    in_ = data
    all_info = []
    for o in range(order):
        info_ = _find_mid_(in_, H, W, axes, dev)
        all_info.append(info_)
        in_ = info_[1] # next delta

    out_data = all_info[0][0]
    for o in range(order-1):
        sz = all_info[o][0].size()
        signs = torch.ones(sz, device=torch.device(dev))
        for p in range(o+1):
            signs = torch.mul(signs, all_info[p][2])
            #print(signs)
        out_data += torch.mul(signs, all_info[o+1][0])
    return out_data


#def find_mid(data, H, W, order, dev):
#    " finds the mid of a tensor window of size HxW "
#    axes = (2, 3)
#    max_vals = multi_maxmin(data, axes, 1, keepdim=True)
#    min_vals = multi_maxmin(data, axes, 0, keepdim=True)
#    out_data = torch.zeros(data.size(), device=torch.device(dev))
#
#    mid = (max_vals + min_vals) / 2
#
#    mid1 = mid_tensor(mid, H, W, data.size(), dev)
#
#    if order == 1:
#        out_data = mid1
#
#    else:
#        delta1 = data - mid1
#        sign1 = torch.where(delta1 >= 0, \
#                torch.tensor(1.0, device=torch.device(dev)), \
#                torch.tensor(-1.0, device=torch.device(dev)))
#        mag1 = torch.abs(delta1)
#        max_vals = multi_maxmin(mag1, axes, 1, keepdim=True)
#        min_vals = multi_maxmin(mag1, axes, 0, keepdim=True)
#        mid = (max_vals + min_vals) / 2
#        mid2 = mid_tensor(mid, H, W, data.size(), dev)
#
#        if order == 2:
#            out_data = torch.addcmul(mid1, 1.0, sign1, mid2) #mid1 + sign1*mid2
#
#        elif order == 3:
#            delta2 = mag1 - mid2
#            sign2 = torch.where(delta2 >= 0, \
#                    torch.tensor(1.0, device=torch.device(dev)), \
#                    torch.tensor(-1.0, device=torch.device(dev)))
#            mag2 = torch.abs(delta2)
#            max_vals = multi_maxmin(mag2, axes, 1, keepdim=True)
#            min_vals = multi_maxmin(mag2, axes, 0, keepdim=True)
#            mid = (max_vals + min_vals) / 2
#            mid3 = mid_tensor(mid, H, W, data.size(), dev)
#
#            temp = torch.addcmul(mid2, 1.0, sign2, mid3)
#            out_data = torch.addcmul(mid1, 1.0, sign1, temp)
#
#    return out_data

def save_mse(mean_sqr_err, filename):
    import h5py
    hf = h5py.File(filename+'.h5', 'a')
    mean_sqr_err_np = mean_sqr_err.cpu().detach().numpy()
    #print(mean_sqr_err_np.shape)
    if '/MSE' not in hf:
        hf.create_dataset('MSE', data=mean_sqr_err_np, chunks=True, maxshape=(None, 2048, 200, 200))
    else:
        hf["MSE"].resize((hf["MSE"].shape[0] + mean_sqr_err_np.shape[0]), axis = 0)
        hf["MSE"][-mean_sqr_err_np.shape[0]:] = mean_sqr_err_np
    hf.close()


def find_mean(data, H, W, dev):
    axes = (2,3)
    out_data = torch.zeros(data.size(), device=torch.device(dev))
    for i in range(H):
        for j in range(W):
            out_data[:,:,i,j] = data.mean(axes)
    return out_data

#def approximate(w_sz, data, order, dev):
def approximate(w_sz, data, order, dev, out_dir, filename):
    n, c, h, w = data.size() # input feature map

    stride = w_sz
    xy_sz_lb = math.floor(((h - w_sz) / stride) + 1)
    xy_sz = math.ceil(((h - w_sz) / stride) + 1)

    output = torch.zeros((n,c,h,w), device=torch.device(dev))

    mean_sqr_err = torch.zeros((n,c,xy_sz, xy_sz), device=torch.device(dev))
    for i in range(xy_sz):
        h_up = min((i+1)*w_sz, h)
        if i == xy_sz-1:
            H = xy_sz - xy_sz_lb
        else:
            H = w_sz

        for j in range(xy_sz):
            if j == xy_sz-1:
                W = xy_sz - xy_sz_lb
            else:
                W = w_sz

            w_up = min((j+1)*w_sz, w)

            #### NOTE: snippet for data collection!!!
            in_patch = data[:, :, i*w_sz:h_up, j*w_sz:w_up]
            mean_patch = find_mean(in_patch, H, W, dev)
            sqr_err = torch.pow(in_patch-mean_patch, 2)
            mse_patch = sqr_err.mean((2,3))
            mean_sqr_err[:,:,i,j] = mse_patch

    filenm = os.path.join(out_dir, filename)
    save_mse(mean_sqr_err, filenm)
            ######## End of snippet #######

            #output[:, :, i*w_sz:h_up, j*w_sz:w_up] = find_mid(data[:, :, i*w_sz:h_up, j*w_sz:w_up], H, W, order, dev)
    output = data

    return output

class Adder(nn.Module):
    def __init__(self):
        super(Adder, self).__init__()

    def forward(self, x, y):
        return x + y


class approximator(nn.Module):
    def __init__(self, device, w_sz=1, order=1, approx=False):
        super(approximator, self).__init__()
        self.dev = device
        self.w_sz = w_sz
        self.order = order
        self.approx = approx
        self.name = ''
        self.out_dir = ''

    def forward(self, x):
        if self.approx:
            #x = approximate(self.w_sz, x, self.order, self.dev)
            x = approximate(self.w_sz, x, self.order, self.dev, self.out_dir, self.name)
        return x

    def get_info(self):
        return (self.w_sz, self.order, self.self.approx)

    def extra_repr(self):
        return 'w_sz={}, order={}, approx={}'.format(self.w_sz, self.order, self.approx)


layer_types = [nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.BatchNorm2d, \
        nn.ReLU6, nn.ReLU, nn.AdaptiveAvgPool2d, nn.Dropout, approximator]

type_to_name = {nn.Conv2d: 'conv', nn.Linear: 'linear', nn.MaxPool2d: 'maxpool', \
        nn.BatchNorm2d: 'batchnorm', nn.ReLU6: 'relu6', nn.ReLU: 'relu', \
        nn.AdaptiveAvgPool2d: 'avg', nn.Dropout: 'dropout', approximator: 'approx', \
        Adder: 'adder'}

def serialize_model(net, layer_types):
    od = OrderedDict()
    i = 0
    for k,v in net.named_modules():
        for l in layer_types:
            if isinstance(v, l):
                od[i] = (v, type_to_name[l])
                i += 1
                break
    return od

class Hook():
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

def ofm_size(*args):
    " size of conv output in kB "
    sz = 1
    for a in args:
        sz *= a
    sz *= 4 # assuming 32-bit floats
    sz /= 1024 # in KBs
    return sz


def compute_memory(net, img_size, dev):

    net.to(dev)
    layer_map = serialize_model(net, layer_types)
    hooks = [Hook(l[0]) for l in layer_map.values()]
    output = net(torch.ones(img_size, device=torch.device(dev)))
    #output = net(torch.ones(img_size))

    ftr_size = []
    for i in range(len(layer_map.keys())-1):
        if not isinstance(layer_map[i][0], approximator):
            if isinstance(layer_map[i+1][0], approximator):
                module = layer_map[i+1][0]
                w_sz, order, approx = module.w_sz, module.order, module.approx
                #print(w_sz, order, approx)

                if approx:
                    if len(hooks[i+1].output.shape) > 2:
                        _,c,h,w = hooks[i+1].output.shape
                        mid_c, mid_h, mid_w = c, math.ceil(h/w_sz), math.ceil(w/w_sz)
                        mid_sz = order * mid_c * mid_h * mid_w * 4  # assuming 32-bit floats

                        sign_c, sign_h, sign_w = c, h, w
                        sign_sz = ((order-1) * (sign_c * sign_h * sign_w)) / 8 # 1-bit for sign
                        sz = mid_sz + sign_sz
                        sz /= 1024 # in KBs

                        ftr_size.append(((mid_c, mid_h, mid_w), sz, layer_map[i][1]))
                    else:
                        _, w = hooks[i+1].output.shape
                        _, mid_w = _, math.ceil(w/w_sz)
                        mid_sz = order * mid_w * 4  # assuming 32-bit floats

                        sign_w = w
                        sign_sz = ((order-1) * sign_w) / 8 # 1-bit for sign

                        sz = mid_sz + sign_sz
                        sz /= 1024 # in KBs

                        ftr_size.append(((mid_w), sz, layer_map[i][1]))

                else:
                    if len(hooks[i].output.shape) > 2:
                        _,c,h,w = hooks[i].output.shape
                        #sz = c * h * w * 4 # assuming 32-bit floats
                        sz = ofm_size(c,h,w)
                        ftr_size.append(((c,h,w), sz, layer_map[i][1]))
                    else:
                        _,w = hooks[i].output.shape
                        sz = ofm_size(w)
                        ftr_size.append(((w), sz, layer_map[i][1]))

            else:
                if len(hooks[i].output.shape) > 2:
                    _,c,h,w = hooks[i].output.shape
                    #sz = c * h * w * 4 # assuming 32-bit floats
                    #sz /= 1024 # in KBs
                    sz = ofm_size(c,h,w)
                    ftr_size.append(((c,h,w), sz, layer_map[i][1]))
                else:
                    _,w = hooks[i].output.shape
                    #sz = w * 4 # assuming 32-bit floats
                    #sz /= 1024 # in KBs
                    sz = ofm_size(w)
                    ftr_size.append(((w), sz, layer_map[i][1]))

    #NOTE: disregarding classifier from calculations

    net.to('cpu')
    return ftr_size
