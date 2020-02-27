import os
import numpy as np
import time
import sys
import math

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
from configobj import ConfigObj

from models import mobv2_approx, vgg11_approx
from models.common import serialize_model, print_dict
from models.common import Adder, approximator

def plot_act_mse(mse, filename):
    mse = mse.flatten()
    mse = np.sort(mse)
    plt.figure()
    #plt.plot(mse)
    num_bins = 10
    max_mse = mse[mse.shape[0]-1] # max_val
    max_perc = np.percentile(mse, 99) # 90th percentile
    bins = num_bins * math.ceil( max_mse / num_bins)
    print(bins)
    #plt.hist(mse, bins=bins)#, normed=1)
    plt.hist(mse, bins=bins, density=True)
    plt.xlim(0,max_perc+0.5)
    plt.savefig(filename)

def load_model(net, model_path, file_name):
    # Load checkpoint.
    file_path = os.path.join(model_path, file_name)
    if not os.path.exists(file_path):
        print('file {} not found! Exiting..'.format(file_path))
        exit()

    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print (best_acc, start_epoch)
    return checkpoint, net

def create_wt_map(module):
    i = 0
    dict_ = {}
    for k,v in module.named_modules():
        if isinstance(v, nn.Conv2d) or isinstance(v, nn.Linear) or \
                isinstance(v, nn.BatchNorm2d):
            dict_[i] = v
            i += 1
    return dict_


def copy_weights(dest, src):
    src_dict = create_wt_map(src)
    dest_dict = create_wt_map(dest)

    for k,v in src_dict.items():
        dest_dict[k].weight.data = v.weight.data.clone()

        if v.bias is not None:
            dest_dict[k].bias.data = v.bias.data.clone()

        ## NOTE: how to copy the BN layer parameters
        if isinstance(v, nn.BatchNorm2d):
            dest_dict[k].running_mean = v.running_mean
            dest_dict[k].running_var = v.running_var

    for k,v in src_dict.items():
        assert torch.all(dest_dict[k].weight.data == v.weight.data)

    return dest


name_to_type = {'conv': nn.Conv2d, 'linear': nn.Linear, 'maxpool': nn.MaxPool2d , \
        'batchnorm': nn.BatchNorm2d, 'relu6': nn.ReLU6, 'relu': nn.ReLU, \
        'avg': nn.AdaptiveAvgPool2d, 'dropout': nn.Dropout, 'approx': approximator, \
        'adder': Adder}

def apply_uniform(net, u_app_config):
    layer_types = [nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.BatchNorm2d, \
        nn.ReLU6, nn.ReLU, nn.AdaptiveAvgPool2d, nn.Dropout, Adder, \
        approximator]

    layer_map = serialize_model(net, layer_types)
    approx_layer_type = u_app_config['layer_type']
    pre_approx_layer = None
    for k,v in layer_map.items():
        if approx_layer_type in v[1]:
            pre_approx_layer = v[0]
            break

    approx_layers = []
    for i in range(len(list(layer_map.keys()))):
        type_ = name_to_type[layer_map[i][1]]
        #print(pre_approx_layer, type_)
        if isinstance(pre_approx_layer, type_):
            approx_layer = layer_map[i+1][0]

            approx_layer.w_sz = u_app_config.getint('window')
            approx_layer.order = u_app_config.getint('order')
            approx_layer.approx = True

#def apply_approx_config(net, approx_config):
def apply_approx_config(net, approx_config, out_dir):
    layer_types = [nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.BatchNorm2d, \
        nn.ReLU6, nn.ReLU, nn.AdaptiveAvgPool2d, nn.Dropout, Adder, \
        approximator]
    #print_dict(serialize_model(net, layer_types))
    approx_layers = []
    for name, m in net.named_modules():
        if isinstance(m, approximator):
            approx_layers.append(m)

    #assert(len(approx_layers) == len(approx_config.items())-1)
    print('len(approx_layers): {}, len(approx_config.items(): {}'.format(len(approx_layers), len(approx_config)))
    for i, (k,v) in enumerate(approx_config.items()):
        approx_layer = approx_layers[i]

        #print(i, v.as_int('window'), v.as_int('order'), v.as_bool('approx'))
        approx_layer.w_sz = v.as_int('window')
        approx_layer.order = v.as_int('order')
        approx_layer.approx = v.as_bool('approx')
        approx_layer.name = k # save the layer name
        approx_layer.out_dir = out_dir

        if i == len(approx_layers)-1:
            break


#def create_approximate(net, arch, device, approx_config ):
def create_approximate(net, arch, device, approx_config, out_dir):
    model = None
    if arch == 'mob_v2':
        model = mobv2_approx.MobileNetV2_approx(device, width_mult=1)
    model = copy_weights(model, net)
    apply_approx_config(model, approx_config, out_dir)
    #apply_approx_config(model, approx_config)

    return model


#def multi_maxmin(data, axes, max_=True, keepdim=False):
#    '''
#    performs `torch.max` over multiple dimensions of `input`
#    '''
#    axes = sorted(axes)
#    output = data
#    for axis in reversed(axes):
#        if max_ == True:
#            output, _ = output.max(axis, keepdim)
#        else:
#            output, _ = output.min(axis, keepdim)
#    return output
#
#def mid_tensor(mid, H, W, sz, dev):
#    mid1 = torch.zeros(sz, device=torch.device(dev))
#    for i in range(H):
#        for j in range(W):
#            #mid = (max_vals + min_vals) / 2
#            #out_data[:,:,i,j] = mid[:,:,0,0]
#            mid1[:,:,i,j] = mid[:,:,0,0]
#    return mid1
#
#
#def find_mid(data, H, W, order, dev):
#    axes = (2, 3)
#    max_vals = multi_maxmin(data, axes, 1, keepdim=True)
#    min_vals = multi_maxmin(data, axes, 0, keepdim=True)
#    out_data = torch.zeros(data.size(), device=torch.device(dev))
#
#    mid = (max_vals + min_vals) / 2
#
#    #mid1 = torch.zeros(data.size(), device=torch.device(dev))
#    #delta1 = torch.zeros(data.size(), device=torch.device(dev))
#    #sign1 = torch.zeros(data.size(), device=torch.device(dev))
#    #mag1 = torch.zeros(data.size(), device=torch.device(dev))
#
#    mid = (max_vals + min_vals) / 2
#    mid1 = mid_tensor(mid, H, W, data.size(), dev)
#    #for i in range(H):
#    #    for j in range(W):
#    #        mid = (max_vals + min_vals) / 2
#    #        #out_data[:,:,i,j] = mid[:,:,0,0]
#    #        mid1[:,:,i,j] = mid[:,:,0,0]
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
#            out_data = torch.addcmul(mid1, 1.0, sign1, mid2) #mid1 + sign1
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
#
#
#def find_mean(data, H, W, dev):
#    axes = (2,3)
#    out_data = torch.zeros(data.size(), device=torch.device(dev))
#    for i in range(H):
#        for j in range(W):
#            out_data[:,:,i,j] = data.mean(axes)
#    return out_data
#
##def approximate(w_sz, data, dev):
#def approximate(w_sz, data, order, dev):
#    n, c, h, w = data.size() # input feature map
#
#    stride = w_sz
#    xy_sz_lb = math.floor(((h - w_sz) / stride) + 1)
#    xy_sz = math.ceil(((h - w_sz) / stride) + 1)
#
#    output = torch.zeros((n,c,h,w), device=torch.device(dev))
#
#    for i in range(xy_sz):
#        h_up = min((i+1)*w_sz, h)
#        if i == xy_sz-1:
#            H = xy_sz - xy_sz_lb
#        else:
#            H = w_sz
#
#        for j in range(xy_sz):
#            if j == xy_sz-1:
#                W = xy_sz - xy_sz_lb
#            else:
#                W = w_sz
#
#            w_up = min((j+1)*w_sz, w)
#            output[:, :, i*w_sz:h_up, j*w_sz:w_up] = find_mid(data[:, :, i*w_sz:h_up, j*w_sz:w_up], H, W, order, dev)
#            #output[:, :, i*w_sz:h_up, j*w_sz:w_up] = find_mid(data[:, :, i*w_sz:h_up, j*w_sz:w_up], H, W, dev)
#            #output[:, :, i*w_sz:h_up, j*w_sz:w_up] = find_mean(data[:, :, i*w_sz:h_up, j*w_sz:w_up], H, W, dev)
#
#    return output
#
#
#class approximator(nn.Module):
#    #def __init__(self, w_sz, device):
#    def __init__(self, w_sz, order, device):
#        super(approximator, self).__init__()
#        self.w_sz = w_sz
#        self.dev = device
#        self.order = order
#
#    def forward(self, x):
#        x = approximate(self.w_sz, x, self.order, self.dev)
#        #x = approximate(self.w_sz, x, self.dev)
#        return x
#
#    def extra_repr(self):
#        return 'w_sz={}, order={}'.format(self.w_sz, self.order)

#term_width = '143' # note: edit by sms821 for nohup to work
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    #print('current: {}, total: {}'.format(current, total))
    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



def validate(net, testloader, device='cuda:0'):
     net.eval()
     net.to(device)
     test_loss = 0
     correct = 0
     total = 0
     display_imgs = False
     criterion = nn.CrossEntropyLoss()
     with torch.no_grad():
         for batch_idx, (inputs, targets) in enumerate(testloader):
             #print(inputs.size())
             if display_imgs:
                 inp = inputs[0].numpy().transpose((1, 2, 0))
                 inp = np.clip(inp, 0, 1)
                 plt.imshow(inp)
                 plt.show()

             inputs, targets = inputs.to(device), targets.to(device)
             outputs = net(inputs)
             loss = criterion(outputs, targets)

             test_loss += loss.item()
             _, predicted = outputs.max(1)
             total += targets.size(0)
             correct += predicted.eq(targets).sum().item()
             acc = 100.*correct/total


             #print(batch_idx)
             print(' %d/%d  Loss: %.3f | Acc: %.3f%% (%d/%d)' \
                     % (batch_idx+1, len(testloader), test_loss/(batch_idx+1), acc, correct, total))
             #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
             #    % (test_loss/(batch_idx+1), acc, correct, total))
             if batch_idx > 101:
                break # NOTE!!!
     return acc



def load_imagenet(data_dir, batch_size=128, shuffle=True):
    """
    Load the Tiny ImageNet dataset.
    """
    #print('loading tiny imagenet')
    train_dir = os.path.join(data_dir, 'Train_Data')
    test_dir = os.path.join(data_dir, 'Validation_Data')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]) # using mean and std of original imagenet dataset

    #print('reading data..')

    train_transform = transforms.Compose([
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    train_loader = None


    test_transform = transforms.Compose([
        transforms.Resize(256), # this line is imp for pre-trained imagenet models to yield reported acc.
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = datasets.ImageFolder(test_dir, test_transform)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return (train_loader, val_loader)


#class Hook():
#    def __init__(self, module, backward=False):
#        if backward == False:
#            self.hook = module.register_forward_hook(self.hook_fn)
#        else:
#            self.hook = module.register_backward_hook(self.hook_fn)
#
#    def hook_fn(self, module, input, output):
#        self.input = input
#        self.output = output
#
#    def close(self):
#        self.hook.remove()

def total_size(sizes):
    sum_ = 0
    for a,b,c in sizes:
        sum_ += b
    return sum_

layer_types = [nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.BatchNorm2d, \
        nn.ReLU6, nn.ReLU, nn.AdaptiveAvgPool2d, nn.Dropout, Adder]

#def create_approx_config(net, out_dir):
def create_approx_config(net, out_dir, u_app_config):
    layer_map = serialize_model(net, layer_types)
    #print_dict(dict_)
    config = ConfigObj()
    config.filename = os.path.join(out_dir, 'approx_config.ini')

    u_layer_type, u_approx, u_order, u_window = None, None, None, None
    if u_app_config.getboolean('apply_uniform'):
        u_approx = True
        u_order = u_app_config.getint('order')
        u_window = u_app_config.getint('window')
        u_layer_type = u_app_config['layer_type']

    for k,v in layer_map.items():
        name = v[1] + '_' + str(k)
        config[name] = {}
        if u_layer_type is not None and u_layer_type in v[1]:
            config[name]['approx'] = True
            config[name]['order'] = u_order
            config[name]['window'] = u_window
        else:
            config[name]['approx'] = False
            config[name]['order'] = 0
            config[name]['window'] = 0

    config.write()
























