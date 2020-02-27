import math
import numpy as np
import csv
import os
import argparse
import configparser
import pprint
import ast
from configobj import ConfigObj

import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

# Taken from https://github.com/tonylins/pytorch-mobilenet-v2
from models.MobileNetV2 import mobilenet_v2
from models.common import compute_memory
from utils import *


def main():
    parser = argparse.ArgumentParser(description='Feature Map Codecs Memory Requirements')
    parser.add_argument('--config-file', default='config_mobv2.ini')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)
    pprint.pprint({section: dict(config[section]) for section in config.sections()})

    defaults = config['DEFAULT']
    device = defaults['device']
    app_name = defaults['app_name']

    print('[INFO] Using {}'.format(app_name))
    if not os.path.isdir(app_name):
        os.mkdir(app_name)
    if not os.path.isdir(os.path.join(app_name, 'logs')):
        os.mkdir(os.path.join(app_name, 'logs'))
    out_dir = app_name
    log_dir = os.path.join(out_dir, 'logs')

    net_config = config['model']
    arch = net_config['arch']
    model_path = net_config['model_path']
    model_file = net_config['file_name']
    classes = net_config['num_classes']

    " Load the original model "
    net = None
    if 'mob_v2' in arch:
        net = mobilenet_v2(pretrained=True) # loads the weights as well

    print(net)
    #state, net = load_model(net, model_path, model_file)
    # add this case-wise

    " Load the dataset "
    data_config = config['dataset']
    data_dir = data_config['data_dir']
    batch_size = data_config.getint('batch_size')
    img_size = ast.literal_eval(data_config['img_size'])
    if data_config['dataset'] == 'imagenet':
        _, testloader = load_imagenet(data_dir, batch_size)

    " Tasks to do "
    tasks = config['functions']
    if tasks.getboolean('validate'): # 1
        validate(net, testloader, device)

    if tasks.getboolean('create_approx_config'): # 2
        # create the approximator config file
        uniform_approx_config = config['approximator']
        create_approx_config(net, out_dir, uniform_approx_config)

    if tasks.getboolean('approximate'): # 3

        # read the approximator config file, create approximated model and validate it
        approx_file = os.path.join(out_dir, 'approx_config.ini')
        approx_config = ConfigObj(approx_file)

        new_net = None
        if 'mob_v2' in arch:
            #new_net = create_approximate(net, arch, device, approx_config)
            new_net = create_approximate(net, arch, device, approx_config, out_dir)
            print(new_net)

        if tasks.getboolean('compute_act_memory'): # 4
            org_size = compute_memory(net, img_size, device)
            new_size = compute_memory(new_net, img_size, device)

            org_tot_size = total_size(org_size)
            new_tot_size = total_size(new_size)

            assert(len(org_size) == len(new_size))
            combined = [(a,b) for a,b in zip(org_size, new_size)]
            for c in combined:
                print(c)

            print('Original Total Size: {} MB'.format(org_tot_size / 1024))
            print('New Total Size: {} MB'.format(new_tot_size / 1024))
            data_saved = (1 - (new_tot_size / org_tot_size))
            print('Data saved: {}'.format(data_saved))

        validate(new_net, testloader, device)

    if tasks.getboolean('plot_act_mse'):
        #filenames = ['conv_0']#, 'batchnorm_1', 'relu6_2', 'conv_3', 'batchnorm_4', 'relu6_5']
        # read the approximator config file
        approx_file = os.path.join(out_dir, 'approx_config.ini')
        approx_config = ConfigObj(approx_file)

        filenames = []
        for k,v in approx_config.items():
            if v.as_bool('approx'):
                filenames.append(os.path.join(out_dir, k))

        import h5py
        for filename in filenames:
            nm = filename + '.h5'
            hf = h5py.File(nm, 'r')
            mse = hf.get('MSE')
            mse = np.array(mse)
            print(mse.shape)
            #np.savetxt(filename+'.csv', mse.flatten())
            plot_act_mse(mse, filename + '.png')

            ############### DEBUGGING ##############
            #new_modules = []
            ## creating forward hook on approximator
            #i = 0
            #for n,v in new_net.named_modules():
            #    if isinstance(v, nn.ReLU6):
            #        new_modules.append(v)
            #        break
            #    i += 1

            #j = 0
            #for n,v in new_net.named_modules():
            #    if j == i+1 and isinstance(v, approximator):
            #    #if isinstance(v, approximator):
            #        new_modules.append(v)
            #        break
            #    j += 1

            ##print(len(new_modules))
            #from models.common import Hook
            #hooks = [Hook(l) for l in new_modules]

            #_, testloader = load_imagenet(data_dir, batch_size)
            #validate(new_net, testloader, device)

            #out = []
            #for h in hooks:
            #    out.append(h.output.data)

            #print('original')
            ##print(out[0][0,0,0:10, 0:10])
            #print(out[0][0,0,50:60, 50:60])
            ##print(out[0][0,0,100:150, 100:150])
            ##print(out[0][0,0,150:200, 150:200])
            ##print(out[0][126,32,200:224, 200:224])
            #print('\napproximated')
            ##print(out[1][0,0,0:10, 0:10])
            #print(out[1][0,0,50:60, 50:60])
            ##print(out[1][0,0,100:150, 100:150])
            ##print(out[1][0,0,150:200, 150:200])
            ##print(out[1][126,32,200:224, 200:224])




#device = 'cuda:0'
#data_dir = '/i3c/hpcl/avs6194/ILSVRC_DATA'
#batch_size = 128
#w_sz = 3
#order = 3
#inputs = inputs.to(device)
#
##net = models.resnet18(pretrained=True)
##net = models.resnet101(pretrained=True)
##net = models.densenet169(pretrained=True)
##net = models.vgg19(pretrained=True)
#net = models.vgg11(pretrained=True)
## Taken from https://github.com/tonylins/pytorch-mobilenet-v2
##from MobileNetV2 import mobilenet_v2
##net = mobilenet_v2(pretrained=True)
#
##print(net)
##exit()
##_, testloader = load_imagenet(data_dir, batch_size)
##summary(net, input_size=(3,256,256), device='cpu')
##img_size = (1,3,224,224)
##net = net.to(device)
##org_size = compute_memory(net, img_size, device)
##for o in org_size:
##    print(o)
##org_tot_size = total_size(org_size)
##print('Total Size: {} MB'.format(org_tot_size / 1024))
##validate(net, testloader, device)
##exit()
#
#new_net = vgg11_approx.create_approximate(net, w_sz, order, device)
##layer_num = 2
##new_net = mobv2_approx.create_approximate(net, w_sz, device, layer_num, order, approx=True)
#
##new_net = MyNet()
#new_net = new_net.to(device)
#print(new_net)
##new_net = new_net.to(device)
#
##validate(new_net, testloader, device)
##exit()
#
##img_size = (1,3,256,256)
#org_size = compute_memory(net, img_size, device)
#new_size = compute_memory(new_net, img_size, device)
#assert(len(org_size) == len(new_size))
#combined = [(a,b) for a,b in zip(org_size, new_size)]
#for c in combined:
#    print(c)
#org_tot_size = total_size(org_size)
#new_tot_size = total_size(new_size)
#print('New Total Size: {} MB'.format(new_tot_size / 1024))
##validate(net, testloader, device)
#data_saved = (1 - (new_tot_size / org_tot_size))
#print('Data saved: {}'.format(data_saved))
#exit()

#print(org_size)
#print(new_size)

#new_modules = []
## creating forward hook on approximator
#i = 0
#for n,v in new_net.named_modules():
#    if isinstance(v, nn.Conv2d):
#        new_modules.append(v)
#        break
#
#for n,v in new_net.named_modules():
#    if isinstance(v, approximator):
#        new_modules.append(v)
#        break
#
##print(len(new_modules))
#hooks = [Hook(l) for l in new_modules]
#
#_, testloader = load_imagenet(data_dir, batch_size)
##validate(new_net, testloader, device)

'''
out = []
for h in hooks:
    out.append(h.output.data)

print('original')
#print(out[0][0,0,0:10, 0:10])
print(out[0][0,0,50:60, 50:60])
#print(out[0][0,0,100:150, 100:150])
#print(out[0][0,0,150:200, 150:200])
#print(out[0][126,32,200:224, 200:224])
print('\napproximated')
#print(out[1][0,0,0:10, 0:10])
print(out[1][0,0,50:60, 50:60])
#print(out[1][0,0,100:150, 100:150])
#print(out[1][0,0,150:200, 150:200])
#print(out[1][126,32,200:224, 200:224])
'''

def dump_data(np_file, num_layers):
    summary(net, input_size=(3,224,224), device='cpu')
    net = net.to(device)

    summary(net, input_size=(3,224,224), device=device[:-2])
    modules = []
    for k,v in net.named_modules():
        if isinstance(v, nn.Conv2d) or \
           isinstance(v, nn.Linear) or \
           isinstance(v, nn.MaxPool2d) or \
           isinstance(v, nn.BatchNorm2d) or \
           isinstance(v, nn.ReLU) or \
           isinstance(v, nn.AdaptiveAvgPool2d):
               modules.append(v)
    hooks = [Hook(l) for l in modules]

    _, testloader = load_imagenet(data_dir, batch_size)

    with torch.no_grad():
        for n, data in enumerate(testloader):
            images, targets = data
            images = images.to(device)
            outputs = net(images.float())
            break # inferencing only one batch

    #print('Feature map size: ')
    ftr_maps = []
    n = 0
    for h in hooks:
        if n >= num_layers:
            break

        #print(h.output.size())
        ftr_maps.append(h.output.cpu().numpy())

    data = np.asarray(ftr_maps)
    np.savez_compressed(np_file, data)


def find_range(data):
    max_vals = np.amax(data, axis=(2,3))
    min_vals = np.amin(data, axis=(2,3))
    return max_vals - min_vals

#def binnify(bin_freq, val):
def binnify(data):
    bin_freq = [0] * 8
    # 0: [0,0], 1: (0,1], 2: (1,2], 3: (2,4]
    # 4: (4,6], 5: (6,8], 6: (8,10], 7: > 10
    for val in data:
        if val == 0:
            bin_freq[0] += 1

        elif val > 0 and val <= 1:
            bin_freq[1] += 1

        elif val > 1 and val <= 2:
            bin_freq[2] += 1

        elif val > 2 and val <= 4:
            bin_freq[3] += 1

        elif val > 4 and val <= 6:
            bin_freq[4] += 1

        elif val > 6 and val <= 8:
            bin_freq[5] += 1

        elif val > 8 and val <= 10:
            bin_freq[6] += 1

        elif val > 10:
            bin_freq[7] += 1
    return bin_freq


def read_data(np_file, w_sz):
    dict_ = np.load(np_file)
    data = dict_['arr_0'][0]
    n, c, h, w = data.shape # input feature map

    stride = w_sz
    xy_sz = math.floor(((h - w_sz) / stride) + 1)
    o_sz = (n, c, int(xy_sz), int(xy_sz)) # analysis output size
    print(o_sz)

    output = np.zeros(o_sz)

    for i in range(xy_sz):
        for j in range(xy_sz):
            output[:,:,i,j] = find_range(data[:,:,i*w_sz:(i+1)*w_sz,j*w_sz:(j+1)*w_sz])

    #output = output.flatten() * 100 # scaling up
    output = output.flatten()
    bin_freq = binnify(output)
    total = sum(bin_freq)
    bin_freq = [i/total for i in bin_freq]
    #bin_fn = np.vectorize(binnify)
    #bin_fn(bin_freq, output)
    #bin_fn(output)
    overall_range = np.amax(output) - np.amin(output)
    print('overall range: {}'.format(overall_range))
    filenm = np_file[:-4] + '_layer1.csv'
    with open(filenm, mode='w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['0-0', '(0-1]', '(1-2]', '(2-4]', '(4-6]', '(6-8]', '(8-10]', '> 10'])
        writer.writerow(bin_freq)

    #print('overall range scaled up by 100: {}'.format(overall_range))
    #hist, bin_edges = np.histogram(output, density=True)
    #from matplotlib import pyplot as plt

    #plt.hist(output, bins=10, density=True)
    #plt.hist(output, density=True)
    #plt.plot(bin_freq)
    #plt.savefig(filenm)

    #print(output[0,0,0:w_sz,0:w_sz])

#read_data(np_file='resnet18.npz', w_sz=3)

'''
outputs = net(inputs.float())

num_entries = 0
for h in hooks:
    sz = 1
    for a in h.output.size():
        sz *= a
    num_entries += sz
num_entries *= 4 # Bytes
in_sz = 3 * 224 * 224 * 4
print('input size: {} MB'.format(in_sz / math.pow(2, 20)))
print('forward pass size: {} MB'.format(num_entries / math.pow(2, 20)))
'''

if __name__ == '__main__':
    main()
