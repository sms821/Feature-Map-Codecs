import torch.nn as nn
from .common import *

def conv_bn(inp, oup, stride, device=None):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        ## my edits: sms821
        approximator(device),

        nn.BatchNorm2d(oup),
        approximator(device),

        nn.ReLU6(inplace=True),
        approximator(device),
    )


def conv_1x1_bn(inp, oup, device=None):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        ## my edits: sms821
        approximator(device),

        nn.BatchNorm2d(oup),
        approximator(device),

        nn.ReLU6(inplace=True),
        approximator(device),
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, device=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        #if self.use_res_connect:
        #    print('RESIDUAL!!')

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                approximator(device),

                nn.BatchNorm2d(hidden_dim),
                approximator(device),

                nn.ReLU6(inplace=True),
                approximator(device),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                approximator(device),

                nn.BatchNorm2d(oup),
                approximator(device),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                approximator(device),

                nn.BatchNorm2d(hidden_dim),
                approximator(device),

                nn.ReLU6(inplace=True),
                approximator(device),

                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                approximator(device),

                nn.BatchNorm2d(hidden_dim),
                approximator(device),

                nn.ReLU6(inplace=True),
                approximator(device),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                approximator(device),

                nn.BatchNorm2d(oup),
                approximator(device),
            )
            self.adder = Adder()
            self.approx = approximator(device)

    def forward(self, x):
        if self.use_res_connect:
            x = self.adder(x, self.conv(x))
        else:
            x = self.conv(x)
        return x


class MobileNetV2_approx(nn.Module):
    def __init__(self, device, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2_approx, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        # my edits: sms821
        self.features = [conv_bn(3, input_channel, 2, device=device)]

        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, device=device))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, device=device))

                input_channel = output_channel

        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel, device))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


'''
def apply_approx_config(net, approx_config):
    approx_layers = []
    for name, m in net.named_modules():
        if isinstance(m, approximator):
            approx_layers.append(m)

    for i, (k,v) in enumerate(approx_config.items()):
        approx_layer = approx_layers[i]

        approx_layer.w_sz = v.as_int('window')
        approx_layer.order = v.as_int('order')
        approx_layer.approx = v.as_bool('approx')


def create_approximate(net, device, approx_config):
    #model = MobileNetV2_approx(order, w_sz, device, width_mult=1)
    model = MobileNetV2_approx(device, width_mult=1)
    model = copy_weights(model, net)
    apply_approx_config(model, approx_config)
    return model
'''

'''
def create_approximate(net, w_sz, device, layer_num, order):
    layer_dict = serialize_model(net)
    #print_dict(layer_dict)
    modules = []
    modules2 = []
    i = 0
    for k in list(layer_dict.keys()):
        if (isinstance(layer_dict[i], nn.Linear)):
            modules2.append(layer_dict[i])
            if i == layer_num:
                modules2.append(approximator(w_sz, order, device))
        else:
            modules.append(layer_dict[i])
            if i == layer_num:
                modules.append(approximator(w_sz, order, device))
        i += 1

    #return (modules, modules2)

    class MobV2_approx(nn.Module):
        def __init__(self):
            super(MobV2_approx, self).__init__()
            self.part1 = nn.Sequential(*modules)
            self.part2 = nn.Sequential(*modules2)

        def forward(self, x):
            x = self.part1(x)
            _,c,h,w = x.size()
            print(c,h,w)
            x = x.view(-1, c*h*w)
            x = self.part2(x)
            return x

    new_net = MobV2_approx()
    return new_net
'''
