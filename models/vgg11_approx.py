import torch.nn as nn
from .common import approximator

layer_types = [nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.BatchNorm2d, \
        nn.ReLU, nn.AdaptiveAvgPool2d, nn.Dropout]

def create_approximate(net, w_sz, order, device):
    modules = []
    modules2 = []
    i = 0
    for n,v in net.named_modules():
        if isinstance(v, nn.Conv2d) or \
            isinstance(v, nn.Linear) or \
            isinstance(v, nn.MaxPool2d) or \
            isinstance(v, nn.BatchNorm2d) or \
            isinstance(v, nn.ReLU) or \
            isinstance(v, nn.AdaptiveAvgPool2d) or \
            isinstance(v, nn.Dropout):

            if i <= 21:
                #if i == 19:
                ##if i == 18:
                ##if i == 17:
                ##if i == 15:
                ##if i == 14:
                ##if i == 12:
                ##if i == 9:
                ##if i == 7:
                ##if i == 4:
                ##if i == 3:
                #if i == 2:
                #if i == 1:
                modules.append(v)
                if isinstance(v, nn.ReLU):
                    modules.append(approximator(w_sz, order, device))
                #modules.append(v)
            else:
                #if i == 22:
                #    modules.append(approximator(w_sz, order, device))

                modules2.append(v)
            i += 1

    class MyNet(nn.Module):
        def __init__(self):
            super(MyNet, self).__init__()
            self.part1 = nn.Sequential(*modules)
            self.part2 = nn.Sequential(*modules2)

        def forward(self, x):
            x = self.part1(x)
            _,c,h,w = x.size()
            #print(c,h,w)
            x = x.view(-1, c*h*w)
            x = self.part2(x)
            return x

    new_net = MyNet()
    return new_net
