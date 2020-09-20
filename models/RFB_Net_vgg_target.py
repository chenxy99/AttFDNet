import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os

multiply_embedding = 20

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out



class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4


        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out


class Classifier(nn.Module):
    def __init__(self, pre_classes, num_classes):
        super(Classifier,self).__init__()
        self.fc = nn.Linear(pre_classes, num_classes, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x

class normal_Classifier(nn.Module):
    def __init__(self, pre_classes, num_classes):
        super(normal_Classifier,self).__init__()
        self.fc = nn.Linear(pre_classes, num_classes, bias=True)

    def forward(self, x):
        x = self.fc(x)
        return x


class RFBNet(nn.Module):
    """RFB Net for object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) conv2d for object or not predictions bin_conf_layers
        4) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1711.07767.pdf for more details on RFB Net.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes, num_classes_target1):
        super(RFBNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.num_classes_target1 = num_classes_target1
        self.size = size

        if size == 300:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            print("Error: Sorry only SSD300 and SSD512 are supported!")
            return
        # vgg network
        self.base = nn.ModuleList(base)
        # conv_4
        self.Norm = BasicRFB_a(512,512,stride = 1,scale=1.0)
        self.extras = nn.ModuleList(extras)


        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.bin_conf = nn.ModuleList(head[2])

        self.loc_target1 = nn.ModuleList(head[3])
        self.bin_conf_target1 = nn.ModuleList(head[4])

        # minus 1
        self.classifier_source = Classifier(self.num_classes * multiply_embedding, self.num_classes)
        self.classifier_target1 = Classifier(self.num_classes * multiply_embedding, self.num_classes + self.num_classes_target1)
        self.scale_source = nn.Parameter(torch.FloatTensor([10]))

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.softmax_binary = nn.Softmax(dim=-1)

        # GC block
        self.GCBlock = GCBlock(512, 4)

    def forward(self, x, feed_conv_4_3):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        bin_conf = list()
        loc_target1 = list()
        conf_target1 = list()
        bin_conf_target1 = list()
        # traces = list()

        # apply vgg up to conv4_3 relu

        for k in range(23):
            x = self.base[k](x)

        x = self.GCBlock(x)

        x = x * torch.log(feed_conv_4_3 + 2.71828)
        '''
        for k in range(19):
            x = self.base[k](x)

        x = self.GCBlock(x)

        x = x * torch.log(feed_conv_4_3 + 2.71828)
        for k in range(19, 23):
            x = self.base[k](x)
        '''

        s = self.Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k%2 ==0:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c, b) in zip(sources, self.loc, self.conf, self.bin_conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            bin_conf.append(b(x).permute(0, 2, 3, 1).contiguous())

        # apply multibox head to source layers
        for (x, l, b) in zip(sources, self.loc_target1, self.bin_conf_target1):
            loc_target1.append(l(x).permute(0, 2, 3, 1).contiguous())
            bin_conf_target1.append(b(x).permute(0, 2, 3, 1).contiguous())

        #print([o.size() for o in loc])


        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        bin_conf = torch.cat([o.view(o.size(0), -1) for o in bin_conf], 1)

        loc_target1 = torch.cat([o.view(o.size(0), -1) for o in loc_target1], 1)
        bin_conf_target1 = torch.cat([o.view(o.size(0), -1) for o in bin_conf_target1], 1)

        # embedding
        embedding_source = conf.view(-1, self.num_classes * multiply_embedding)
        embedding_source_norm = self.l2_norm(embedding_source)
        embedding_source_output = embedding_source_norm.view(loc.size(0), -1, self.num_classes * multiply_embedding)

        conf_target_stand_dist = self.classifier_target1(embedding_source_norm)
        conf_target_stand_dist = conf_target_stand_dist.view(conf.shape[0], -1)
        conf_target1 = conf_target_stand_dist * self.scale_source
        conf_target1 = conf_target1.view(conf.shape[0], -1)


        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf_target1.view(-1, self.num_classes + self.num_classes_target1)),  # conf preds
                self.softmax_binary(bin_conf.view(-1, 2)),      # bin_conf preds
            )
            output1 = (
                loc_target1.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf_target1.view(-1, self.num_classes + self.num_classes_target1)),  # conf preds
                self.softmax_binary(bin_conf_target1.view(-1, 2)),  # bin_conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf_target1.view(loc.size(0), -1, self.num_classes + self.num_classes_target1),
                bin_conf.view(bin_conf.size(0), -1, 2),
            )
            output1 = (
                loc_target1.view(loc_target1.size(0), -1, 4),
                conf_target1.view(conf_target1.size(0), -1, self.num_classes + self.num_classes_target1),
                bin_conf_target1.view(bin_conf_target1.size(0), -1, 2),
            )
        conf_target_stand_dist = conf_target_stand_dist.view(loc.size(0), -1, self.num_classes + self.num_classes_target1)
        return output, output1, embedding_source_output, conf_target_stand_dist

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def weight_source_norm(self):
        w = self.classifier_source.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier_source.fc.weight.data = w.div(norm.expand_as(w))

    def weight_target1_norm(self):
        w = self.classifier_target1.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier_target1.fc.weight.data = w.div(norm.expand_as(w))

class GCBlock(nn.Module):
    def __init__(self, Cin, ratio):
        super(GCBlock, self).__init__()
        self.conv_WK = nn.Conv2d(Cin, 1, kernel_size=1, bias=True)
        #self.softmax = nn.Softmax(dim=1)
        self.conv_WV1 = nn.Conv2d(Cin, Cin//ratio, kernel_size=1, bias=True)
        self.LayerNorm = nn.LayerNorm([Cin//ratio, 1, 1])
        #self.relu = nn.ReLU(inplace=True)
        self.conv_WV2 = nn.Conv2d(Cin // ratio, Cin, kernel_size=1, bias=True)


    def forward(self, x):
        N, C, H, W = x.shape
        hm = F.softmax(self.conv_WK(x).view(N, -1, 1, 1), dim=1)
        x_c = x.view(N, C, -1)
        #a = hm.unsqueeze(1)
        y = torch.matmul(x_c.unsqueeze(1), hm.unsqueeze(1).squeeze(4)).view(N, C, 1, 1)
        y = self.conv_WV1(y)
        y = self.LayerNorm(y)
        y = F.relu(y, inplace=True)
        y = self.conv_WV2(y)
        y = y + x
        y = F.relu(y, inplace=True)
        return y


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}


def add_extras(size, cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if in_channels == 256 and size == 512:
                    layers += [BasicRFB(in_channels, cfg[k+1], stride=2, scale = 1.0, visual=1)]
                else:
                    layers += [BasicRFB(in_channels, cfg[k+1], stride=2, scale = 1.0, visual=2)]
            else:
                layers += [BasicRFB(in_channels, v, scale = 1.0, visual=2)]
        in_channels = v
    if size == 512:
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=4,stride=1,padding=1)]
    elif size ==300:
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=3,stride=1)]
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=3,stride=1)]
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return
    return layers

extras = {
    '300': [1024, 'S', 512, 'S', 256],
    '512': [1024, 'S', 512, 'S', 256, 'S', 256,'S',256],
}


def multibox(size, vgg, extra_layers, cfg, num_classes, num_classes_target1):
    # for the source
    loc_layers = []
    conf_layers = []
    bin_conf_layers = []
    vgg_source = [-2]
    for k, v in enumerate(vgg_source):
        if k == 0:
            loc_layers += [nn.Conv2d(512,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(512,
                                 cfg[k] * num_classes * multiply_embedding, kernel_size=3, padding=1)]
            # in this program, I assume that it is a prediction that indicates whether this bounding box is an object
            bin_conf_layers += [nn.Conv2d(512,
                                 cfg[k] * 2, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes * multiply_embedding, kernel_size=3, padding=1)]
            # in this program, I assume that it is a prediction that indicates whether this bounding box is an object
            bin_conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * 2, kernel_size=3, padding=1)]
    i = 1
    indicator = 0
    if size == 300:
        indicator = 3
    elif size == 512:
        indicator = 5
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    for k, v in enumerate(extra_layers):
        if k < indicator or k%2== 0:
            loc_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                 * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                  * num_classes * multiply_embedding, kernel_size=3, padding=1)]
            # in this program, I assume that it is a prediction that indicates whether this bounding box is an object
            bin_conf_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                  * 2, kernel_size=3, padding=1)]
            i +=1




    # for the target1
    loc_layers_target1 = []
    bin_conf_layers_target1 = []
    vgg_source = [-2]
    for k, v in enumerate(vgg_source):
        if k == 0:
            loc_layers_target1 += [nn.Conv2d(512,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
            # in this program, I assume that it is a prediction that indicates whether this bounding box is an object
            bin_conf_layers_target1 += [nn.Conv2d(512,
                                          cfg[k] * 2, kernel_size=3, padding=1)]
        else:
            loc_layers_target1 += [nn.Conv2d(vgg[v].out_channels,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
            # in this program, I assume that it is a prediction that indicates whether this bounding box is an object
            bin_conf_layers_target1 += [nn.Conv2d(vgg[v].out_channels,
                                          cfg[k] * 2, kernel_size=3, padding=1)]
    i = 1
    indicator = 0
    if size == 300:
        indicator = 3
    elif size == 512:
        indicator = 5
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    for k, v in enumerate(extra_layers):
        if k < indicator or k % 2 == 0:
            loc_layers_target1 += [nn.Conv2d(v.out_channels, cfg[i]
                                     * 4, kernel_size=3, padding=1)]
            # in this program, I assume that it is a prediction that indicates whether this bounding box is an object
            bin_conf_layers_target1 += [nn.Conv2d(v.out_channels, cfg[i]
                                          * 2, kernel_size=3, padding=1)]
            i += 1

    # return vgg, extra_layers, (loc_layers, conf_layers, bin_conf_layers)
    return vgg, extra_layers, (loc_layers, conf_layers, bin_conf_layers, loc_layers_target1, bin_conf_layers_target1)

mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}

def build_net(phase, size=300, num_classes=21, num_classes_target1 = 21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    return RFBNet(phase, size, *multibox(size, vgg(base[str(size)], 3),
                                add_extras(size, extras[str(size)], 1024),
                                mbox[str(size)], num_classes, num_classes_target1), num_classes, num_classes_target1)
