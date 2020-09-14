import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.autograd import Variable
from utils.box_utils import decode
from utils.nms_wrapper import nms
import numpy as np


class Detect_tf_soft_cls(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, cfg):
        self.num_classes = num_classes
        self.background_label = bkg_label

        self.variance = cfg['variance']

    def forward(self, predictions, prior, scale):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            bin_conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """

        loc, conf, bin_conf = predictions

        loc_data = loc.data
        conf_data = conf.data
        bin_conf_data = bin_conf.data
        prior_data = prior.data
        num = loc_data.size(0)  # batch size
        self.num_priors = prior_data.size(0)
        self.boxes = torch.zeros(1, self.num_priors, 4)
        self.bin_scores = torch.zeros(1, self.num_priors, 2)
        self.scores = torch.zeros(1, self.num_priors, self.num_classes)

        if loc_data.is_cuda:
            self.boxes = self.boxes.cuda()
            self.bin_scores = self.bin_scores.cuda()
            self.scores = self.scores.cuda()

        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.unsqueeze(0)
            bin_conf_preds = bin_conf_data.unsqueeze(0)

        else:
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes)
            bin_conf_preds = bin_conf_data.view(num, num_priors,
                                        self.num_classes)
            self.boxes.expand_(num, self.num_priors, 4)
            self.bin_scores.expand_(num, self.num_priors, 2)
            self.scores.expand_(num, self.num_priors, self.num_classes)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()
            bin_scores = bin_conf_preds[i].clone()

            soft_conf_scores = torch.zeros(self.num_priors, self.num_classes)
            soft_conf_scores[:, 1:] = (conf_scores[:, ].t() * bin_scores[:, 1]).t()
            soft_conf_scores[:, 0] = bin_scores[:, 0]

            self.boxes[i] = decoded_boxes
            self.scores[i] = soft_conf_scores
            #self.scores[i] = conf_scores
            #self.scores[i] = bin_scores

        return self.boxes, self.scores
