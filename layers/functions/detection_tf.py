import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.autograd import Variable
from utils.box_utils import decode
from utils.nms_wrapper import nms
import numpy as np


class Detect_tf(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, proposal_top, top_k, cfg, force_cpu):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.proposal_top = proposal_top
        self.top_k = top_k

        self.variance = cfg['variance']
        self.force_cpu = force_cpu

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
        self.boxes = torch.zeros(1, self.top_k, 4)
        self.bin_scores = torch.zeros(1, self.top_k, 2)
        self.scores = torch.zeros(1, self.top_k, self.num_classes)
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
            self.boxes.expand_(num, self.top_k, 4)
            self.bin_scores.expand_(num, self.top_k, 2)
            self.scores.expand_(num, self.top_k, self.num_classes)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()
            bin_scores = bin_conf_preds[i].clone()
            bin_scores_sorted, bin_scores_indices = torch.sort(bin_scores[:, 1], descending=True)

            scale_boxs = decoded_boxes * scale
            c_dets = torch.cat((scale_boxs[bin_scores_indices[: self.proposal_top]],
                                bin_scores[bin_scores_indices[: self.proposal_top], 1].unsqueeze(1)), 1)
            c_dets = c_dets.cpu().numpy()
            keep = nms(c_dets, 0.65, self.force_cpu)
            count = np.minimum(len(keep), self.top_k)
            selected_keep = keep[: count]
            selected_indices = bin_scores_indices[selected_keep]

            self.boxes[i,:count] = decoded_boxes[selected_indices]
            self.scores[i, :count] = conf_scores[selected_indices]

        return self.boxes, self.scores
