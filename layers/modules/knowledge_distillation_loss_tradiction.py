import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp, match_3_terms
GPU = False
if torch.cuda.is_available():
    GPU = True


class KD_loss_tradiction(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g) + \beta Lbinconf(x, c)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """


    def __init__(self, num_classes,overlap_thresh,prior_for_matching,bkg_label,neg_mining,neg_pos,neg_overlap,encode_target):
        super(KD_loss_tradiction, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching  = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1,0.2]
    '''
    def TK_regularization(self, p_soft, q_soft):
        tao = 2
        loss_TD = 0
        p = p_soft / tao
        q = q_soft / tao
        max_p, _ = torch.max(p, 1)
        max_q, _ = torch.max(q, 1)
        p = p.t() - max_p
        q = q.t() - max_q
        p_sum_exp = torch.exp(p).sum(0)
        q_sum_exp = torch.exp(q).sum(0)
        p_probability = torch.exp(p) / p_sum_exp
        q_probability = torch.exp(q) / q_sum_exp
        # loss_TD_all = (- p_probability * torch.log(q_probability) + p_probability * torch.log(p_probability)).sum()
        loss_TD = (- p_probability * torch.log(q_probability) + p_probability * torch.log(p_probability)).sum()
        return loss_TD
    '''

    def TK_regularization(self, p_soft, q_soft):
        tao = 2
        loss_TD = 0
        p = p_soft / tao
        q = q_soft / tao
        max_p, _ = torch.max(p, 1)
        max_q, _ = torch.max(q, 1)
        p = p.t() - max_p
        q = q.t() - max_q
        p_sum_exp = torch.exp(p).sum(0)
        q_sum_exp = torch.exp(q).sum(0)
        p_probability = torch.exp(p) / p_sum_exp
        q_probability = torch.exp(q) / q_sum_exp
        # loss_TD_all = (- p_probability * torch.log(q_probability) + p_probability * torch.log(p_probability)).sum()
        loss_TD = (- p_probability * torch.log(q_probability)).sum()
        return loss_TD

    def forward(self, source_out, target1_out, target1_out1, priors, targets, pos, neg):
        """Knowledge distillation Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                bin_conf: torch.size(batch_size,num_priors,2)
                priors shape: torch.size(num_priors,4)

            prediction_target (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                bin_conf: torch.size(batch_size,num_priors,2)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        kd_loss_l = 0
        kd_loss_c = 0

        loc_data, conf_data, bin_conf_data = source_out
        loc_data_target, conf_data_target, bin_conf_data_target = target1_out
        loc_data_target_prediction, conf_data_target_prediction, bin_conf_data_target_prediction = target1_out1

        # Localization KD_Loss (L2)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_source = loc_data[(pos_idx).gt(0)].view(-1, 4)
        loc_target = loc_data_target[(pos_idx).gt(0)].view(-1, 4)
        # kd_loss_l += F.mse_loss(loc_source, loc_target)
        kd_loss_l += ((loc_source - loc_target) ** 2).sum()

        num = bin_conf_data.shape[0]

        # Confidence KD_Loss
        # pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(conf_data)
        neg_idx = neg.unsqueeze(neg.dim()).expand_as(conf_data)
        conf_source = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        conf_target = conf_data_target[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        kd_loss_c = self.TK_regularization(conf_source, conf_target)


        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g) + \beta Lbinconf(x, c)) / N
        num_pos = pos.long().sum(1, keepdim=True)
        N = max(num_pos.data.sum().float(), 1)
        kd_loss_l/=N
        kd_loss_c/=N
        return kd_loss_l, kd_loss_c

