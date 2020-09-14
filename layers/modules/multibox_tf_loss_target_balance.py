import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp, match_3_terms
GPU = False
if torch.cuda.is_available():
    GPU = True


class MultiBoxLoss_tf_target_balance(nn.Module):
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
        super(MultiBoxLoss_tf_target_balance, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching  = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1,0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                bin_conf: torch.size(batch_size,num_priors,2)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, bin_conf_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        bin_conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:,:-1].data
            labels = targets[idx][:,-1].data
            defaults = priors.data
            match_3_terms(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, bin_conf_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            bin_conf_t = bin_conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)
        bin_conf_t = Variable(bin_conf_t,requires_grad=False)

        pos = bin_conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)

        #[0.,  47.,  54.,  54.,  48.,  69.,  45.,  91.,  41., 189.,  51.,  57.,  51.,  43.,  48., 514.,  80., 107.,  68.,  32.,  65.]

        loss_l_elements = F.smooth_l1_loss(loc_p, loc_t, reduction='none').sum(1)
        conf_t_label = conf_t[conf_t>0]
        conf_t_stat = torch.zeros(num_classes).cuda()
        for index in range(1, num_classes):
            conf_t_stat[index] = (conf_t_label == index).sum()
        conf_t_weights = torch.zeros(num_classes).cuda()
        for index in range(1, num_classes):
            if conf_t_stat[index] > 0:
                conf_t_weights[index] = 1. / conf_t_stat[index]
        loss_l = 0
        #conf_t_weights = torch.exp(1-torch.tensor([0., 10., 12., 12., 10., 15., 11., 17., 16., 34., 14., 19., 15., 12., 10., 66., 16., 10.,
        # 17., 10., 15.])/66).cuda()
        conf_t_weights = torch.exp(torch.tensor([2,1.3930,    1.2377,    2.3323 ,   1.7176,    2.0606,    1.3950,    1.0631,   1.7324,    1.3473,    1.6245,    1.1843,    2.2063,
                                                 1.3083,    1.3608, 1.2330,    1.8409,    1.6459,    1.3869,    1.4504,    1.2606])).cuda()
        conf_t_weights = torch.ones(21).cuda()
        for index in range(1, num_classes):
            loss_l += conf_t_weights[index] * (loss_l_elements[conf_t_label == index]).sum()



        # Compute max binary_conf across batch for hard negative mining
        batch_bin_conf = bin_conf_data.view(-1, 2)
        loss_bin = log_sum_exp(batch_bin_conf) - batch_bin_conf.gather(1, bin_conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_bin[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_bin = loss_bin.view(num, -1)
        _, loss_idx = loss_bin.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        neg_binary = neg


        # Binary confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(bin_conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(bin_conf_data)
        bin_conf_p = bin_conf_data[(pos_idx + neg_idx).gt(0)].view(-1, 2)
        targets_weighted = bin_conf_t[(pos + neg).gt(0)]
        loss_bin_elements = F.cross_entropy(bin_conf_p, targets_weighted, reduction='none')

        conf_t_f_label_for_bin = conf_t[(pos + neg) > 0]
        conf_t_f_stat_for_bin = torch.zeros(num_classes).cuda()
        for index in range(0, num_classes):
            conf_t_f_stat_for_bin[index] = (conf_t_f_label_for_bin == index).sum()
        conf_t_f_weights_for_bin = torch.zeros(num_classes).cuda()
        for index in range(0, num_classes):
            if conf_t_f_stat_for_bin[index] > 0:
                conf_t_f_weights_for_bin[index] = 1. / conf_t_f_stat_for_bin[index]
        loss_b = 0
        #conf_t_f_weights_for_bin = torch.exp(1 - torch.tensor(
        #    [0., 10., 12., 12., 10., 15., 11., 17., 16., 34., 14., 19., 15., 12., 10., 66., 16., 10.,
         #    17., 10., 15.]) / 66).cuda()
        conf_t_f_weights_for_bin = torch.exp(torch.tensor(
            [2, 1.3930, 1.2377, 2.3323, 1.7176, 2.0606, 1.3950, 1.0631, 1.7324, 1.3473, 1.6245, 1.1843, 2.2063,
             1.3083, 1.3608, 1.2330, 1.8409, 1.6459, 1.3869, 1.4504, 1.2606])).cuda()
        conf_t_f_weights_for_bin = torch.ones(21).cuda()
        conf_t_f_weights_for_bin[0]*=3
        for index in range(0, num_classes):
            loss_b += conf_t_f_weights_for_bin[index] * (loss_bin_elements[conf_t_f_label_for_bin == index]).sum()




        # Compute max binary_conf across batch for hard negative mining
        batch_bin_conf = bin_conf_data.view(-1, 2)
        batch_conf = conf_data.view(-1, self.num_classes - 1)

        P_k = (batch_conf[:, ].t() + batch_bin_conf[:, 1]).t()
        P_0 =  batch_bin_conf[:, 0].unsqueeze(1) + torch.log(torch.exp(batch_conf).sum(dim = 1, keepdim = True))
        P_logit = torch.cat((P_0, P_k), dim=1).view(num, -1, self.num_classes)

        # Compute max conf across batch for hard negative mining
        batch_P_logit = P_logit.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_P_logit) - batch_P_logit.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        neg_multi = neg

        # Confidence Loss Including Positive and Negative Examples
        '''
        pos_idx = pos.unsqueeze(2).expand_as(P_logit)
        neg_idx = neg.unsqueeze(2).expand_as(P_logit)
        conf_p = P_logit[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_cls = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
        '''
        pos_idx = pos.unsqueeze(2).expand_as(P_logit)
        conf_p = P_logit[(pos_idx ).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos ).gt(0)]
        loss_mul_elements = F.cross_entropy(conf_p, targets_weighted, reduction='none')

        conf_t_f_label_for_mul = conf_t[(pos ) > 0]
        conf_t_f_stat_for_mul = torch.zeros(num_classes).cuda()
        for index in range(1, num_classes):
            conf_t_f_stat_for_mul[index] = (conf_t_f_label_for_mul == index).sum()
        conf_t_f_weights_for_mul = torch.zeros(num_classes).cuda()
        for index in range(1, num_classes):
            if conf_t_f_stat_for_mul[index] > 0:
                conf_t_f_weights_for_mul[index] = 1. / conf_t_f_stat_for_mul[index]
        loss_cls = 0
        #conf_t_f_weights_for_mul = torch.exp(1 - torch.tensor(
        #    [0., 10., 12., 12., 10., 15., 11., 17., 16., 34., 14., 19., 15., 12., 10., 66., 16., 10.,
        #     17., 10., 15.]) / 66).cuda()
        conf_t_f_weights_for_mul = torch.exp(torch.tensor(
            [2, 1.3930, 1.2377, 2.3323, 1.7176, 2.0606, 1.3950, 1.0631, 1.7324, 1.3473, 1.6245, 1.1843, 2.2063,
             1.3083, 1.3608, 1.2330, 1.8409, 1.6459, 1.3869, 1.4504, 1.2606])).cuda()
        conf_t_f_weights_for_mul = torch.ones(21).cuda()
        for index in range(1, num_classes):
            loss_cls += conf_t_f_weights_for_mul[index] * (loss_mul_elements[conf_t_f_label_for_mul == index]).sum()


        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g) + \beta Lbinconf(x, c)) / N

        N = max(num_pos.data.sum().float(), 1)
        loss_l/=N
        loss_cls/=N
        loss_b/=N
        return loss_l, loss_cls, loss_b, pos, neg_binary, neg_multi
