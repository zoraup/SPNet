# This code is modified from https://github.com/jakesnell/prototypical-networks

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate


class DualProtoNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(ProtoNet, self).__init__(model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)  # 5*5*512*1*1, 5*16*512*1*1

        z_support_lst = [z_support[i] for i in range(z_support.size(0))]
        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # 5*512(way,channel),average of 5 supports in each way
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)  # 80*512

        scores2 = []
        for z_suppor in z_support_lst:  # 5*512
            dist2 = cos_dist(z_suppor, z_proto)
            scores2.append(dist2)
        scores2 = torch.cat(scores2, dim=0).view(5, 5).cuda()  # 5*5
        #print(scores2)
        scores1 = cos_dist1(z_query, z_proto)  # 80*5

        scores3 = []
        for z_suppor in z_support_lst:  # 5*512
            scores1_proto = scores1.mean(0)
            dist3 = cos_dist2(z_suppor, scores1_proto)
            scores3.append(dist3)
        scores3 = torch.cat(scores3, dim=0).view(5, 5).cuda()

        # print(z_query,z_proto)
        return scores1, scores2, scores3

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(
            np.repeat(range(self.n_way), self.n_query))  # (0,0,...,0,1,1,...,1,...,4,4,...,4)  80*1
        y_query = Variable(y_query.cuda())
        y_support = torch.from_numpy(np.array(range(self.n_way)))
        y_support = Variable(y_support.cuda())
        scores1, scores2, scores3 = self.set_forward(x)
        loss1 = self.loss_fn(scores1.float(), y_query.long())  # 80*5,  80*1
        loss2 = self.loss_fn(scores2.float(), y_support.long())  # 5*5,   5*1
        loss3 = self.loss_fn(scores3.float(), y_support.long())  # 5*5,   5*1
        return loss1, loss2, loss3


def cos_dist(x, y):
    # x:n*d(5*512)
    # y:n*d(5*512) #proto
    n = x.size(0)
    d = x.size(1)
    assert d == y.size(1)
    #assert n == y.size(0)
    '''
    original code to calculate euclidean distance:
    x = x.unsqueeze(1).expand(n, n, d)
    y = y.unsqueeze(0).expand(n, n, d)
    dist = torch.pow(x - y, 2).sum(2).mean(0)
    '''

    ####5shot:
    
    x_lst = [x[i] for i in range(n)]
    cos_lst = []
    for x_c in x_lst:
        x_c = x_c.unsqueeze(0) #1*512
        cos = F.cosine_similarity(x_c, y, dim=1) * 50
        cos_lst.append(cos)
    cosi = torch.cat(cos_lst, dim=0).view(5, 5).mean(0)
    '''
    # 1shot:
    cosi = F.cosine_similarity(x, y, dim=1) * 50
    '''
    #print(cosi)
    return cosi


def cos_dist1(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)  # 80
    m = y.size(0)  # 5
    d = x.size(1)  # 512
    assert d == y.size(1)

    x_lst = [x[i] for i in range(n)]
    cos_lst = []
    for x_c in x_lst:
        x_c = x_c.unsqueeze(0) #1*512
        cos = F.cosine_similarity(x_c, y, dim=1) * 50
        cos_lst.append(cos)
    cosi = torch.cat(cos_lst, dim=0).view(80, 5)
    return cosi  # 80*5


def cos_dist2(x, y):
    # x: 5*512
    # y: 5
    n = x.size(0)  # 5
    d = y.size()  # 5

    y = y.view(5,1).cpu().detach().numpy()
    y = np.repeat(y, 512, axis=1) #5*512
    y = torch.from_numpy(y).cuda()
    ####5shot:
    
    x_lst = [x[i] for i in range(n)]
    cos_lst = []
    for x_c in x_lst:
        x_c = x_c.unsqueeze(0) #1*512
        cos = F.cosine_similarity(x_c, y, dim=1) * 50
        cos_lst.append(cos)
    cosi = torch.cat(cos_lst, dim=0).view(5, 5).mean(0)
    '''
    ## 1shot:
    cosi = F.cosine_similarity(x, y, dim=1) * 50
    '''
    return cosi #5
