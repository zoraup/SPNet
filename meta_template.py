import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import utils
from abc import abstractmethod
import cv2

class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way = False):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = -1 #(change depends on input) 
        self.feature    = model_func()
        self.feat_dim   = self.feature.final_feat_dim

    '''
    @abstractmethod:Abstract a base class and specify which methods to use, but only abstract methods, do not implement 
                    functions, the class can only be inherited, not instantiated, but subclasses must implement the methods.
    '''
    @abstractmethod
    def set_forward(self,x,is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self,x):
        out  = self.feature.forward(x)
        return out

    def parse_feature(self,x,is_feature):
        x    = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) #images:(5*21)*3*224*224
            z_all       = self.feature.forward(x)       #features:(5*21)*512*1*1
            c = z_all.shape[1]
            h = z_all.shape[-2]
            w = z_all.shape[-1]
            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, c)   #5*21*512*1*1
        z_support   = z_all[:, :self.n_support]       #5*5*512*1*1
        z_query     = z_all[:, self.n_support:]       #5*16*512*1*1
        return z_support, z_query

    def correct(self, x):       
        scores1, scores2, scores3 = self.set_forward(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores1.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer, scheduler, const1, const2):
        print_freq = 10
        avg_loss=0
        for i, (x,_ ) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            ###### add calibration loss ########################################################################
            loss1, loss2, loss3 = self.set_forward_loss( x )
            loss = loss1 + const1 * loss2 + const2 * loss3
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()

            if i % print_freq==0:
                print('Epoch [%d], Batch [%d/%d], Loss: %.6f, lr: %f' % (epoch, i, len(train_loader), avg_loss/float(i+1), scheduler.get_lr()[0]))
        scheduler.step(epoch)
    def test_loop(self, test_loader, record = None):
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this*100  )

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        return acc_mean
