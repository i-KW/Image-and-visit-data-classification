import numpy as np
from functools import partial
import pandas as pd
import os
from tqdm import tqdm_notebook, tnrange, tqdm
import sys

import torch
from torch import nn
from torch.nn.init import kaiming_normal
import torch.nn.functional as F
from torch.optim import SGD,Adam
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from torch.optim.optimizer import Optimizer

import torchvision
from torchvision import models
import pretrainedmodels
from pretrainedmodels.models import *
from torch import nn
from config import config
from collections import OrderedDict
import torch.nn.functional as F
from torchvision import transforms as T
from imgaug import augmenters as iaa
import random
import pathlib
import cv2

from sklearn import preprocessing


random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

# create dataset class
class MultiModalDataset(Dataset):
    def __init__(self,images_df, base_path,vis_path,augument=True,mode="train"):
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)
        if not isinstance(vis_path, pathlib.Path):
            vis_path = pathlib.Path(vis_path)
        self.images_df = images_df.copy() #csv
        self.augument = augument
        self.vis_path = vis_path #vist npy path,作者已经把visit数据生成好.npy文件
        self.images_df.Id = self.images_df.Id.apply(lambda x:base_path / str(x).zfill(6)) #原字符串右对齐，取前6位，不够则补0
        self.mode = mode

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self,index):
        visit=self.read_npy(index).transpose(1,2,0) #7*26*24的特征矩阵转换为26*24*7，将7作为通道数
        visit=visit.reshape(-1,1)  #转化为一维，一列的数组
        # print('111111',visit.shape)
        min_max_scaler = preprocessing.MinMaxScaler()
        visit01 = min_max_scaler.fit_transform(visit)

        # visit = visit + visit01
        # visit = visit.append(visit01)
        visit = visit01
        # visit =  np.column_stack((visit,visit01))
        # print('222222',visit.shape)

        # print(visit)


        if not self.mode == "test": #如果不是“test”模式，则执行下一语句
            y = self.images_df.iloc[index].Target #iloc[]根据标签的所在位置，从0开始计数，选取数据
        else:
            y = str(self.images_df.iloc[index].Id.absolute())

        visit=T.Compose([T.ToTensor()])(visit)
        return visit.float(),y  #返回的是访问数据以及label


    def read_npy(self,index):  #读取访问数据的函数
        row = self.images_df.iloc[index]
        filename = os.path.basename(str(row.Id.absolute())) #返回文件名
        pth=os.path.join(self.vis_path.absolute(),filename+'.npy')
        visit=np.load(pth)
        return visit


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.optimizer = optimizer
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]
    
    def _reset(self, epoch, T_max):
        """
        Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        return CosineAnnealingLR(self.optimizer, self.T_max, self.eta_min, last_epoch=epoch)



class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    
'''Dual Path Networks in PyTorch.'''
class Bottleneck(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        self.conv1 = nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, out_planes+dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes+dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, out_planes+dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes+dense_depth)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        x = self.shortcut(x)
        d = self.out_planes
        out = torch.cat([x[:,:d,:,:]+out[:,:d,:,:], x[:,d:,:,:], out[:,d:,:,:]], 1)
        out = F.relu(out)
        return out


class DPN(nn.Module):
    def __init__(self, cfg):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        self.conv1 = nn.Conv2d(7, 64, kernel_size=3, stride=1, padding=1, bias=False) #
        self.bn1 = nn.BatchNorm2d(64)
        self.last_planes = 64
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear(out_planes[3]+(num_blocks[3]+1)*dense_depth[3], 64) 

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i,stride in enumerate(strides):
            layers.append(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i==0))
            self.last_planes = out_planes + (i+2) * dense_depth
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def DPN26():
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (2,2,2,2),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg)

def DPN92():
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (3,4,20,3),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg)


#1维卷积
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2),  # straid是1时，维度不变，令padding = 0 #size为3是0.591
            nn.BatchNorm1d(64),
            nn.ReLU())

        # 4368*1
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=24, stride=1, padding=23),  # straid是1时，维度不变，令padding = 0
            nn.BatchNorm1d(64),
            nn.ReLU())
        # 4368*64
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=24, stride=24, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU())
            # nn.MaxPool1d(2, stride=2))  # 2 2 维度变为1/2，channel不变
        # 182*128

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=6),
            nn.BatchNorm1d(128),
            nn.ReLU())
            # nn.MaxPool1d(3, stride=3))

        # 182*128


        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, stride=7, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))

        # 13*256

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))

        # 6*256
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
            # nn.Dropout(config.DROPOUT)
        )

        # 3*256
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU())
            # nn.MaxPool1d(3, stride=3))

        # 3*256
        self.conv9 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))

        # 1 x 512
        self.conv10 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
            # nn.Dropout(config.DROPOUT)
        )

        # 1 x 512
        self.fc = nn.Linear(512, 50)
        self.activation = nn.Sigmoid()


    def forward(self, x):
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width

        x = x.view(x.shape[0], 1, -1)  #改成2
        # x : 23 x 1 x 59049

        out = self.conv1(x)
        out = self.conv2(out)
        # out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        # out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)

        out = out.view(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc(out)

        # logit = self.activation(logit)

        return logit







class MultiModalNet(nn.Module):
    def __init__(self, backbone2, drop, pretrained=True):
        super().__init__()
       
        # self.visit_model=DPN92()
        self.visit_model = CNN1D()

        self.cls = nn.Linear(64,config.num_classes)

    def forward(self, x_vis):
        # x_img = self.img_encoder(x_img)
        # x_img = self.img_fc(x_img)

        x_vis=self.visit_model(x_vis)
        # x_cat = torch.cat((x_img,x_vis),1)
        # x_cat = self.cls(x_cat)
        return x_vis
