# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 22:14:37 2019

@author: Harshit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import random

import pytorch_ssim as ssim
import cv2
import os
import numpy as np  
import torchvision.models as models
from tensorboard_logger import configure, log_value

class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            vgg - encoder pre-trained with VGG11
        """
        super(UNet16,self).__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        if pretrained == 'vgg':
            self.encoder = torchvision.models.vgg16(pretrained=True).features
        else:
            self.encoder = torchvision.models.vgg16(pretrained=False).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = self.final(dec1)
        else:
            x_out = self.final(dec1)

        return x_out

class conv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding):
        super(conv2d,self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = kernel_size,padding = padding)
        self.batch = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self,X):
        X = self.conv(X)
        X = self.batch(X)
        X = self.relu(X)
        return X

class conv_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,codeword):
        super(conv_block,self).__init__()
        self.conv1 = conv2d(in_channels,out_channels,kernel_size,padding)
        if codeword == 'C':
            self.conv2 = conv2d(out_channels,out_channels,kernel_size,padding)
        else:
            self.conv2 = conv2d(out_channels,out_channels//2,kernel_size,padding)
        
    def forward(self,X):
        X = self.conv1(X)
        X = self.conv2(X)
        return X
    
class up_sample(nn.Module):
    def __init__(self):
        super(up_sample,self).__init__()
        
    def forward(self,X):
        return F.interpolate(X,scale_factor = (2,2))

class down_sample(nn.Module):
    def __init__(self):
        super(down_sample,self).__init__()
        
    def forward(self,X):
        return F.max_pool2d(X,kernel_size = (2,2))
        
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.model = models.vgg16_bn(pretrained=True).features
        self.vgg.features[0].weight = nn.Parameter(self.vgg.features[0].weight.sum(dim=1).unsqueeze_(1))
        self.conv_block1 = conv_block(1,64,(3,3),(1,1),'C')
        self.conv_block2 = conv_block(64,128,(3,3),(1,1),'C')
        self.conv_block3 = conv_block(128,256,(3,3),(1,1),'C')
        self.conv_block4 = conv_block(256,512,(3,3),(1,1),'C')
        self.conv_block5 = conv_block(512,1024,(3,3),(1,1),'D')
        self.conv_block6 = conv_block(1024,512,(3,3),(1,1),'D')
        self.conv_block7 = conv_block(512,256,(3,3),(1,1),'D')
        self.conv_block8 = conv_block(256,128,(3,3),(1,1),'D')
        self.conv_block9 = conv_block(128,64,(3,3),(1,1),'C')
        
        self.down = down_sample()
        self.up = up_sample()
        
        self.conv_1x1 = nn.Conv2d(in_channels = 64,out_channels = 2,kernel_size = (1,1))
          
    def forward(self,X):
        '''first = self.conv_block1(X)
        X = self.down(first)
        second = self.conv_block2(X)
        X = self.down(second)
        third = self.conv_block3(X)
        X = self.down(third)
        fourth = self.conv_block4(X)
        X = self.down(fourth)'''
        
        X = self.model(X)
        
        import pdb;pdb.set_trace()
        
        
        
        fifth = self.conv_block5(X)
        X = self.up(fifth)
        #import pdb;pdb.set_trace()
        sixth = torch.cat((X,fourth),dim = 1)
        sixth = self.conv_block6(sixth)
        X = self.up(sixth)
        seventh = torch.cat((X,third),dim = 1)
        seventh = self.conv_block7(seventh)
        X = self.up(seventh)
        eigth = torch.cat((X,second),dim = 1)
        eigth = self.conv_block8(eigth)
        X = self.up(eigth)
        ninth = torch.cat((X,first),dim= 1)
        ninth = self.conv_block9(ninth)
        
        out = self.conv_1x1(ninth)
        
        return out
    
class Data(Dataset):
    def __init__(self,path,transforms):
        self.path = path
        #self.transform = transform
        self.images = list(os.listdir(path))
        self.transform = transforms   
          
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        img = cv2.imread(os.path.join(self.path,self.images[index]))
        lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        lab = self.transform(lab)
        L,A,B = cv2.split(np.array(lab))
        
        L = L/255.
        A = A/255.
        B = B/255.
        X_train = torch.from_numpy(L).unsqueeze_(0).float()
        Y_train = cv2.merge((A,B))
        return X_train,Y_train, os.path.join(self.path,self.images[index])
    
def loss_function(ab,AB,loss):
    # s = StandardScaler()
    b,c,_,_ = ab.shape
    ab = ab.view(b,c,-1).view(b,-1).view(-1).reshape(-1,1)
    AB = AB.view(b,c,-1).view(b,-1).view(-1).reshape(-1,1)
    # s.fit(AB)
    # AB=s.transform(AB)
    #ab = torch.from_numpy(ab)
    AB = AB.float().cuda()
    
    #import pdb;pdb.set_trace()
    batch_loss = loss(ab,AB)
    return batch_loss


def visualize(model, epoch, data_dir):
    t = [transforms.ToPILImage(),transforms.Resize((224,224))]
    t = transforms.Compose(t)
    data = Data(data_dir,transforms=t)

    for i in range(10):
        X, Y, orig_im = data[i]
        orig_im = cv2.imread(orig_im)
        X = X.unsqueeze(0).cuda()
        ab = model(X)
        ab = ab.data.cpu().numpy()
        ab = np.clip(ab, 0, 1)
        X = X.data.cpu().numpy()
        X = X[0].transpose(1,2,0)
        ab = ab[0].transpose(1,2,0)
        img = cv2.merge((X,ab))
        img = (img*255).astype(np.uint8)
        img = cv2.cvtColor(img,cv2.COLOR_LAB2BGR)
        if not os.path.isdir("Sanity"):
            os.makedirs("Sanity")
        orig_im = cv2.resize(orig_im, (224,224))
        cv2.imwrite("Sanity/{}_{}.jpg".format(epoch, i), np.concatenate((img, orig_im), axis=1))
     
def train(data_dir):
    
    epochs = 2000
    learning_rate = 0.001
    gamma = 0.9
    
    t = [transforms.ToPILImage(),transforms.Resize((224,224))]
    t = transforms.Compose(t)
    data = Data(data_dir,transforms=t)
    loader = torch.utils.data.DataLoader(data, batch_size = 24, shuffle = True, num_workers=4)
    
    model = Model().cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    loss1 = nn.SmoothL1Loss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=gamma)
    
    for epoch in range(epochs):
        count = 0
        train_loss = 0.
        for i,(X,AB,_) in enumerate(loader):
            optimizer.zero_grad()
            
            X = X.cuda()
            AB = AB.float().cuda()
            ab = model(X)
            
            loss = loss_function(ab,AB,loss1)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_loss += loss.item()
            count+=1
        #import pdb;pdb.set_trace()
        torch.save(model.state_dict,'{}/model_{}.pth'.format(out_dir,epoch))    
        print('Epoch : {}, Loss : {}'.format(epoch, train_loss/count))
        visualize(model, epoch, data_dir)
        # print('Epoch : {}, Mean ssim score(U) = {}, Mean ssim score(V) = {}'.format(epoch, u_sim/count, v_sim/count))

        log_value('loss', train_loss/count, epoch)
        
        # log_value('ssim/U', u_sim/count, i)
        # log_value('ssim/V', v_sim/count, i)

if __name__ == '__main__':

    out_dir = "./model_harshit_8"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    configure(out_dir, flush_secs=5)

    train('test2014')
    

        
        
        