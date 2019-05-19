
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 10:35:43 2019

@author: h.oberoi
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
        luv = cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
        luv = self.transform(luv)
        L,U,V = cv2.split(np.array(luv))

        
        
        L = L/255
        X_train = torch.from_numpy(L).unsqueeze_(0).float()
        
        #U = np.floor(U).astype(int)
        #V = np.floor(V).astype(int)
        U = self.bins_U(U).astype(int)
        V = self.bins_V(V).astype(int)
        #Y_train = cv2.merge((U,V))
        
        
        return X_train,U,V
    
    def bins_V(self,Y_train):
        
        #min = -140
        #max = 122
        Y_train = np.floor(Y_train/5)
        return Y_train
    
    def bins_U(self,Y_train):
        
        #min = -134
        #max = 220
        Y_train = np.floor(Y_train/5)
        return Y_train

        

def accuracy(Y_pred, Y_actual):
    Y_pred = Y_pred.reshape(-1)
    Y_actual = Y_actual.reshape(-1)
    return (np.sum((Y_pred==Y_actual)))/(len(Y_actual))


def loss_function(U_pred,U_actual,V_pred,V_actual,loss1,loss2):
    #total_loss = 0
    b,c,_,_ = U_pred.shape
    
    U_pred = U_pred.contiguous().view(b,c,-1).permute(0,2,1).contiguous().view(-1,c)
    U_actual = U_actual.contiguous().view(b,-1).contiguous().view(-1,1).squeeze_()
    
    #import pdb;pdb.set_trace()
    classes,counts = np.unique(U_actual,return_counts=True)
    med = int(np.median(counts))
    indices  = np.where(counts > med)[0].tolist()
    zero_list = list()
    for index  in indices :
        l = np.where(U_actual == int(classes[index]))[0].tolist()
        zero_list+= random.sample(l,len(l)-med)
    mask = np.ones(U_actual.shape[0])
    mask[zero_list] = 0
    mask = torch.from_numpy(np.array(mask).reshape(len(mask),1)).float().cuda()
    #print(U_pred.shape,mask.shape)
    
    U_pred = U_pred*mask
    
    b,c,_,_ = V_pred.shape
    V_pred = V_pred.contiguous().view(b,c,-1).permute(0,2,1).contiguous().view(-1,c)
    V_actual = V_actual.contiguous().view(b,-1).contiguous().view(-1,1).squeeze_()
    
    #import pdb;pdb.set_trace()
    
    classes,counts = np.unique(V_actual,return_counts=True)
    med = int(np.median(counts))
    indices  = np.where(counts > med)[0].tolist()
    zero_list = list()
    for index  in indices :
        l = np.where(V_actual == int(classes[index]))[0].tolist()
        zero_list+= random.sample(l,len(l)-med)
    mask = np.ones(V_actual.shape[0])
    mask[zero_list] = 0
    mask = torch.from_numpy(np.array(mask).reshape(len(mask),1)).float().cuda()
    V_pred = V_pred*mask
    
    l1 = loss1(U_pred,U_actual)
    l2 = loss2(V_pred,V_actual)
    return l1+l2


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
    def __init__(self,in_,out_):
        super(up_sample,self).__init__()
        self.convT = nn.ConvTranspose2d(in_channels= in_,out_channels = out_ ,padding = (1,1),kernel_size= (4,4),stride = (2,2))
        self.batch_norm = nn.BatchNorm2d(in_)
        self.relu = nn.ReLU()
        
    def forward(self,X):
        X = self.convT(X)
        X = self.batch_norm(X)
        X = self.relu(X)
        return X

class down_sample(nn.Module):
    def __init__(self):
        super(down_sample,self).__init__()
        
    def forward(self,X):
        return F.max_pool2d(X,kernel_size = (2,2))
    
class Model(nn.Module):
    def __init__(self,boolean):
        super(Model,self).__init__()
        self.model = models.vgg16_bn(pretrained=boolean)
        self.model.features[0].weight = nn.Parameter(self.model.features[0].weight.sum(dim=1).unsqueeze_(1))
        self.model = self.model.features;
        
        
        self.conv_block1 = self.model[:6]
        self.conv_block2 = self.model[7:13]
        self.conv_block3 = self.model[14:23]
        self.conv_block4 = self.model[24:33]
        self.conv_block5 = self.model[34:43]
        self.up_512_1 = up_sample(512,512)
        self.conv_block6 = conv_block(1024,512,(3,3),(1,1),'D')
        self.up_512_2 = up_sample(256,256)
        self.conv_block7 = conv_block(512,256,(3,3),(1,1),'D')
        self.up_256 = up_sample(128,128)
        self.conv_block8 = conv_block(256,128,(3,3),(1,1),'D')
        self.up_128 = up_sample(64,64)
        self.conv_block9 = conv_block(128,64,(3,3),(1,1),'C')
        
        self.down = down_sample()
        self.upsample = nn.Upsample(scale_factor=2)
        
        
        self.conv_1x1 = nn.Conv2d(in_channels = 64,out_channels = 22,kernel_size = (1,1))
          
    def forward(self,X):
        
        #X = self.model(X)
        
        first = self.conv_block1(X)
        #import pdb;pdb.set_trace()
        X = self.down(first)
        second = self.conv_block2(X)
        X = self.down(second)
        third = self.conv_block3(X)
        X = self.down(third)
        fourth = self.conv_block4(X)
        X = self.down(fourth)
        fifth = self.conv_block5(X)
        #import pdb;pdb.set_trace()
        X = self.up_512_1(fifth)
        #import pdb;pdb.set_trace()
        sixth = torch.cat((X,fourth),dim = 1)
        sixth = self.conv_block6(sixth)
        X = self.upsample(sixth)
        seventh = torch.cat((X,third),dim = 1)
        seventh = self.conv_block7(seventh)
        X = self.upsample(seventh)
        eigth = torch.cat((X,second),dim = 1)
        eigth = self.conv_block8(eigth)
        X = self.up_128(eigth)
        ninth = torch.cat((X,first),dim= 1)
        ninth = self.conv_block9(ninth)
        
        out = self.conv_1x1(ninth)
        return torch.relu(out)
    
class Final_layer(nn.Module):
    def __init__(self,model):
        super(Final_layer,self).__init__()
        self.model = model
        self.model.conv_1x1 = nn.Conv2d(64,51,(1,1))
        self.conv_1x1_U = nn.Conv2d(51,51,(1,1))
        self.conv_1x1_V = nn.Conv2d(51,51,(1,1))
        
    def forward(self,X):
        X = torch.relu(self.model(X))
        U = (self.conv_1x1_U(X))
        V = (self.conv_1x1_V(X))
        return U,V
    
def train(data_dir):
    
    epochs = 2000
    learning_rate = 0.001
    gamma = 0.9
    
    
    
    t = [transforms.ToPILImage(),transforms.Resize((224,224))]
    t = transforms.Compose(t)
    s = StandardScaler()
    data = Data(data_dir,transforms=t)
    data[0]

    #import pdb;pdb.set_trace()
    loader = torch.utils.data.DataLoader(data, batch_size = 16, shuffle = True, num_workers=4)
    
    
    
    model = Model(False).cuda()
    model.load_state_dict(torch.load('./segmentation_unet1/model_410.pth'))
    final_layer = Final_layer(model).cuda()
    
    optimizer = torch.optim.Adam(final_layer.parameters(),lr = learning_rate)
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=gamma)
    
    for epoch in range(epochs):
        u_acc = 0.
        v_acc = 0.
        count = 0
        u_sim = 0.
        v_sim = 0.
        train_loss = 0.
        for i,(X,U_actual,V_actual) in enumerate(loader):
            optimizer.zero_grad()
            
            
            X = X.cuda()
            U_actual = U_actual.cuda()
            V_actual = V_actual.cuda()
            U_pred,V_pred = final_layer(X)
            
            loss = loss_function(U_pred,U_actual.long(),V_pred,V_actual.long(),loss1,loss2)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # u_sim += ssim.ssim(U_pred,U_actual)
            # v_sim += ssim.ssim(V_pred,V_actual)
            #import pdb;pdb.set_trace()
            U_pred = torch.max(F.softmax(U_pred, dim=1), dim=1)[1]
            V_pred = torch.max(F.softmax(V_pred, dim=1), dim=1)[1]
            
            U_pred = U_pred.data.cpu().numpy()
            V_pred = V_pred.data.cpu().numpy()
            U_actual = U_actual.data.cpu().numpy()
            V_actual = V_actual.data.cpu().numpy()

            train_loss += loss.item()
            #print(loss.item())
            u_acc += accuracy(U_pred,U_actual)
            v_acc += accuracy(V_pred,V_actual)
            #import pdb;pdb.set_trace()
            count+=1
        #import pdb;pdb.set_trace()
        torch.save(final_layer.state_dict(),'{}/model_{}.pth'.format(out_dir,epoch))    
        print('Epoch : {}, Mean Accuracy(U) = {}, Mean Accuracy(V) = {}, Loss : {}'.format(epoch, u_acc/count, v_acc/count, train_loss/count))
        # print('Epoch : {}, Mean ssim score(U) = {}, Mean ssim score(V) = {}'.format(epoch, u_sim/count, v_sim/count))

        log_value('loss', train_loss/count, epoch)
        log_value('Acc/U', u_acc/count, epoch)
        log_value('Acc/V', v_acc/count, epoch)
        # log_value('ssim/U', u_sim/count, i)
        # log_value('ssim/V', v_sim/count, i)

if __name__ == '__main__':

    out_dir = "./model_harshit_class_1"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    configure(out_dir, flush_secs=5)

    train('/media/test_1k')