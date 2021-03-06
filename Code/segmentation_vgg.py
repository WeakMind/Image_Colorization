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
from skimage.color import rgb2lab, rgb2gray, lab2rgb
import matplotlib.pyplot as plt
from skimage import io

import pytorch_ssim as ssim
import cv2
import os
import numpy as np  
import torchvision.models as models
from tensorboard_logger import configure, log_value

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ColorizationNet(nn.Module):
    def __init__(self, midlevel_input_size=128, global_input_size=512):
        super(ColorizationNet, self).__init__()
        # Fusion layer to combine midlevel and global features
        self.midlevel_input_size = midlevel_input_size
        self.global_input_size = global_input_size
        self.fusion = nn.Linear(midlevel_input_size + global_input_size, midlevel_input_size)
        self.bn1 = nn.BatchNorm1d(midlevel_input_size)

        # Convolutional layers and upsampling
        self.deconv1 = nn.ConvTranspose2d(midlevel_input_size, 128, kernel_size=4, stride=2, padding=1)
        self.bn_deconv = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(midlevel_input_size, 128, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(midlevel_input_size, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)

        print('Loaded colorization net.')

    def forward(self, midlevel_input): #, global_input):
        
        # Convolutional layers and upsampling
        x = F.sigmoid(self.bn2(self.conv1(midlevel_input)))
        x = self.bn_deconv(self.deconv1(x))
        x = F.sigmoid(self.bn3(self.conv2(x)))
        x = self.bn4(F.relu(self.conv3(x)))
        x = self.upsample(x)
        x = F.sigmoid(self.conv4(x))
        x = F.sigmoid(self.deconv3((self.conv5(x))))
        return x


class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        
        # Build ResNet and change first conv layer to accept single-channel input
        resnet_gray_model = models.resnet18(num_classes=365)
        resnet_gray_model.conv1.weight = nn.Parameter(resnet_gray_model.conv1.weight.sum(dim=1).unsqueeze(1).data)
        
        # Only needed if not resuming from a checkpoint: load pretrained ResNet-gray model
        if torch.cuda.is_available(): # and only if gpu is available
            resnet_gray_weights = torch.load('Automatic-Image-Colorization/pretrained/resnet_gray_weights.pth.tar') #torch.load('pretrained/resnet_gray.tar')['state_dict']
            resnet_gray_model.load_state_dict(resnet_gray_weights)
            print('Pretrained ResNet-gray weights loaded')

        # Extract midlevel and global features from ResNet-gray
        self.midlevel_resnet = nn.Sequential(*list(resnet_gray_model.children())[0:6])
        self.global_resnet = nn.Sequential(*list(resnet_gray_model.children())[0:9])
        self.fusion_and_colorization_net = ColorizationNet()

    def forward(self, input_image):

        # Pass input through ResNet-gray to extract features
        midlevel_output = self.midlevel_resnet(input_image)
        global_output = self.global_resnet(input_image)
        # Combine features in fusion layer and upsample
        output = self.fusion_and_colorization_net(midlevel_output) #, global_output)
        #import pdb;pdb.set_trace()
        return output




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
    def __init__(self):
        super(Model,self).__init__()
        
        self.model = models.vgg16_bn(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096,257)
        self.model.features[0].weight = nn.Parameter(self.model.features[0].weight.sum(dim=1).unsqueeze_(1))
        self.model.load_state_dict(torch.load('./vgg16bn_classification/vgg16bn_model_97.pth'))
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
        return (out)
    
class Data(Dataset):
    def __init__(self,X_dir,Y_dir,transforms):
        self.X_dir = X_dir
        self.Y_dir = Y_dir
        #self.transform = transform
        self.images_X = list(os.listdir(X_dir))
        self.images_Y = list(os.listdir(Y_dir))
        self.names = []
        self.transform = transforms   
          
        
    def __len__(self):
        return len(self.images_Y)

    def __getitem__(self,index):
        name,formatt  = self.images_Y[index].split(".")
        img_Y = io.imread(os.path.join(self.Y_dir,self.images_Y[index]))
        img_X = io.imread(os.path.join(self.X_dir,'{}.jpg'.format(name)))
        img_original = self.transform(img_X)
        img_original = np.asarray(img_original)
        img_original = rgb2gray(img_original)
        img_seg = self.transform(img_Y)
        img_seg = np.asarray(img_seg)
        #img_gray = rgb2gray(img_seg)
        img_gray = img_seg[:,:,0]*0.2125 + img_seg[:,:,1]*0.7154 + img_seg[:,:,2]*0.0721
        
        #img_original = np.transpose(img_original,(2,0,1))
        img_original = torch.from_numpy(img_original).unsqueeze_(0).float()
        img_gray = torch.from_numpy(img_gray).unsqueeze_(0).float()
        
        return (img_original,img_gray,os.path.join(self.Y_dir,self.images_Y[index]))


    
def loss_function(Y_pred,Y,loss,gray2class):
    b,c,_,_ = Y_pred.shape
    #import pdb;pdb.set_trace()
    Y_pred = Y_pred.contiguous().view(b,c,-1).permute(0,2,1).contiguous().view(-1,c)
    Y = Y.contiguous().view(b,-1).contiguous().view(-1,1).squeeze_()
    Y = Y.cpu().numpy()
    temp = np.zeros_like(Y)
    '''#import pdb;pdb.set_trace()
    for i,val in enumerate(Y):
        temp[i] = gray2class[val]'''
    temp = [gray2class[val] for val in Y]
    Y = np.asarray(temp)
    Y = torch.from_numpy(Y).long().cuda()
    batch_loss = loss(Y_pred,Y)
    return batch_loss


'''def visualize(model, epoch, data_dir):
    t = [transforms.ToPILImage(),transforms.Resize((224,224))]
    t = transforms.Compose(t)
    data = Data(data_dir,transforms=t)

    for i in range(5):
        X, Y, orig_im = data[i]
        orig_im = cv2.imread(orig_im)
        X = X.unsqueeze(0).cuda()
        ab = model(X)
        X = X[0,:,:,:]
        ab = ab[0,:,:,:]
        
        
        #import pdb;pdb.set_trace()
        color_image = torch.cat((X, ab), 0).cpu().detach().numpy()
        color_image = color_image.transpose((1, 2, 0))  
        color_image[:, :, 0:1] = np.clip(color_image[:, :, 0:1] * 100,0,100)
        color_image[:, :, 1:3 ] = np.clip(color_image[:, :, 1:3] * 255 - 128,-126,126)   
        color_image = lab2rgb(color_image.astype(np.float64))
        
        
        
        if not os.path.isdir("Sanity"):
            os.makedirs("Sanity")
        orig_im = cv2.resize(orig_im, (224,224))/255
        color_image = np.clip(color_image,0,1)
        orig_im = np.clip(orig_im,0,1)
        plt.imsave("Sanity/{}_{}.jpg".format(epoch, i), np.concatenate((color_image, orig_im[:,:,::-1]), axis=1))'''
        #import pdb;pdb.set_trace()
     
def train(X_dir,Y_dir,out_dir):
    
    epochs = 2000
    learning_rate = 0.0005
    gamma = 0.9
    
    t = [transforms.ToPILImage(),transforms.CenterCrop((224,224))]
    t = transforms.Compose(t)
    data = Data(X_dir,Y_dir,transforms=t)
    print(data.__len__())
    #data2 = Data2(X_dir,Y_dir,transforms=t)
    loader = torch.utils.data.DataLoader(data, batch_size = 16, shuffle = True, num_workers=4)
    pre_loader = torch.utils.data.DataLoader(data,batch_size = 16,shuffle=False,num_workers=4)
    
    model = Model().cuda()
    
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    #loss1 = nn.SmoothL1Loss()
    loss1 = nn.CrossEntropyLoss().cuda()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=gamma)
    
    
    gray2class = {}
    class_index = 0
    for i,(X,Y,path) in enumerate(pre_loader):
        Y = Y.contiguous().view(-1)
        values = np.unique(Y)
        for val in values:
            if int(val) not in gray2class.keys():
                gray2class[int(val)] = class_index
                class_index+=1
    #import pdb;pdb.set_trace()
    import time
    for epoch in range(epochs):
        count = 0
        train_loss = 0.
        lr_scheduler.step()
        for i,(X,Y,path) in enumerate(loader):
            # start = time.time()
            
            X = X.cuda()
            Y = Y.long().cuda()
            #import pdb;pdb.set_trace()
            #import pdb;pdb.set_trace()
            X = model(X)
            
            optimizer.zero_grad()
            
            loss = loss_function(X,Y,loss1,gray2class)
            loss.backward()
            optimizer.step()
            #print(loss.item())
            train_loss += loss.item()
            count+=1

            end = time.time()
            # print(end-start)
        #import pdb;pdb.set_trace()
        torch.save(model.state_dict(),'{}/model_{}.pth'.format(out_dir,epoch))    
        print('Epoch : {}, Loss : {}'.format(epoch, train_loss/count))
        #visualize(model, epoch, data_dir)
        # print('Epoch : {}, Mean ssim score(U) = {}, Mean ssim score(V) = {}'.format(epoch, u_sim/count, v_sim/count))
        log_value('loss', train_loss/count, epoch)
        
        # log_value('ssim/U', u_sim/count, i)
        # log_value('ssim/V', v_sim/count, i)'''

if __name__ == '__main__':

    out_dir = "./segmentation_unet1"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    configure(out_dir, flush_secs=5)
    X_dir = '/media/VOC2012/JPEGImages/' 
    Y_dir = '/media/VOC2012/SegmentationClass/'
    train(X_dir,Y_dir,out_dir)
