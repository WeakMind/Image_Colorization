# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:06:30 2019

@author: h.oberoi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 22:14:37 2019

@author: Harshit
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from skimage.color import rgb2lab, rgb2gray, lab2rgb
import torch.nn.functional as F
import numpy as np
import os
import torchvision.models as models
from tensorboard_logger import configure, log_value





class Data(torchvision.datasets.ImageFolder):
    
    def __getitem__(self,index):
        path , target = self.imgs[index]
        img = self.loader(path)
        img = self.transform(img)
        R,G,B = img[0,:,:],img[1,:,:],img[2,:,:]
        img  = 0.2125*R + 0.7154*G + 0.0721*B
        img = (img.unsqueeze_(0))
        return (img,target)
    

     
def train(train_dir,val_dir,out_dir):
    
    epochs = 100
    learning_rate = 0.0001
    gamma = 0.9
    
    t = [transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    t = transforms.Compose(t)
    data = Data(train_dir,transform=t)
    loader = torch.utils.data.DataLoader(data, batch_size = 32, shuffle = True, num_workers=4)
    
    val_data = Data(val_dir,transform=t)
    val_loader = torch.utils.data.DataLoader(val_data,batch_size = 32,shuffle=False,num_workers=4)
    
    
    model = models.vgg16_bn(pretrained=True).cuda()
   # model.load_state_dict(torch.load('D:/vgg16_bn-6c64b313.pth'))
    model.classifier[6] = nn.Linear(4096,257).cuda()
    model.features[0].weight = nn.Parameter(model.features[0].weight.sum(1).unsqueeze_(1)).cuda()
    
    
    
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    loss1 = nn.CrossEntropyLoss().cuda()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=gamma)
    
    
    max_acc = 0
    for epoch in range(epochs):
        count = 0
        train_loss = 0.
        lr_scheduler.step()
        for i,(X,Y) in enumerate(loader):
            correct=0
            X = X.cuda()
            Y = Y.cuda()
            Y_pred = model(X)
            optimizer.zero_grad()
            loss = loss1(Y_pred,Y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            count+=1
            
        correct = 0
        total = 0
        for i,(X,Y) in enumerate(val_loader):
            X = X.cuda()
            Y = Y.cuda()
            Y_pred = model(X)
            Y_pred = torch.max(F.softmax(Y_pred,dim=1),1)[1]
            correct+= (Y_pred==Y).sum()
            total+=Y.shape[0]
        
        accuracy = (correct*100)/total
        #if accuracy > max_acc:
        torch.save(model.state_dict(),'{}/vgg16bn_model_{}.pth'.format(out_dir,epoch))
        #max_acc = accuracy
        print('Accuracy : {}'.format(accuracy))    
        print('Epoch : {}, Loss : {}'.format(epoch, train_loss/count))
        log_value('loss', train_loss/count, epoch)
        log_value('Val Accuracy',accuracy,epoch)
        
        

if __name__ == '__main__':

    out_dir = "./vgg16bn_classification"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    configure(out_dir, flush_secs=5)
    train_dir = '/media/256_ObjectCategories'
    val_dir = '/media/val_256_ObjectCategories'
    train(train_dir,val_dir,out_dir)
    