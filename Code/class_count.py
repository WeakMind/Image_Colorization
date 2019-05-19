import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import StandardScaler


import pytorch_ssim as ssim
import cv2
import os
import numpy as np  
import torchvision.models as models


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
        
        U = np.floor(U).astype(int)
        V = np.floor(V).astype(int)
        #Y_train = cv2.merge((U,V))
        
        
        return X_train,U,V
    
def train(data_dir):
    
    t = [transforms.ToPILImage(),transforms.Resize((224,224))]
    t = transforms.Compose(t)
    s = StandardScaler()
    data = Data(data_dir,transforms=t)
    u = []
    v = []

    #import pdb;pdb.set_trace()
    loader = torch.utils.data.DataLoader(data, batch_size = 8, shuffle = True, num_workers=4)
    for i,(X,U_actual,V_actual) in enumerate(loader):
        n = U_actual.numpy().flatten()
        #import pdb;pdb.set_trace()
        u+= list(n)
        n = V_actual.numpy().flatten()
        #import pdb;pdb.set_trace()
        v+= list(n)
        
    
    print(np.unique(u,return_counts= True))
    print(np.unique(v,return_counts=True))
        
if __name__ == '__main__':
    train('temp')