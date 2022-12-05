import numpy as np
import pandas as pd
import rasterio.features
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Sigmoid
from torch.utils.data import Dataset
import os
import rasterio
from SimpleCNN import Simple_CNN
import copy
import torch
from vit_pytorch import ViT
from CNN3D import *
from ViViT import *
from MTMSST import *
class train_model():
    import copy
    def eval(self,net, loader, criterion,batch_size,device):
        total = 0
        loss = 0
        with torch.no_grad(): 
            for data in loader:
                images, labels = data[0].to(device), data[1].to(device)         
                outputs = net(images)
                total += labels.size(0)
                loss += criterion(outputs, labels).item()
        return loss/(total/batch_size)

    def train(self,net, train_loader, val_loader, epoch, device, criterion, optimizer):
        best_model = None
        best_loss = 9999
        losses = []
        val_loss = []
        for e in range(epoch):
            l = 0
            for idx,data in enumerate(train_loader):
                images = data[0].to(device)
                labels = data[1].to(device)
                optimizer.zero_grad()
                predicts = net(images)
                loss = criterion(predicts, labels)
                loss.backward()
                optimizer.step()
                l += loss.item()
                # Show stats per 50 batch iter
                if idx % 50 == 49:
                    cur_loss = l/50
                    val_l = self.eval(net, val_loader, criterion,50,device)
                    losses.append(cur_loss)
                    val_loss.append(val_l)
                    if val_l<best_loss:
                        best_loss = val_l
                        best_model = copy.deepcopy(net)
                        print("[{:d}, {:d}] training_loss: {:.4f}, validation_loss: {:.4f}".format(e+1, idx+1, cur_loss, val_l))
                        l = 0

        return best_model,losses,val_loss

    def run(self,model,train_data,train_label,test_data,test_label,epoch=10,save_model=False,save_path=None,use_exist_model = False):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        criterion = torch.nn.MSELoss()
        batch_size = 50
        if model == "Simple_CNN":
            train_x = train_data[:,:1,:,:,:]
            train_x = np.squeeze(train_x)
            train_x = torch.Tensor(train_x)
            train_y = torch.Tensor(train_label)
            train = torch.utils.data.TensorDataset(train_x,train_y)
            test_x = test_data[:,:1,:,:,:]
            test_x = np.squeeze(test_x)
            test_x = torch.Tensor(test_x)
            test_y = torch.Tensor(test_label)
            test = torch.utils.data.TensorDataset(test_x,test_y)
            self.test = test
            train_size = 5311
            train, val= torch.utils.data.random_split(train, [train_size-500, 500])
            train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=True,drop_last=False)
            val_loader = torch.utils.data.DataLoader(dataset=val,batch_size=batch_size,shuffle=True,drop_last=False)
            test_loader= torch.utils.data.DataLoader(dataset=test,batch_size=batch_size,shuffle=True,drop_last=False)

            net = None
            if use_exist_model and save_path is not None:
                net = torch.load(save_path)
            else:
                net = Simple_CNN().to(device)
            optimizer =  torch.optim.Adam(net.parameters(),lr = 0.001)
            result = self.train(net,train_loader,test_loader,20,device,criterion,optimizer)
            if save_model:
                torch.save(result[0],save_path)
            return result

        elif model == "ViT":
            train_x = train_data[:,:1,:,:,:]
            train_x = np.squeeze(train_x)
            train_x = torch.Tensor(train_x)
            train_y = torch.Tensor(train_label)
            train = torch.utils.data.TensorDataset(train_x,train_y)
            test_x = test_data[:,:1,:,:,:]
            test_x = np.squeeze(test_x)
            test_x = torch.Tensor(test_x)
            test_y = torch.Tensor(test_label)
            test = torch.utils.data.TensorDataset(test_x,test_y)
            self.test = test
            train_size = 5311
            train, val= torch.utils.data.random_split(train, [train_size-500, 500])
            train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=True,drop_last=False)
            val_loader = torch.utils.data.DataLoader(dataset=val,batch_size=batch_size,shuffle=True,drop_last=False)
            test_loader= torch.utils.data.DataLoader(dataset=test,batch_size=batch_size,shuffle=True,drop_last=False)
            net = None
            if use_exist_model and save_path is not None:
                net = torch.load(save_path)
            else:
                net = ViT(
                    image_size = 21,
                    patch_size = 3,
                    num_classes = 1,
                    dim = 144,
                    channels = 10,
                    depth = 6,
                    heads = 16,
                    mlp_dim = 2048,
                    dropout = 0.1,
                    emb_dropout = 0.1
                    ).to(device)
            optimizer =  torch.optim.Adam(net.parameters(),lr = 0.001)
            result = self.train(net,train_loader,test_loader,20,device,criterion,optimizer)
            if save_model:
                torch.save(result[0],save_path)
            return result
            
        elif model == "ViViT":
            train_x = train_data
            train_x = torch.Tensor(train_x)
            train_y = torch.Tensor(train_label)
            train = torch.utils.data.TensorDataset(train_x,train_y)
            test_x = test_data
            test_x = torch.Tensor(test_x)
            test_y = torch.Tensor(test_label)
            test = torch.utils.data.TensorDataset(test_x,test_y) 
            self.test = test     
            train_size = 5311
            train, val= torch.utils.data.random_split(train, [train_size-500, 500])
            train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=True,drop_last=False)
            val_loader = torch.utils.data.DataLoader(dataset=val,batch_size=batch_size,shuffle=True,drop_last=False)
            test_loader= torch.utils.data.DataLoader(dataset=test,batch_size=batch_size,shuffle=True,drop_last=False)
            net = None
            if use_exist_model and save_path is not None:
                net = torch.load(save_path)
            else:
                net = Standard_ViViT(
                    image_size = 21,
                    patch_size = 3,
                    num_classes = 1,
                    num_frames = 10,
                    in_channels = 10
                ).to("cuda")
            optimizer =  torch.optim.Adam(net.parameters(),lr = 0.001)
            result = self.train(net,train_loader,test_loader,1,device,criterion,optimizer)
            return result
        elif model == "3DCNN":
            train_x = np.swapaxes(train_data,1,2)
            train_x = torch.Tensor(train_x)
            train_y = torch.Tensor(train_label)
            train = torch.utils.data.TensorDataset(train_x,train_y)
            test_x = np.swapaxes(test_data,1,2)
            test_x = torch.Tensor(test_x)
            test_y = torch.Tensor(test_label)
            test = torch.utils.data.TensorDataset(test_x,test_y)
            self.test = test
            train_size = 5311
            train, val= torch.utils.data.random_split(train, [train_size-500, 500])
            train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=True,drop_last=False)
            val_loader = torch.utils.data.DataLoader(dataset=val,batch_size=batch_size,shuffle=True,drop_last=False)
            test_loader= torch.utils.data.DataLoader(dataset=test,batch_size=batch_size,shuffle=True,drop_last=False)
            net = None
            if use_exist_model and save_path is not None:
                net = torch.load(save_path)
            else:
                net = resnet50(sample_size=21,sample_duration=1, num_classes=1).to("cuda")
            optimizer =  torch.optim.Adam(net.parameters(),lr = 0.001)
            result = self.train(net,train_loader,test_loader,1,device,criterion,optimizer)
            if save_model:
                torch.save(result[0],save_path)
            return result
        elif model == "MTMSST":
            train_x = train_data
            train_x = torch.Tensor(train_x)
            train_y = torch.Tensor(train_label)
            train = torch.utils.data.TensorDataset(train_x,train_y)
            test_x = test_data
            test_x = torch.Tensor(test_x)
            test_y = torch.Tensor(test_label)
            test = torch.utils.data.TensorDataset(test_x,test_y)  
            self.test = test    
            train_size = 5311
            train, val= torch.utils.data.random_split(train, [train_size-500, 500])
            train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=True,drop_last=False)
            val_loader = torch.utils.data.DataLoader(dataset=val,batch_size=batch_size,shuffle=True,drop_last=False)
            test_loader= torch.utils.data.DataLoader(dataset=test,batch_size=batch_size,shuffle=True,drop_last=False)
            net = None
            if use_exist_model and save_path is not None:
                net = torch.load(save_path)
            else:
                net = ViViT(
                    image_size = 21,
                    patch_size = 3,
                    num_classes = 1,
                    num_frames = 10,
                    in_channels = 10
                ).to("cuda")
            optimizer =  torch.optim.Adam(net.parameters(),lr = 0.001)
            result = self.train(net,train_loader,test_loader,1,device,criterion,optimizer)
            return result
            
        else:
            print("Model Not included")
            return

class result_visualize():
    def __init__(self,model,data,use_existed_model=False,model_path=None):
        self.net = None
        if use_existed_model:
            self.net = torch.load(model_path)
        else:
            self.net = model
        self.loader = torch.utils.data.DataLoader(dataset=data,batch_size=1,shuffle=False,drop_last=False)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def RMSE_eval(self):
        self.net.eval()
        true_yield = []
        predicted_yield = []
        total = 0
        sum_sqr_error = 0
        sum_percentage_error = 0
        sum = 0
        with torch.no_grad(): 
            for data in self.loader:
                images, labels = data[0].to(self.device), data[1].to(self.device) 
                outputs = self.net(images)
                total += labels.size(0)
                for i in range(labels.size(0)):
                    true_yield.append(labels[i][0].cpu().detach())
                    predicted_yield.append(outputs[i][0].cpu().detach())
                    sum_sqr_error += (labels[i][0]-outputs[i][0])**2 
                    sum_percentage_error += abs(labels[i][0]-outputs[i][0]) /labels[i][0]
                    sum+=labels[i][0]
        avg_yield = sum/total
        sst = 0
        for i in range(len(true_yield)):
           sst += (true_yield[i]-avg_yield)**2
        print(sum_sqr_error/sst)
        return math.sqrt(sum_sqr_error/total),sum_percentage_error/total,true_yield,predicted_yield,1-sum_sqr_error/sst

    def show_metrics(self):
        rmse,mape,true_yield,predicted_yield,r_sq = self.RMSE_eval()
        print("Calculated RMSE is: " , rmse)
        print("Calculated MAPE is: " , float(mape))
        print("Calculated R squared is: " , float(r_sq))


    def show_scatter(self):
        rmse,mape,true_yield,predicted_yield,r_sq = self.RMSE_eval()
        import mpl_scatter_density # adds projection='scatter_density'
        from matplotlib.colors import LinearSegmentedColormap

        # "Viridis-like" colormap with white background
        white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
            (0, '#ffffff'),
            (1e-20, '#440053'),
            (0.2, '#404388'),
            (0.4, '#2a788e'),
            (0.6, '#21a784'),
            (0.8, '#78d151'),
            (1, '#fde624'),
        ], N=256)

        def using_mpl_scatter_density(fig, x, y):
            ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
            
            density = ax.scatter_density(x, y,dpi=15, cmap=white_viridis)
            fig.colorbar(density, label='Number of points per pixel')

        fig = plt.figure()

        using_mpl_scatter_density(fig, true_yield, predicted_yield)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.plot([0, 1], [0, 1], color='k', linestyle='-', linewidth=2)
        plt.show()

    def Visualize(self,net, coords):
        predicted_yield = []
        pic = np.zeros((20,20))
        with torch.no_grad(): 
            for data in self.loader:
                images, labels = data[0].to(self.device), data[1].to(self.device) 
                outputs = net(images)
                for i in range(labels.size(0)):
                    predicted_yield.append(outputs[i][0].cpu().detach()-labels[i][0].cpu().detach())
        for y in range(len(predicted_yield)):
            i = (coords[y][0]+coords[y][1])//2
            j = (coords[y][2]+coords[y][3])//2
            pic[i-111][j-71] = predicted_yield[y]
        print(len(predicted_yield))
        plt.imshow(pic, cmap='bwr_r')
        plt.clim(-0.4,0.4)
        plt.colorbar() 