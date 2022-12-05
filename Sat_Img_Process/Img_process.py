import os
import rasterio
import numpy as np
import pandas as pd
import rasterio.features
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Sigmoid
from torch.utils.data import Dataset
import os
import rasterio

class img_process():
    def __init__(self, img_path , yield_tif_path):

        dirs = os.listdir(img_path)
        img_paths = []
        for file in dirs:
            img_paths.append(img_path+file)
        self.img_paths = img_paths
        self.img_paths.sort()
        def pad_with(vector, pad_width, iaxis, kwargs):
            pad_value = kwargs.get('padder', 10)
            vector[:pad_width[0]] = pad_value
            vector[-pad_width[1]:] = pad_value

        sample_S2 = []

        for i in range(len(img_paths)):
            img = []
            with rasterio.open(img_paths[i]) as dataset:
                mask = dataset.dataset_mask()
                for geom,val in rasterio.features.shapes(mask,transform = dataset.transform):
                    geom = rasterio.warp.transform_geom(
                        dataset.crs,'EPSG:4326',geom,precision=12
                    )
                for i in range(1,11):
                    layer= dataset.read(i)
                    layer = np.nan_to_num(layer,0)
                    img.append(np.pad(layer, 10, pad_with, padder=0))

            sample_S2.append(img)  
        sample_S2 = np.array(sample_S2)
        self.sample_S2 = sample_S2

        
        field_mask = np.zeros((147,112))

        with rasterio.open(yield_tif_path) as dataset:
            true_field = dataset.read(1)
            
            min_y  = np.nanmin(true_field)
            for i in range(len(true_field)):
                for j in range(len(true_field[i])):
                    if true_field[i][j] == 1.7976931348623157e+308:
                        true_field[i][j] = 0
                    else:
                        field_mask[i][j] =1
            max_y = np.nanmax(true_field)

  
        for i in range(len(true_field)):
            for j in range(len(true_field[i])):
                if field_mask[i][j]==1:
                    true_field[i][j] = (true_field[i][j]-min_y)/(max_y-min_y)

        train_label = []
        pic_slice = []
        test_label =[]
        test_slice = []
        for i in range(len(true_field)):
            for j in range(len(true_field[i])):
                if ((i<=100 or i>120) or (j<=40 or j >60)):
                    if field_mask[i][j]!=0 and ((i<=80 or i>140) or (j<=20 or j >80)):
                        train_label.append(true_field[i][j])
                        pic_slice.append([i,i+21,j,j+21])
                else:
                    test_label.append(true_field[i][j])
                    test_slice.append([i,i+21,j,j+21])

        self.true_field = true_field

        def generate_data(pic_slice,sample_S2,sample_label):
            train_data_S2 = []
            for i in range(len(pic_slice)):
                S2 = []
                for t in range(len(sample_S2)):
                    temporal = []
                    for c in range(len(sample_S2[t])):
                        temporal.append(sample_S2[t][c][pic_slice[i][0]:pic_slice[i][1],pic_slice[i][2]:pic_slice[i][3]])
                    S2.append(temporal)
                train_data_S2.append(S2)
            train_data_S2 = np.array(train_data_S2)
            sample_label = np.array(sample_label)
            sample_label = sample_label.reshape(sample_label.shape[0],-1)

            return train_data_S2, sample_label
        
        train_S2,train_label = generate_data(pic_slice,sample_S2,train_label)
        
        test_S2,test_label = generate_data(test_slice,sample_S2,test_label)
        
        self.pic_slice = pic_slice
        self.test_slice = test_slice
        self.train_S2,self.train_label = train_S2,train_label
        self.test_S2,self.test_label = test_S2,test_label

        



    def show_paths(self):
        print(self.img_paths)

    def sat_visualize(self,i_img,j_channel):
        plt.imshow(self.sample_S2[i_img][j_channel],cmap="gray")

    def yield_visualize(self):
        plt.imshow(self.true_field)
        plt.colorbar()

    def show_train_test_shape(self):
        print("Train data shape: "+str(self.train_S2.shape))
        print("Train label shape: "+str(self.train_label.shape))
        print("Test data shape: "+str(self.test_S2.shape))
        print("Test label shape: "+str(self.test_label.shape))


