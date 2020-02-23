# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from utils import *
from torchvision import transforms, utils
import pickle
import sklearn.metrics as metrics
import json
import logging
from torchvision import models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

######################################
# build DNN
######################################

class EGDNN(nn.Module):
# 'EGNet' means Earphone Gesture Net. What a straightforward name.
    def getname(self):
        return "EGDNN"

    def __init__(self):
        super(EGDNN, self).__init__()
        input_size = 20
        kernel_size = 3

        input_size = int(input_size)
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 300)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        self.fc3 = nn.Linear(300, 50)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)

        self.fc4 = nn.Linear(50, 2)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)
        # out = self.maxpool2(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.relu(out)
        # out = self.maxpool3(out)
        out = self.dropout(out)

        out = self.fc4(out)
        return out


######################################
# build CNN
######################################

class EGCNN(nn.Module):
# 'EGNet' means Earphone Gesture Net. What a straightforward name.
    def getname(self):
        return "EGCNN"

    def __init__(self):
        super(EGCNN, self).__init__()
        input_size = 64
        kernel_size = 3

        padding = 1
        output_channels1 = 8
        self.conv1 = nn.Conv2d(in_channels=1, 
                                out_channels=output_channels1, 
                                kernel_size=kernel_size,
                                padding=padding) # 64 * 64
        input_size = input_size + 2 * padding - kernel_size + 1

        self.batchnorm1 = nn.BatchNorm2d(output_channels1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) # 32 * 32
        input_size /= 2

        output_channels2 = 16
        self.conv2 = nn.Conv2d(in_channels=output_channels1, 
                                out_channels=output_channels2, 
                                kernel_size=kernel_size,
                                padding=padding) # 32 * 32
        input_size = input_size + 2 * padding - kernel_size + 1

        self.batchnorm2 = nn.BatchNorm2d(output_channels2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) # 16 * 16
        input_size /= 2

        # output_channels3 = 64
        # self.conv3 = nn.Conv2d(in_channels=output_channels2, 
        #                         out_channels=output_channels3, 
        #                         kernel_size=kernel_size,
        #                         padding=padding) # 16 * 16
        # input_size = input_size + 2 * padding - kernel_size + 1

        # self.batchnorm3 = nn.BatchNorm2d(output_channels3)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2) # 8 * 8
        # input_size /= 2

        input_size = int(input_size)
        self.fc1 = nn.Linear(input_size * input_size * output_channels2, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        out = self.dropout(out)

        # out = self.conv3(out)
        # out = self.batchnorm3(out)
        # out = self.relu(out)
        # out = self.maxpool3(out)
        
        out = out.view(-1, self.num_flat_features(out))

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        # out = self.sigmoid(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

######################################
# Transfer Learning CNN
######################################

class EGCNN_transfer():

    def __init__(self, model_name = "vgg", num_classes = 8, feature_extract=True, use_pretrained=True, extra_training = False):
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        self.use_pretrained = use_pretrained
        self.extra_training = extra_training

    def getname(self):
        return self.name

    def get_simple_sequential(self, dim_in, num_classes):
        hidden = 200
        return nn.Sequential(
                  nn.Linear(dim_in, hidden),
                  nn.ReLU(),
                  nn.Dropout(p = 0.5),
                  nn.Linear(hidden, num_classes),
            )

    def getmodel(self):
        self.name = 'model_from_' + self.model_name 

        if "vgg" in self.model_name:
            model_ft = models.vgg16_bn(pretrained=self.use_pretrained)
            set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features

            model_ft.classifier[6] = self.get_simple_sequential(num_ftrs, self.num_classes)
            input_size = 224

        elif "densenet" in self.model_name:
            model_ft = models.densenet121(pretrained=self.use_pretrained)
            set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = self.get_simple_sequential(num_ftrs, self.num_classes)
            input_size = 224
            if (self.extra_training):
                model_ft.load_state_dict(torch.load("model_from_" + self.model_name[:-8] + "_mark1.pt"))

        elif "inception" in self.model_name:
            model_ft = models.inception_v3(pretrained=self.use_pretrained)
            set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = self.get_simple_sequential(num_ftrs, self.num_classes)
            input_size = 299
        elif "wide_resnet" in self.model_name:
            model_ft = models.wide_resnet50_2(pretrained=self.use_pretrained)
            set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = self.get_simple_sequential(num_ftrs, self.num_classes)
            input_size = 224
        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft

######################################
# build CNN-autoencoder-decoder
######################################

class EGCNNAutoEncoder(nn.Module):
    def getname(self):
        return "EGCNNAutoEncoder"
        
    def __init__(self):
        super(EGCNNAutoEncoder, self).__init__()
        input_size = 64
        kernel_size = 3

        padding = 1
        stride = 1
        output_channels1 = 8
        self.conv1 = nn.Conv2d(in_channels=1, 
                                out_channels=output_channels1, 
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride) # 64 * 64
        input_size = input_size + 2 * padding - kernel_size + 1

        self.batchnorm1 = nn.BatchNorm2d(output_channels1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) # 32 * 32
        input_size /= 2

        output_channels2 = 16
        self.conv2 = nn.Conv2d(in_channels=output_channels1, 
                                out_channels=output_channels2, 
                                kernel_size=kernel_size,
                                padding=padding) # 32 * 32
        input_size = input_size + 2 * padding - kernel_size + 1

        self.batchnorm2 = nn.BatchNorm2d(output_channels2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) # 16 * 16
        input_size /= 2

        output_channels3 = 4
        self.conv3 = nn.Conv2d(in_channels=output_channels2, 
                                out_channels=output_channels3, 
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride) # 16 * 16
        input_size = input_size + 2 * padding - kernel_size + 1

        self.batchnorm3 = nn.BatchNorm2d(output_channels3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) # 8 * 8
        input_size /= 2

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.batchnorm3(out)
        out = self.relu(out)
        out = self.maxpool3(out)
        out = self.dropout(out)
        return out


class EGCNNAutoDecoder(nn.Module):
    def getname(self):
        return "EGCNNAutoDecoder"
    def __init__(self):
        super(EGCNNAutoDecoder, self).__init__()
        input_size = 8
        kernel_size = 4
        padding = 1
        stride = 2

        output_channels3 = 4
        output_channels2 = 16
        output_channels1 = 8

        self.conv3 = nn.ConvTranspose2d(in_channels=output_channels3, 
                                out_channels=output_channels2, 
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride) # 16 * 16
        input_size = input_size - 2 * padding + kernel_size - 1

        self.batchnorm3 = nn.BatchNorm2d(output_channels1)
        self.maxpool3 = nn.MaxUnpool2d(kernel_size=2) # 8 * 8
        input_size *= 2

        self.conv2 = nn.ConvTranspose2d(in_channels=output_channels2, 
                                out_channels=output_channels1, 
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride) # 32 * 32
        input_size = input_size - 2 * padding + kernel_size - 1
        self.batchnorm2 = nn.BatchNorm2d(output_channels1)
        self.maxpool2 = nn.MaxUnpool2d(kernel_size=2) # 16 * 16
        input_size *= 2

        self.conv1 = nn.ConvTranspose2d(in_channels=output_channels1, 
                                out_channels=1, 
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride) # 64 * 64
        input_size = input_size - 2 * padding + kernel_size - 1
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.maxpool1 = nn.MaxUnpool2d(kernel_size=2) # 32 * 32
        input_size *= 2

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv3(x)
        # out = self.batchnorm3(out)
        out = self.relu(out)
        # out = self.maxpool3(out)
        # out = self.dropout(out)

        out = self.conv2(out)
        # out = self.batchnorm2(out)
        out = self.relu(out)
        # # out = self.maxpool2(out)
        # out = self.dropout(out)

        out = self.conv1(out)
        # out = self.batchnorm1(out)
        out = self.sigmoid(out)
        # out = self.maxpool1(out)
        # out = self.dropout(out)
        return out


class EGFCClassifier(nn.Module):
    def getname(self):
        return "EGFCClassifier"
    def __init__(self):
        super(EGFCClassifier, self).__init__()

        self.output_size = 10

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        # self.fc1 = nn.Linear(256, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, self.output_size)

    def forward(self, x):
        l = self.num_flat_features(x)
        out = x.view(-1, l)

        # out = self.fc1(out)
        out = nn.Linear(l, 500)(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


######################################
# build RNN
######################################

class EGRNN(nn.Module):
# 'EGNet' means Earphone Gesture Net. What a straightforward name.
    def getname(self):
        return "EGRNN"

    def __init__(self):
        super(EGRNN, self).__init__()

        self.input_dim = 64
        self.hidden_dim = 200
        self.layer_dim = 2
        self.hiddenoutput_dim = 50
        self.output_dim = 10

        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim,
            num_layers=self.layer_dim,
            batch_first=True,
            dropout = 0.5,
            bidirectional=True)
        self.hidden2out = nn.Linear(self.hidden_dim*self.layer_dim*2, self.hiddenoutput_dim)
        self.fc = nn.Linear(self.hiddenoutput_dim, self.output_dim)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=torch.device(self.device)).requires_grad_()
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=torch.device(self.device)).requires_grad_()

        out, (hn, cn) = self.rnn(x, (h0.detach(), c0.detach()))
        # out, (hn, cn) = self.rnn(x)

        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.hidden2out(torch.cat([cn[i,:, :] for i in range(cn.shape[0])], dim=1))
        out = self.fc(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



