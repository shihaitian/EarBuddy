# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
try:
    from torchaudio import transforms
except:
    pass
from torch.autograd import Variable
from utils import *
import pickle
import json
from torch.utils.data.dataset import Dataset
import random
from copy import deepcopy
import scipy
import librosa

class AudioDatasetTrain(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, pickle_file_prefix, device = "cpu", rawinput = False,
        leaveoneuserout = False, userids = [],
        augment = False, input_size = 64, channels = 1):
        """
        Args:
            pickle_file (string): Path to the pickle file with data and labels
        """

        # Some pre-defined model
        self.window_time = 1.2

        if (leaveoneuserout):
            count = 0
            if (len(userids) != 1):
                print("invalid input")
                return
            # for userid in userids:
            #     pickle_file = pickle_file_prefix + "_" + str(userid) + ".pkl"
            #     with open(pickle_file, "rb") as f:
            #         data_ = pickle.load(f)
            #     if (count == 0):
            #         self.data = deepcopy(data_)
            #     else:
            #         self.data["X"] = np.append(self.data["X"], deepcopy(data_["X"]), axis = 0)
            #         self.data["Y"] = np.append(self.data["Y"], deepcopy(data_["Y"]), axis = 0)
            #     count += 1
            print(pickle_file_prefix + "_" + str(userids[0]) + ".pkl")
            with open(pickle_file_prefix + "_" + str(userids[0]) + ".pkl", "rb") as f:
                self.data = pickle.load(f)
        else:
            with open(pickle_file_prefix + ".pkl", "rb") as f:
                self.data = pickle.load(f)
        with open("../GESTURE_CONFIG.json", "r") as f:
            self.gesture_config = json.load(f)
            self.sr = self.gesture_config["samplerate"]
        
        self.rawinput = rawinput
        self.augment = augment
        self.input_size = input_size
        self.channels = channels
        self.street_noise = None
        self.office_noise = None
        self.device = device

        self.street_noise, _ = librosa.load("./user_data/processed/street_noise.wav",sr = self.sr)
        # self.street_noise = np.expand_dims(self.street_noise, axis = 1)
        self.street_noise = torch.from_numpy(self.street_noise).float()
        self.office_noise, _ = librosa.load("./user_data/processed/office_noise.wav",sr = self.sr)
        # self.office_noise = np.expand_dims(self.office_noise, axis = 1)
        self.office_noise = torch.from_numpy(self.office_noise).float()

        # self.transform = transforms.AmplitudeToDB(transforms.MelSpectrogram(sample_rate = self.sr,
        #                                             n_fft = 2048,
        #                                             hop_length = int(np.floor(1.2 * self.sr / self.input_size)),
        #                                             f_max = 735,
        #                                             n_mels = self.input_size
        #                                             ))
        self.transform = raw_audio_to_logmelspec(self.sr, self.input_size)

        self.X = []
        self.Y = []
        for x, i in zip(self.data["X"], self.data["Y"]):
            if (i[0] in self.gesture_config["train_labels_no_noise"]):
                idx = self.gesture_config["train_labels_no_noise"].index(i[0])
                self.Y.append(deepcopy(torch.tensor(idx, dtype = torch.long)))
                self.X.append(deepcopy(x))
        self.Y = np.array(self.Y)
        if (not self.rawinput): # if not the raw input, the input will be the processed 64 * 64 mel spec image
            self.X = np.array([scipy.ndimage.zoom(y, self.input_size / 64, order=0) for y in self.X])
            self.X = np.expand_dims(self.X, axis = 1)
            # if (channels > 1):
            #     self.X = np.repeat(self.X, repeats = channels, axis = 1)
        else:
            self.X = np.expand_dims(self.X, axis = 1)

        self.X = torch.from_numpy(self.X).float()
        self.Y = torch.from_numpy(self.Y)

        print(self.X.shape, self.Y.shape)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]

        if (self.rawinput):
            if (self.augment):
                rand = random.random()
                if (rand < 0.5):
                    rand2 = random.random()
                    if (rand2 < 0.5):
                        X = add_noise(X, self.street_noise, self.sr)
                    else:
                        X = add_noise(X, self.office_noise, self.sr)
            # X = raw_audio_to_logmelspec(X.numpy(), self.sr, self.input_size)
            # X = torch.from_numpy(X).float()
            X = self.transform(X)[:,:self.input_size,:self.input_size]
            # if (self.channels > 1):
            #     X = np.repeat(X, repeats = self.channels, axis = 0)
        
        # X = np.array(X)
        if (self.augment):
            rand = random.random()
            if (rand < 0.5):
                X = flip_hori(X)
            rand = random.random()
            if (rand < 0.5):
                X = freq_mask(X,F = self.input_size)
            rand = random.random()
            if (rand < 0.5):
                X = time_mask(X,T = self.input_size)
        if (self.channels > 1):
            X = X.repeat([self.channels,1,1])

        return X, Y

class AudioDatasetTest(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, pickle_file_prefix, device = "cpu", rawinput = False,
        leaveoneuserout = False, userids = [],
        augment = False, input_size = 64, channels = 1):
        """
        Args:
            pickle_file (string): Path to the pickle file with data and labels
        """

        # Some pre-defined model
        self.window_time = 1.2

        if (leaveoneuserout):
            count = 0
            if (len(userids) != 1):
                print("invalid input")
                return
            # for userid in userids:
            #     pickle_file = pickle_file_prefix + "_" + str(userid) + ".pkl"
            #     with open(pickle_file, "rb") as f:
            #         data_ = pickle.load(f)
            #     if (count == 0):
            #         self.data = deepcopy(data_)
            #     else:
            #         self.data["X"] = np.append(self.data["X"], deepcopy(data_["X"]), axis = 0)
            #         self.data["Y"] = np.append(self.data["Y"], deepcopy(data_["Y"]), axis = 0)
            #     count += 1
            print(pickle_file_prefix + "_" + str(userids[0]) + ".pkl")
            with open(pickle_file_prefix + "_" + str(userids[0]) + ".pkl", "rb") as f:
                self.data = pickle.load(f)
        else:
            with open(pickle_file_prefix + ".pkl", "rb") as f:
                self.data = pickle.load(f)
        with open("../GESTURE_CONFIG.json", "r") as f:
            self.gesture_config = json.load(f)
            self.sr = self.gesture_config["samplerate"]
        
        self.rawinput = rawinput
        self.augment = augment
        self.input_size = input_size
        self.channels = channels
        self.street_noise = None
        self.office_noise = None
        self.device = device

        self.street_noise, _ = librosa.load("./user_data/processed/street_noise.wav",sr = self.sr)
        # self.street_noise = np.expand_dims(self.street_noise, axis = 1)
        self.street_noise = torch.from_numpy(self.street_noise).float()
        self.office_noise, _ = librosa.load("./user_data/processed/office_noise.wav",sr = self.sr)
        # self.office_noise = np.expand_dims(self.office_noise, axis = 1)
        self.office_noise = torch.from_numpy(self.office_noise).float()

        # self.transform = transforms.AmplitudeToDB(transforms.MelSpectrogram(sample_rate = self.sr,
        #                                             n_fft = 2048,
        #                                             hop_length = int(np.floor(1.2 * self.sr / self.input_size)),
        #                                             f_max = 735,
        #                                             n_mels = self.input_size
        #                                             ))
        self.transform = raw_audio_to_logmelspec(self.sr, self.input_size)

        self.X = []
        self.Y = []
        for x, i in zip(self.data["X"], self.data["Y"]):
            if (i[0] in self.gesture_config["train_labels_no_noise"]):
                idx = self.gesture_config["train_labels_no_noise"].index(i[0])
                self.Y.append(deepcopy(torch.tensor(idx, dtype = torch.long)))
                self.X.append(deepcopy(x))
        self.Y = np.array(self.Y)
        if (not self.rawinput): # if not the raw input, the input will be the processed 64 * 64 mel spec image
            self.X = np.array([scipy.ndimage.zoom(y, self.input_size / 64, order=0) for y in self.X])
            self.X = np.expand_dims(self.X, axis = 1)
            # if (channels > 1):
            #     self.X = np.repeat(self.X, repeats = channels, axis = 1)
        else:
            self.X = np.expand_dims(self.X, axis = 1)

        self.X = torch.from_numpy(self.X).float()
        self.Y = torch.from_numpy(self.Y)
        
        if (self.rawinput):
            self.X = [self.transform(x)[:,:self.input_size,:self.input_size] for x in self.X]
            self.X = torch.stack(self.X)
            if (self.channels > 1):
                self.X = self.X.repeat([1,self.channels,1,1])

        print(self.X.shape, self.Y.shape)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]
        return X,Y        


class SegDataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, pickle_file_prefix, device = "cpu", rawinput = False,
        leaveoneuserout = False, userids = [],
        augment = False, input_size = 64, channels = 1):
        """
        Args:
            pickle_file (string): Path to the pickle file with data and labels
        """

        # Some pre-defined model
        self.window_time = 1.2

        if (leaveoneuserout):
            count = 0
            if (len(userids) == 0):
                print("no userid, no data")
                return
            for userid in userids:
                pickle_file = pickle_file_prefix + "_" + str(userid) + ".pkl"
                with open(pickle_file, "rb") as f:
                    data_ = pickle.load(f)
                if (count == 0):
                    self.data = deepcopy(data_)
                else:
                    self.data["X"] = np.append(self.data["X"], deepcopy(data_["X"]), axis = 0)
                    self.data["Y"] = np.append(self.data["Y"], deepcopy(data_["Y"]), axis = 0)
                count += 1
        else:
            with open(pickle_file_prefix + ".pkl", "rb") as f:
                self.data = pickle.load(f)
        with open("../GESTURE_CONFIG.json", "r") as f:
            self.gesture_config = json.load(f)
            self.sr = self.gesture_config["samplerate"]
        
        self.rawinput = rawinput
        self.augment = augment
        self.input_size = input_size
        self.channels = channels
        self.street_noise = None
        self.office_noise = None
        self.device = device

        self.X = self.data["X"]
        self.Y = torch.tensor(self.data["Y"], dtype = torch.long)

        self.X = torch.from_numpy(self.X).float()
        # self.Y = torch.from_numpy(self.Y)

        print(self.X.shape, self.Y.shape)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]
        return X,Y        
