# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import json


with open("../GESTURE_CONFIG.json", "r") as f:
    gesture_config = json.load(f)

GESTURE = gesture_config["gesture_list_formal"]

if not os.path.exists('./DataStudy1_Plot'):
    os.mkdir('./DataStudy1_Plot')

for g in range(0,len(GESTURE)):
    if os.path.exists('./DataStudy1/'+str(GESTURE[g])):
        if not os.path.exists('./DataStudy1_Plot/'+str(GESTURE[g])):
            os.mkdir('./DataStudy1_Plot/'+str(GESTURE[g]))
        for u in os.listdir('./DataStudy1/'+str(GESTURE[g])):
            for r, r_name in enumerate(os.listdir('./DataStudy1/'+str(GESTURE[g])+'/'+str(u))):
                
                NAME_I = './DataStudy1/'+str(GESTURE[g])+'/'+str(u)+'/'+r_name
                NAME_O = './DataStudy1_Plot/'+str(GESTURE[g])+'/'+str(u)+'_'+str(r)

                x, sr = librosa.load(NAME_I)

                fig, axes = plt.subplots(figsize=(18, 9), ncols = 2)
                librosa.display.waveplot(x, sr=sr, ax = axes[0])

                X = librosa.stft(x)
                Xdb = librosa.amplitude_to_db(abs(X))
                librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', ax = axes[1])

                plt.savefig(NAME_O + '.png')
                print(NAME_O + '  ..DONE')

    else:
        pass
