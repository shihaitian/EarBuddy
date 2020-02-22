# -*- coding:utf-8 -*-
import os
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
folder_path = os.path.dirname(abspath(__file__))

import numpy as np
import pyaudio
from pathlib import Path
import time
import wave
import argparse
import wget
import threading
import random
import json
import librosa
import librosa.display
import matplotlib.pyplot as plt
from utils.microphone import select_microphone_cmd


###########################    
# Settings
###########################    

with open("../GESTURE_CONFIG.json", "r") as f:
    gesture_config = json.load(f)
GESTURE = gesture_config["gesture_list_formal"]
#RECORD_SECONDS = 4
RECORD_SECONDS = 3
FORMAT = pyaudio.paInt16
CHANNELS = gesture_config["channels"]
RATE = gesture_config["samplerate"]
HZ = 25
CHUNK = RATE // HZ

###########################
# Check microphones
# Then read the command line input to select a microphone
###########################    

MICROPHONE_INDEX = select_microphone_cmd()

p = pyaudio.PyAudio()

u = 0
###########################
# Sub functions
###########################

def TIMER():

    print('2')
    time.sleep(1)

    print('1')
    time.sleep(1)

    print('.')
    time.sleep(0.5)

    print('.')
    time.sleep(0.5)

    print('.')
    time.sleep(0.5)
    
    return

def RECORDER(NAME):
    # Start 0.5s after the "2" sign shows up
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK,input_device_index = MICROPHONE_INDEX)

    t2 = threading.Thread(target=TIMER, args=())
    t2.start()

    frames = []
    time.sleep(0.5)
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(NAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    # show a spectrogram right away to timely discover any errors
    # x, sr = librosa.load(NAME)
    # fig, axes = plt.subplots(figsize=(18, 9), ncols = 2)
    # librosa.display.waveplot(x, sr=sr, ax = axes[0])
    # X = librosa.stft(x)
    # Xdb = librosa.amplitude_to_db(abs(X))
    # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', ax = axes[1])
    # plt.show()
    return

def RecordOnce(r,g, PATH):
    global u

    NAME = PATH + str(r)+'.wav'

    t1 = threading.Thread(target=RECORDER, args=(NAME,))
    t1.start()

    t1.join()

    print('Nice job! Record done\n')
    
    return


def Record(t, g, rept):
    global u

    print('\n=================')
    time.sleep(0.5)
    print('Test: ', t, 'of ' + str(len(GESTURE)))
    print('Test description: ', GESTURE[g])
    print('\n=================')

    while True:
        key = input('Be careful! Please pay attention to the countdown\nPress "y/1" to start: ')
        if (key == 'Y' or key == 'y' or key == 1 or key == "1"):
            break

    r=1

    PATH = folder_path + '/DataStudy1/'+str(GESTURE[g]).replace(" ","_")+'/'+'user'+str(u) + "/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    while r<rept+1:

        RecordOnce(r,g, PATH)

        #just pressing enter will save the file,any other key pressed before enter means redo
        if r < rept:
            key = input('Press "y/1" to save, or press "n/2" to discard: ').strip()
            
            if (key == 'Y' or key == 'y' or key == 1 or key == "1"):
                print('Data has been saved\n')
                time.sleep(0.5)
                print('round %d is coming: \n'%((r+1)))
                r = r + 1
            else:
                print('Data has been discarded')
                time.sleep(0.5)
                print('this round will be repeat: \n')

        elif r == rept:
            key = input('Press "y/1" to save, or press "n/2" to discard: ').strip()
            if (key == 'Y' or key == 'y' or key == 1 or key == "1"):
                print('\nCongratulations, this test has been completed')
                print('================')
                time.sleep(0.5)
                print('The next test starts right away: \n' )
                r = r + 1
            else:
                print('Data has been discarded')
                time.sleep(0.5)
                print('this round will be repeat: \n')

        time.sleep(1)        
    return


###########################
# Main function
###########################


def main():

    global u
    u = input('Please enter your User ID: ')
    print('\nHello!','\nYour ID is No.', str(u), "\n")

    REPEAT = int(input('Please set the number of repetitions for each gesture: '))
    print('Repeat time: ', REPEAT)

    gesture_list = list(range(0,len(GESTURE)))
    
    for session in range(1,11):
        print('\nsession %s' % session)
        random.shuffle(gesture_list)
    
        for t in range(0,len(GESTURE)):
            g = gesture_list[t]
            Record(t+1, g, REPEAT)
        print('\nsession %s finished\n' % session)

    print('Congratulations! You have completed all the tests.\nThank you for your participation!\nYou can close the window now')
    time.sleep(0.5)

if __name__ == "__main__":
    main()
