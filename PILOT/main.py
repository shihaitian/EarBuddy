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

from utils.microphone import select_microphone_cmd
# from gestures import GESTURE

# Read from the universal config file

with open("../GESTURE_CONFIG.json", "r") as f:
    gesture_config = json.load(f)

GESTURE = gesture_config["gesture_list_pilot"]

###########################    
#SETTING FOR RECOEDER
###########################    
RECORD_SECONDS = 3
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = gesture_config["channels"]
RATE = gesture_config["samplerate"]


# 检查麦克风 读取命令行输入选取麦克风
###########################    
MICROPHONE_INDEX = select_microphone_cmd()

p = pyaudio.PyAudio()

u = 999

###########################
#结构概述：
#整个实验分三层：USER-TEST-ROUND
#USER是不同用户（数量不限），TEST是不同手势（21个），ROUND是每个手势的不同重复（自己定义，即REPEAT参数！）

def TIMER():#ROUND函数中的倒计时线程
    print('3')
    time.sleep(1)

    print('2')
    time.sleep(1)

    print('1')
    time.sleep(1)

    print('Go! Start Recording...')
    # Comment this sleep since it user should start performing gestures when seeing this text
    # time.sleep(1)

    print('.')
    time.sleep(1)

    print('.')
    time.sleep(1)

    print('.')
    time.sleep(1)
    
    return

def RECORDER(NAME):#ROUND函数中的录音线程
    time.sleep(1) # Start when the "1" sign shows up
    # print("RECORDER ON")
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK,input_device_index = MICROPHONE_INDEX)
    # print("STREAM OPEN")

    frames = []

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
    return

def RecordOnce(r,g, PATH):#ROUND函数
    global u

    NAME = PATH + str(r)+'.wav'

    t1 = threading.Thread(target=RECORDER, args=(NAME,))
    t2 = threading.Thread(target=TIMER, args=())
    t1.start()
    t2.start()
    t1.join()
    t2.join()

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

    PATH = folder_path + '/Data/'+str(GESTURE[g]).replace(" ","_")+'/'+'user'+str(u) + "/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    while r<rept+1:
        print("round", r)
        RecordOnce(r,g, PATH)

        #just pressing enter will save the file,any other key pressed before enter means redo
        if r < rept:
            key = input('Press "y/1" to save, or press "n/2" to discard: ').strip()
            
            if (key == 'Y' or key == 'y' or key == 1 or key == "1"):
                print('Data has been saved')
                time.sleep(0.5)
                print('the next round is coming: \n')
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
                print('For this gesture, please rate from 1 - 7, with 1 being Strongly Disagree, to 7 Strongly Agree: ')
                print('Q1: The gesture is convenient to perform.')
                score_1 = input('Your rating (1 - 7): ')
                print('Q2: The gesture is easy to remember.')
                score_2 = input('Your rating (1 - 7): ')
                print('Q3: The gesture makes me tired.')
                score_3 = input('Your rating (1 - 7): ')

                with open(PATH + "rate.txt", "w") as f:
                    f.write(str(score_1) + "," + str(score_2) + "," + str(score_3))

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
# 主程序
###########################


def main():
    global u

    u = int(input('Please enter your User ID: '))

    print('\nHello!','\nYour ID is No.%s \n' % u)

    REPEAT = int(input('Please set the number of repetitions for each gesture: '))
    print('Repeat time: ', REPEAT)

    gesture_list = list(range(0,len(GESTURE)))
    # random.shuffle(gesture_list)
    
    for t in range(0,len(GESTURE)):#这个是最核心的主程序体，循环进行Record
        g = gesture_list[t]
        Record(t+1, g, REPEAT)

    print('Congratulations! You have completed all the tests.\nThank you for your participation!\nYou can close the window now')
    time.sleep(0.5)

if __name__ == "__main__":
    main()
