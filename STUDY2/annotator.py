import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
import json
from copy import deepcopy
from matplotlib import cm
from numpy.linalg import norm
import threading
import collections
import time
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from matplotlib.widgets import Button

with open("../GESTURE_CONFIG.json", "r") as f:
    gesture_list_config = json.load(f)
gesture_list = gesture_list_config["gesture_list_formal"]

annotation_index = 0
file_list = []

class ClickPlot:
    """
    A clickable matplotlib figure
    """

    def __init__(self, fig=None, filename = "default"):
    
        """
        Constructor
        
        Arguments:
        fig -- a matplotlib figure
        """
    
        if fig != None:
            self.fig = fig      
        else:
            self.fig = plt.get_current_fig_manager().canvas.figure
        self.nSubPlots = len(self.fig.axes)
        self.filename = filename
        self.dragFrom = None
        self.comment = '0'
        self.markers = []
                
        self.retVal = {'comment' : self.comment, 'x' : [], 'y' : [],
            'subPlot' : None}       

        self.sanityCheck()
        self.supTitle = plt.suptitle(self.filename.split("/")[-2] + "\n" + self.filename.split("/")[-1].replace(".txt", ""))
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.fig.canvas.mpl_connect('button_release_event', self.onRelease)     
        self.fig.canvas.mpl_connect('scroll_event', self.onScroll)
        self.fig.canvas.mpl_connect('key_press_event', self.onKey)
        self.lasteventtime = time.time()

    def reset(self, fig, filename):

        # clear the figure
        # plt.pause(0.001)
        # plt.clf()
        plt.pause(0.001)
        plt.close(self.fig)
        plt.pause(0.001)

        self.fig = fig
        self.nSubPlots = len(self.fig.axes)
        self.filename = filename
        self.dragFrom = None
        self.comment = '0'
        self.markers = []
                
        self.retVal = {'comment' : self.comment, 'x' : [], 'y' : [],
            'subPlot' : None}       

        self.sanityCheck()
        self.supTitle = plt.suptitle(self.filename.split("/")[-2] + "\n" + self.filename.split("/")[-1].replace(".txt", ""))
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.fig.canvas.mpl_connect('button_release_event', self.onRelease)     
        self.fig.canvas.mpl_connect('scroll_event', self.onScroll)
        self.fig.canvas.mpl_connect('key_press_event', self.onKey)
        self.lasteventtime = time.time()
        plt.show()
        self.fig.canvas.draw()

        
    def clearMarker(self):
    
        """Remove marker from retVal and plot"""
        
        # self.retVal['x'] = []
        # self.retVal['y'] = []
        self.retVal['subPlot'] = None
        for i in range(self.nSubPlots):
            subPlot = self.selectSubPlot(i)
            for marker in self.markers:
                if marker in subPlot.lines:
                    subPlot.lines.remove(marker)
                if marker in subPlot.patches:
                    subPlot.patches.remove(marker)
        self.markers = []
        self.fig.canvas.draw()
        
    def getSubPlotNr(self, event):
    
        """
        Get the nr of the subplot that has been clicked
        
        Arguments:
        event -- an event
        
        Returns:
        A number or None if no subplot has been clicked
        """
    
        i = 0
        axisNr = None
        for axis in self.fig.axes:
            print("in getSubPlotNr")
            if axis == event.inaxes:
                axisNr = i      
                break
            i += 1
        return axisNr
        
    def sanityCheck(self):
    
        """Prints some warnings if the plot is not correct"""
        
        subPlot = self.selectSubPlot(0)
        minX = subPlot.dataLim.min[0]
        maxX = subPlot.dataLim.max[0]    
        for i in range(self.nSubPlots):
            subPlot = self.selectSubPlot(i)
            _minX = subPlot.dataLim.min[0]
            _maxX = subPlot.dataLim.max[0]
            if abs(_minX-minX) != 0 or (_maxX-maxX) != 0:
                import warnings     
                warnings.warn('Not all subplots have the same X-axis')
        
    def show(self):
    
        """
        Show the plot
        
        Returns:
        A dictionary with information about the response
        """
        plt.show()
        self.retVal['comment'] = self.comment
        return self.retVal
        
    def selectSubPlot(self, i):
    
        """
        Select a subplot
        
        Arguments:
        i -- the nr of the subplot to select
        
        Returns:
        A subplot
        """
        plt.subplot('{0}1{1}'.format(len(self.fig.axes), i+1))
        return self.fig.axes[i]

    def updateClickPoint(self,x,y):
        l = len(self.retVal['x'])
        if (l == 0 or l == 2):
            self.retVal['x'] = [x]
            self.retVal['y'] = [y]
        elif (l == 1):
            self.retVal['x'].append(x)
            self.retVal['y'].append(y)
        print(self.retVal['x'])
        print(self.retVal['y'])

    def validToDrawOnClick(self):
        return (len(self.retVal['x']) == 2)

    def onClick(self, event):
    
        """
        Process a mouse click event. If a mouse is right clicked within a
        subplot, the return value is set to a (subPlotNr, xVal, yVal) tuple and
        the plot is closed. With right-clicking and dragging, the plot can be
        moved.
        
        Arguments:
        event -- a MouseEvent event
        """     
    
        subPlotNr = self.getSubPlotNr(event)        
        if subPlotNr == None:
            print("return early on click")
            return
        
        if event.button == 1:               
            self.retVal['subPlot'] = subPlotNr
            self.updateClickPoint(event.xdata, event.ydata)
            if (self.validToDrawOnClick()):
                self.clearMarker()
                for i in range(self.nSubPlots):
                    subPlot = self.selectSubPlot(i)
                    marker = plt.axvspan(self.retVal['x'][0], self.retVal['x'][1],alpha = 0.3, color = "red")
                    self.markers.append(marker)
            else:
                self.clearMarker()
                for i in range(self.nSubPlots):
                    subPlot = self.selectSubPlot(i)
                    marker = plt.axvline(x = self.retVal['x'][0],
                        linestyle='--',
                        linewidth=1, color='red', alpha = 0.5)
                    self.markers.append(marker)
            # self.retVal['subPlot'] = subPlotNr
            # self.retVal['x'] = event.xdata
            # self.retVal['y'] = event.ydata
            self.fig.canvas.draw()
            
        else:           
            # Start a dragFrom
            self.dragFrom = event.xdata
            

    def recordResults(self, valid = True):
        global annotation_index
        # record previous results
        
        split = self.filename.split("/")
        folder = "/".join(split[:-1])
        if (not os.path.exists(folder)):
            os.makedirs(folder + "/")
        print("Recording... No." + str(annotation_index) + ", " + filename)

        if (valid):
            with open(self.filename, "w") as f:
                f.write(str(self.retVal["x"][0]) + "," + str(self.retVal["x"][1]))
        else:
            with open(self.filename, "w") as f:
                f.write("bad_data")
        with open("annotation_index.txt", "w") as f:
            f.write(str(annotation_index))

    def onKey(self, event):
    
        """
        Handle a keypress event. The plot is closed without return value on
        enter. Other keys are used to add a comment.
        
        Arguments:
        event -- a KeyEvent
        """
        global annotation_index

        currenttime = time.time()
        # print(event.key, len(event.key))
        if ((currenttime - self.lasteventtime) < 0.2):
            return

        if (event.key == 'enter'):
            plt.close()
            # return
        elif (event.key == 'escape'):
            self.clearMarker()
            # return
        elif (event.key == "right"):
            if (self.validToDrawOnClick()):
                self.recordResults(valid = True)
                annotation_index += 1
                if (annotation_index >= len(file_list)):
                    annotation_index = len(file_list) - 1
                    plt.close()
                    return
                audiofile = file_list[annotation_index]
                f, filename = show_audio_signal(audiofile, start_time = 0, duration=3)
                self.reset(f, filename)
            # return
        elif (event.key == "left"):
            annotation_index -= 1
            if (annotation_index < 0):
                annotation_index = 0
            audiofile = file_list[annotation_index]
            f, filename = show_audio_signal(audiofile, start_time = 0, duration=3)
            self.reset(f, filename)
            # return
        elif (event.key == " " or event.key == "down"):
            self.recordResults(valid = False)
            annotation_index += 1
            if (annotation_index >= len(file_list)):
                annotation_index = len(file_list) - 1
                plt.close()
                return
            audiofile = file_list[annotation_index]
            f, filename = show_audio_signal(audiofile, start_time = 0, duration=3)
            self.reset(f, filename)
        self.lasteventtime = currenttime

        # if event.key == 'backspace':
        #   self.comment = self.comment[:-1]
        # elif len(event.key) == 1:         
        #   self.comment += event.key
        # self.supTitle.set_text("comment: %s" % self.comment)
        # event.canvas.draw()
            
    def onRelease(self, event):
    
        """
        Handles a mouse release, which causes a move
        
        Arguments:
        event -- a mouse event
        """
    
        if self.dragFrom == None or event.button != 3:
            return          
        dragTo = event.xdata
        dx = self.dragFrom - dragTo
        for i in range(self.nSubPlots):
            subPlot = self.selectSubPlot(i)         
            xmin, xmax = subPlot.get_xlim()
            xmin += dx
            xmax += dx              
            subPlot.set_xlim(xmin, xmax)
        event.canvas.draw()
                                            
    def onScroll(self, event):
    
        """
        Process scroll events. All subplots are scrolled simultaneously
        
        Arguments:
        event -- a MouseEvent
        """
    
        for i in range(self.nSubPlots):
            subPlot = self.selectSubPlot(i)     
            xmin, xmax = subPlot.get_xlim()
            dx = xmax - xmin
            cx = (xmax+xmin)/2
            if event.button == 'down':
                dx *= 1.1
            else:
                dx /= 1.1
            _xmin = cx - dx/2
            _xmax = cx + dx/2   
            subPlot.set_xlim(_xmin, _xmax)
        event.canvas.draw()
        
def showClickPlot(fig=None, filename = "default"):

    """
    Show a plt and return a dictionary with information
    
    Returns:
    A dictionary with the following keys:
    'subPlot' : the subplot or None if no marker has been set
    'x' : the X coordinate of the marker (or None)
    'y' : the Y coordinate of the marker (or None)
    'comment' : a comment string    
    """
    cp = ClickPlot(fig, filename)
    return cp.show()

def show_audio_signal(NAME, start_time = 0, duration = 1):
    print(NAME)
    y, sr = librosa.load(NAME, sr = gesture_list_config["samplerate"])

    delay = int(sr * start_time)
    y = y[delay: delay + int(duration * sr)]
    fig = plt.figure(figsize=(3, 8))

    subPlot1 = fig.add_subplot('311')
    librosa.display.waveplot(y, sr=sr, figure=fig)
    
    subPlot2 = fig.add_subplot('312')

    Y = librosa.stft(y)
    Ydb = librosa.amplitude_to_db(abs(Y))
    ax = librosa.display.specshow(Ydb, sr=sr, x_axis='time', y_axis='hz', figure=fig)
    ax.set_ylim((0,1024))

    subPlot3 = fig.add_subplot('313')
    mspec = librosa.amplitude_to_db(librosa.feature.melspectrogram(y, sr))
    ax = librosa.display.specshow(mspec, sr=sr, x_axis='time', y_axis='hz', figure=fig)
    ax.set_ylim((0,1024))

    # subPlot4 = fig.add_subplot('414')
    # mfcc = librosa.feature.mfcc(y, sr)
    # librosa.display.specshow(mfcc, sr=sr, x_axis='time', y_axis='hz', figure=fig)

    return fig, NAME.replace(".wav", ".txt").replace("raw", "annotation")


if __name__ == '__main__':
    with open("./user_data/raw_file_list.txt", "r") as f:
        file_list = f.readlines()
    file_list = [x.strip() for x in file_list]

    if (os.path.exists("annotation_index.txt")):
        with open("annotation_index.txt", "r") as f:
            annotation_index = f.readline()
    else:
        annotation_index = 0
        with open("annotation_index.txt", "w") as f:
            f.write(str(annotation_index))
    if (len(str(annotation_index)) < 1):
        annotation_index = 0
    print("start from:", annotation_index)
    annotation_index = int(annotation_index)

    # title = audiofile.split("/")[-2] + "-" + audiofile.split("/")[-1]
    audiofile = file_list[annotation_index]
    f, filename= show_audio_signal(audiofile, start_time = 0, duration=3)
    showClickPlot(f, filename)
