import os
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def MakeMainMatrix():
    pwd_main = os.getcwd()
    folders = glob(pwd_main + "/*/", recursive = True)
    numOfImgs = 14
    numberOfSamples = 24

    mainMatrix3D = np.ones((3,numberOfSamples,14))
    mainMatCount = 0
    for f in folders:
        subFolders = glob(f + "/*/", recursive = True)
        avgMatrix = np.ones((1,numOfImgs))
        for sf in subFolders:
            file_arr = os.listdir(sf)

            avgImg = []
            for img in file_arr:
                theFile = os.path.join(sf , img)
                imgMatrix = cv2.imread(theFile)
                avg = np.sum(np.sum(imgMatrix))
                avgImg.append(avg/10000)

            avgMatrix = np.vstack((avgMatrix, np.array(avgImg))) 
        
        mainMatrix3D[mainMatCount,:,:] = avgMatrix[1:,:]
        mainMatCount+=1
    
    return mainMatrix3D



def plotSignature():
    mainMatrix = MakeMainMatrix()
    numAflLevel = 3
    colorArray = ['green' , 'black', 'red']
    for i in range(0,numAflLevel):
        imgMatrix = mainMatrix[i,:,:]
        numRows = imgMatrix.shape[0]
        for j in range(0,numRows):
            plt.plot(list(range(0,14)), imgMatrix[j], color=colorArray[i], linestyle='solid', linewidth = 3, marker='o')
    
    green_patch = mpatches.Patch(color='green', label='Day 0')
    black_patch = mpatches.Patch(color='black', label='Day 3')
    red_patch = mpatches.Patch(color='red', label='day 5')


    plt.legend(handles = [green_patch,black_patch,red_patch])
    plt.grid()
    plt.show()        



plotSignature()