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
    numOfLevels = 4

    mainMatrix3D = np.ones((numOfLevels,numberOfSamples,14))
    mainMatCount = 0
    for f in folders:
        subFolders = glob(f + "/*/", recursive = True)
        avgMatrix = np.ones((1,numOfImgs))
        for sf in subFolders:
            file_arr = os.listdir(sf)

            avgImg = []
            for img in file_arr:
                theFile = os.path.join(sf , img)
                imgMatrix = cv2.imread(theFile , cv2.IMREAD_GRAYSCALE)
                avg = np.sum(np.sum(imgMatrix))
                # print(np.sum(imgMatrix))
                avgImg.append(avg/10000)

            avgMatrix = np.vstack((avgMatrix, np.array(avgImg))) 
        
        mainMatrix3D[mainMatCount,:,:] = avgMatrix[1:,:]
        mainMatCount+=1
    
    return mainMatrix3D



def plotSignature():
    mainMatrix = MakeMainMatrix()
    numAflLevel = 4
    colorArray = ['green' , 'black', 'red','blue']
    for i in range(0,numAflLevel):
        imgMatrix = mainMatrix[i,:,:]
        numRows = imgMatrix.shape[0]
        for j in range(0,numRows):
            plt.plot(list(range(0,14)), imgMatrix[j], color=colorArray[i], linestyle='solid', linewidth = 1, marker='o')
    
    green_patch = mpatches.Patch(color='green', label='Day 0')
    black_patch = mpatches.Patch(color='black', label='Day 3')
    red_patch = mpatches.Patch(color='red', label='day 5')
    blue_patch = mpatches.Patch(color='blue', label='day 6')
  


    plt.legend(handles = [green_patch,black_patch,red_patch, blue_patch])
    plt.grid()
    plt.show()        


plotSignature()
# print(MakeMainMatrix() , MakeMainMatrix().shape)