import os
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib


def MakeMainMatrix():
    pwd_main = 'F:\Data Sets\Turmeric 2018\Analysis\CroppedSamples' #os.getcwd()
    folders = glob(pwd_main + "/*/", recursive = True)
    numOfImgs = 10
    numberOfSamples = 30
    numOfLevels = 9

    mainMatrix3D = np.ones((numOfLevels,numberOfSamples,numOfImgs))
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
    numAflLevel = 9
    numOfImgs = 10
    
    # colorArray = ['green' , 'black', 'red','blue' , (0,0.4,0.8), (0.2,0.5,0.2), (0.1,0.4,0.6), (0.15,0.45,0.7), (0.45,0.65,0.15)]
    colorArray = matplotlib.cm.tab20(range(20))
    for i in range(0,numAflLevel):
        imgMatrix = mainMatrix[i,:,:]
        numRows = imgMatrix.shape[0]
        for j in range(0,numRows):
            plt.plot(list(range(0,numOfImgs)), imgMatrix[j], color=colorArray[i], linestyle='solid', linewidth = 1, marker='o')
    
    patchArray = []
    levelsArray = ['0','5','10','15','20','25','30','35','40']
    for c,level in zip(colorArray,levelsArray):
        patchArray.append(mpatches.Patch(color=c, label=level ))

    plt.legend(handles = patchArray)
    plt.grid()
    plt.show()        


plotSignature()
# print(MakeMainMatrix() , MakeMainMatrix().shape)