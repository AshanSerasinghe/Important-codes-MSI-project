import os
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt





def makeMain3DMat():
    pwd_main = 'F:\Data Sets\Tumeric\Tumeric MSI\croped_samples'  #os.getcwd()
    folders = glob(pwd_main + "/*/", recursive = True)
    numOfImgs = 14
    numberOfSamples = 9
    numOfLevels = 9
    pixelSize = 10

    mainMatrix3D = np.ones((numOfLevels, numberOfSamples*pixelSize, 14*pixelSize))
    mainMatCount = 0
    for f in folders:
        subFolders = glob(f + "/*/", recursive = True)
        avgMatrix = np.ones((1,numOfImgs*10))
        for sf in subFolders:
            file_arr = os.listdir(sf)

            avgImg = np.zeros((pixelSize,1))
            for img in file_arr:
                theFile = os.path.join(sf , img)
                imgMatrix = cv2.imread(theFile , cv2.IMREAD_GRAYSCALE)
                # avg = np.sum(np.sum(imgMatrix))
                # print(np.sum(imgMatrix))
                superImg = getSuperPixel(imgMatrix, pixelSize)
                avgImg = np.hstack((avgImg , superImg))
                
            avgMatrix = np.vstack((avgMatrix, avgImg[:,1:])) 
        
        mainMatrix3D[mainMatCount,:,:] = avgMatrix[1:,:]
        mainMatCount+=1
    
    return mainMatrix3D


def getSuperPixel(img, size):
    
    shp = img.shape
    hight = shp[0]
    width = shp[1]

    superImage = np.zeros( (round(hight/size) , round(width/size)) )

    for i in range(0,round(width/size)):
        for j in range(0,round(hight/size)):
            piece = img[i*size:size*(i+2)-size, j*size:size*(j+2)-size]
            superImage[i,j] = np.sum(piece)/(size**2)

    
    return superImage



def plotSingleHistogram(data ,  i,j):
    shp = data.shape
    hight = shp[0]
    width = shp[1]
    rowData = data.reshape((1,hight*width))
    # fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)
    global ax
    ax[i][j].hist(rowData, bins = 5)
    


def plotAllHistograms():
    mainMatrix3D = makeMain3DMat()
    numOfImgs = 14
    numberOfSamples = 9
    numOfLevels = 9
    pixelSize = 10

    fig, ax = plt.subplots(9, 14, sharex='col', sharey='row')

    for i in range(numberOfSamples):
        for j in range(numOfImgs):
            superImg = mainMatrix3D[1,i*pixelSize:(i+2)*pixelSize-pixelSize , j*pixelSize:(j+2)*pixelSize-pixelSize]
            # plotSingleHistogram(superImg,i,j)
            rowData = superImg.reshape((10*10))
            # fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)
            
            ax[i][j].hist(rowData, bins = 20)
            print((i,j))
    
    plt.show()




def makeHistograms():
    mainMatrix3D = makeMain3DMat()
    numOfImgs = 14
    numberOfSamples = 9
    numOfLevels = 9
    pixelSize = 10
    numberOfBins = 10

    for i in range(numberOfSamples):
        for j in range(numOfImgs):
            superImg = mainMatrix3D[1,i*pixelSize:(i+2)*pixelSize-pixelSize , j*pixelSize:(j+2)*pixelSize-pixelSize]
            # plotSingleHistogram(superImg,i,j)
            rowData = superImg.reshape((10*10))
            maximum = np.max(rowData)
            minimum = np.min(rowData)
            
            binSize = round(maximum-minimum)/numberOfBins
            binArray = np.zeros((1,numberOfBins))
            for d in rowData:
                if d<= binSize:
                    binArray[0] = binArray[0]+1
                elif d<= binSize*2:



            # fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)
    



def movingAverageSuperImg():
    ''''
    sliding window
    '''
    



# avgImg = np.zeros((10,1))
# superImg = getSuperPixel(np.random.rand(100,100), 10)
# avgImg = np.hstack((avgImg , superImg))
# print(avgImg)
# print(makeMain3DMat().shape)

plotAllHistograms()
