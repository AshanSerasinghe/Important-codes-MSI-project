import os
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt





def makeMain3DMat():
    pwd_main = r'F:\Data Sets\Rubber\Rubber\R\croped_samples\A'  #os.getcwd()
    folders = glob(pwd_main + "/*/", recursive = True)
    numOfImgs = 14
    numberOfSamples = 9
    numOfLevels = 6
    pixelSize = 10
    stride = 5 #(n+2p-f)/s +1---> 100-10/5+1=19
    imgSize = 100

    resultingW = int((imgSize-pixelSize)/stride + 1)*numOfImgs
    resultingH = int((imgSize-pixelSize)/stride + 1)*numberOfSamples
    convolutesDim = int((imgSize-pixelSize)/stride + 1)

    mainMatrix3D = np.ones((numOfLevels,resultingH, resultingW))
    mainMatCount = 0
    for f in folders:
        subFolders = glob(f + "/*/", recursive = True)
        avgMatrix = np.ones((1,resultingW))
        for sf in subFolders:
            file_arr = os.listdir(sf)

            avgImg = np.zeros((convolutesDim,1))
            for img in file_arr:
                theFile = os.path.join(sf , img)
                imgMatrix = cv2.imread(theFile , cv2.IMREAD_GRAYSCALE)
                # avg = np.sum(np.sum(imgMatrix))
                # print(np.sum(imgMatrix))
                superImg = superPixSlidingWindow(imgMatrix ,stride) #getSuperPixel(imgMatrix, pixelSize)
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
    imgSize = 100
    stride = 5
    convolutesDim = int((imgSize-pixelSize)/stride + 1)

    fig, ax = plt.subplots(9, 14, sharex='col', sharey='row')

    for i in range(numberOfSamples):
        for j in range(numOfImgs):
            superImg = mainMatrix3D[1,i*convolutesDim:(i+1)*convolutesDim , j*convolutesDim:(j+1)*convolutesDim]
            # plotSingleHistogram(superImg,i,j)
            rowData = superImg.reshape((convolutesDim**2))
            # fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)
            
            ax[i][j].hist(rowData, bins = 10)
            print((i,j))
    
    plt.show()


def superPixSlidingWindow(img ,overlap):
    '''
    Written by Darsha - ~20/01/2023

    (nxn)*(fxf)=> [(n+2p-f)/s +1]x[(n+2p-f)/s +1]
    n-image dimensions
    f-filter size
    p-padding 
    s-stride 

    '''
    # img = cv2.imread(imgname,cv2.IMREAD_GRAYSCALE)

    superimg = []
    for i in range(0,100,10-overlap):
        superrow = []
        for j in range(0,100,10-overlap):
            xend = j+9
            yend = i+9
            if (j>91) or (i>91):
                continue
            block = img[i:yend,j:xend]
            avgblck = np.sum(block)/100
            superrow.append(avgblck)
        if(superrow == []):
            continue
        try:
            superimg = np.vstack((superimg,superrow))
        except:
            superimg = superrow
    
    return superimg



def makeHistograms():
    mainMatrix3D = makeMain3DMat()
    numOfImgs = 14
    numberOfSamples = 9
    numOfLevels = 6
    pixelSize = 10
    numberOfBins = 10
    imgSize = 100
    stride = 5
    convolutesDim = int((imgSize-pixelSize)/stride + 1)

    
    histogramMatrix = np.zeros((numOfLevels, numberOfSamples, numberOfBins*numOfImgs))

    for level in range(numOfLevels):

        for i in range(numberOfSamples):
            for j in range(numOfImgs):
                superImg = mainMatrix3D[level ,i*convolutesDim:(i+1)*convolutesDim , j*convolutesDim:(j+1)*convolutesDim]
                # plotSingleHistogram(superImg,i,j)
                rowData = superImg.reshape((convolutesDim**2))
                maximum = np.max(rowData)
                minimum = np.min(rowData)
                
                binSize = (maximum-minimum)/(numberOfBins-1)# -1 to keep the number of bins == numberOfBins 
                binArray = np.zeros(numberOfBins)
                # print("Max: ",maximum , "Min ", minimum , "binSize*9 ", binSize*9 )
                for d in rowData:
                    if d<= minimum+binSize*1:
                        binArray[0] = binArray[0]+1
                    elif d<= minimum+binSize*2:
                        binArray[1] = binArray[1]+1
                    elif d<= minimum+binSize*3:
                        binArray[2] = binArray[2]+1
                    elif d<= minimum+binSize*4:
                        binArray[3] = binArray[3]+1
                    elif d<= minimum+binSize*5:
                        binArray[4] = binArray[4]+1
                    elif d<= minimum+binSize*6:
                        binArray[5] = binArray[5]+1
                    elif d<= minimum+binSize*7:
                        binArray[6] = binArray[6]+1
                    elif d<= minimum+binSize*8:
                        binArray[7] = binArray[7]+1
                    elif d<= minimum+binSize*9:
                        binArray[8] = binArray[8]+1
                    elif d>= minimum+binSize*9:
                        # near maximum value --> else
                        binArray[9] = binArray[9]+1
                   

                histogramMatrix[level ,i,  j*numberOfBins:(j+1)*numberOfBins] =  binArray
                # fig, axs = plt.subplots(1, 1,figsize =(10, 7), tight_layout = True)
    
    return histogramMatrix



def movingAverageSuperImg():
    ''''
    sliding window
    '''
    print("not yet implimented")
    



# avgImg = np.zeros((10,1))
# superImg = getSuperPixel(np.random.rand(100,100), 10)
# avgImg = np.hstack((avgImg , superImg))
# print(avgImg)
# print(makeMain3DMat()[0,:,:])

print(makeHistograms()[5,3,:])

# plotAllHistograms()
