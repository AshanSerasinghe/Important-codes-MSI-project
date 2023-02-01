import os
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler




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



def convTo2DforPCA(mat3d):
    '''developer: Darsha '''

    band_no = 14
    samples_no = 9
    levels_no = mat3d.shape[0]
    width = mat3d.shape[2]
    blocksize = int(width/band_no)
    
    level = np.ones((1,band_no))
    for k in range(levels_no):
        for i in range(samples_no):
            sample = np.ones((int(blocksize**2),1))
            for j in range(band_no):
                imageblck = mat3d[k,i*blocksize:(i+1)*blocksize,j*blocksize:(j+1)*blocksize]
                imagecol = imageblck.reshape((blocksize**2,1))
                sample = np.hstack((sample,imagecol))
            level = np.vstack((level,sample[:,1:]))
    
    mat2d = level[1:,:]
    
    return mat2d


def PCA_apply(mat2d,comps):
    '''developer: Darsha '''

    #mat2d = mat2d[:,1:] #remove blackcurrent

    sc = StandardScaler()
    #data_scaled = sc.fit_transform(mat2d)
    data_scaled = mat2d

    pca = PCA(n_components = comps)
    data_reduced = pca.fit_transform(data_scaled)

    return data_reduced






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
    


def reshapeHistogramMat():
    numberOfBins = 10
    numOfLevels = 6
    numOfImgs = 14
    numberOfSamples = 9

    reshapedMat = np.zeros((numOfLevels,  numberOfBins*numberOfSamples, numOfImgs))

    histMat = makeHistograms()
    
    for level in range(numOfLevels):
        for i in range(numberOfSamples):
            for j in range(numOfImgs):
                reshapedMat[level, i*numberOfBins:(i+1)*numberOfBins, j] = histMat[level, i , j*numberOfBins:(j+1)*numberOfBins]
                

    return reshapedMat





def calculateBhattacharyaDistance(refData , compData):
    refMean = np.mean(refData , axis=0)
    compMean = np.mean(compData , axis=0)
    meanDiff = refMean - compMean

    refCov = np.cov(refData.T)
    compCov = np.cov(compData.T)

    Sigma = (refCov + compCov)/2
    SigmaInv = np.linalg.inv(Sigma)

    SigmaDet = np.linalg.det(Sigma)
    refCovDet = np.linalg.det(refCov)
    compCovDet = np.linalg.det(compCov)

    BhattacharyaDistance = (1/8)*(meanDiff.T).dot(SigmaInv).dot(meanDiff) + (1/2)*(np.log(SigmaDet/np.sqrt(refCovDet*compCovDet)))

    return BhattacharyaDistance


def mat2DforPCA_To_3DLevels():

    numOfImgs = 14
    numberOfSamples = 9
    numOfLevels = 6
    pixelSize = 10
    stride = 5 #(n+2p-f)/s +1---> 100-10/5+1=19
    imgSize = 100
    
    convolutesDim = int((imgSize-pixelSize)/stride + 1)

    mat3D = makeMain3DMat()
    reshapedMat = convTo2DforPCA(mat3D)
    
    mat3DLevels = np.zeros((numOfLevels, numberOfSamples*convolutesDim*convolutesDim, numOfImgs))
    

    for i in range(numOfLevels):
        mat3DLevels[i,:,:] = reshapedMat[i*(convolutesDim**2)*numberOfSamples:(i+1)*(convolutesDim**2)*numberOfSamples ,:]
    
    return mat3DLevels 
        

def calculateDistanceForAll():

    reshapedMat = mat2DforPCA_To_3DLevels()
   
    # reshapedMat = reshapedMat/361
    reference = reshapedMat[5,:,:]
    numOfLevels = 6
    numberOfSamples = 9
    pixelSize = 10
    stride = 5 #(n+2p-f)/s +1---> 100-10/5+1=19
    imgSize = 100
    convolutesDim = int((imgSize-pixelSize)/stride + 1)
    sampleSize = convolutesDim**2 # number of bins

    distanceMat = np.zeros((numOfLevels, numberOfSamples ,1))
    for level in range(numOfLevels):
        for i in range(numberOfSamples):
            distanceMat[level,i,0] = calculateBhattacharyaDistance(reference , reshapedMat[level, i*sampleSize:(i+1)*sampleSize, :])

    return distanceMat




def plotDistance():
    distanceMat = calculateDistanceForAll()
    dryRubberContent = [0,0,0,
                        8.858695652,8.972972973,8.855885589,
                        18.02551303,18.46994536,17.71238201,
                        30.39271485,31.45071982,31.27071823,
                        44.46978335,47.14697406,44.55388181,
                        54.84429066,53.27047731,51.72622653,
                        ]
    matShape = distanceMat.shape

    plotingData2D = np.zeros((matShape[0]*matShape[1],2))
    
    # for dryRubber in dryRubberContent:
    rubberContenIndex = 0
    plotIndex = 0

    for i in range(matShape[0]):
        for j in range(matShape[1]):
            plotingData2D[plotIndex,0] = distanceMat[i,j,0]
            plotingData2D[plotIndex,1] = dryRubberContent[rubberContenIndex]
            plotIndex+=1
            if plotIndex%3 == 0:
                rubberContenIndex+=1
    
    for i in plotingData2D[9:-1,0]:
        print(i)
    # print(plotingData2D[9:-1,1].T)   
    

    plt.scatter(plotingData2D[9:-1,1], plotingData2D[9:-1,0])
    plt.grid()
    plt.xlabel('Dry Rubber Content(W/W)')
    plt.ylabel('Bhattacharya Distance')
    plt.title('Dry Rubber Content Vs Bhattacharya Distance')
    plt.show()


    return plotingData2D



# avgImg = np.zeros((10,1))
# superImg = getSuperPixel(np.random.rand(100,100), 10)
# avgImg = np.hstack((avgImg , superImg))
# print(avgImg)
# print(makeMain3DMat()[0,:,:])

# print(makeHistograms()[5,3,:])
# x = reshapeHistogramMat()
# print(calculateBhattacharyaDistance(x[0,:,:] , x[1,:,:]))
# plotAllHistograms()

# x = calculateDistanceForAll()
# print(x.shape)
# print(x[0,:,:])


# x,y = mat2DforPCA_To_3DLevels()

# print(x[5,:,:])
# print("...........................................")
# print(y)

x = plotDistance()
