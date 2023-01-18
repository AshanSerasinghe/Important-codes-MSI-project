import os
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.svm import SVC 


def MakeMainMatrix():
    pwd_main = 'F:\Data Sets\Turmeric 2018\Analysis\CroppedSamples' #os.getcwd()
    folders = glob(pwd_main + "/*/", recursive = True)
    numOfImgs = 10
    numberOfSamples = 30
    numOfLevels = 9

    mainMatrix3D = np.ones((numOfLevels,numberOfSamples, numOfImgs))
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




def stretchMatrixAddLabel(mat):
    numberOfImages = 10
    shapeTupple = mat.shape
    numberOfLevels = shapeTupple[0]
    numberOfSamples = shapeTupple[1]
    stretchedMatrix = np.zeros((1,numberOfImages+1))
    labelArray = [0,5,10,15,20,25,30,35,40]
    for i in range(numberOfLevels):
        label = np.ones((numberOfSamples,1))*labelArray[i]
        matWithLabel = np.hstack((mat[i,:,:],label))
        stretchedMatrix = np.vstack((stretchedMatrix,matWithLabel))
    
    return stretchedMatrix[1:,:]



def fitLinReg():

    labelArray = [0,5,10,15,20,25,30,35,40]
    mat = MakeMainMatrix()
    Xy = stretchMatrixAddLabel(mat)
    X = Xy[:,0:-2]
    y = Xy[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    y_preadicted = regr.predict(X)

    y_category = [min(labelArray, key=lambda x: abs(x - entry)) for entry in y_preadicted]

    print(regr.score(X_test, y_test))
    # print(y_category)
    # print(y_preadicted)

    cof_mat = confusion_matrix(y,y_category); #print(cof_mat)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cof_mat, display_labels = labelArray)
    cm_display.plot()
    plt.show()










def PCApalmOil():

    mat = MakeMainMatrix()
    Xy = stretchMatrixAddLabel(mat)
    X = Xy[:,0:-2]
    y = Xy[:,-1]

    scaling=StandardScaler()

    scaling.fit(X)
    Scaled_data=scaling.transform(X)

    principal=PCA(n_components=3)
    principal.fit(Scaled_data)
    x=principal.transform(Scaled_data)
    print(Xy)


    
    colorArray_gen = matplotlib.cm.Set1(range(10))
    colorArray_gen = [x for x in colorArray_gen]
    colorArray = ['black','green','blue'] + colorArray_gen
    counter = 0
    for i in range(10):
        plt.scatter(x[counter:counter+9,0],x[counter:counter+9,1],color=colorArray[i])
        counter+=9
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    plt.grid()
    
    levelsArray = ['0','5','10','15','20','25','30','35','40']
    patchArray = []
    for c,level in zip(colorArray,levelsArray):
        patchArray.append(mpatches.Patch(color=c, label=level ))

    plt.legend(handles = patchArray)
    


    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    counter = 0
    for i in range(10):
        axis.scatter(x[counter:counter+9,0],x[counter:counter+9,1],x[counter:counter+9,2],color=colorArray[i])
        counter+=9
    
    axis.set_xlabel("PC1", fontsize=10)
    axis.set_ylabel("PC2", fontsize=10)
    axis.set_zlabel("PC3", fontsize=10)
    plt.legend(handles = patchArray)

    plt.show()



def applySVM():
    labelArray = [0,5,10,15,20,25,30,35,40]
    mat = MakeMainMatrix()
    Xy = stretchMatrixAddLabel(mat)
    X = Xy[:,0:-2]
    y = Xy[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    clf = SVC(kernel='linear') 
    clf.fit(X_train,y_train)

    yPred = clf.predict(X)

    print(np.sum(yPred == y)/y.shape[0])

    cof_mat = confusion_matrix(y,yPred); print(cof_mat)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cof_mat, display_labels = labelArray)
    cm_display.plot()
    plt.show()




#==============================================

# print(MakeMainMatrix())

mat = MakeMainMatrix()
print(stretchMatrixAddLabel(mat))


# PCApalmOil()
# fitLinReg()


applySVM()
#==============================================



