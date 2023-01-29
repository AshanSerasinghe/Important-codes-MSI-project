import cv2
import numpy as np

def superPix(imgname,overlap):
    '''
    Written by Darsha - 20/01/2023
    '''
    img = cv2.imread(imgname,cv2.IMREAD_GRAYSCALE)
    superimg = []
    for i in range(0,100,10-overlap):
        superrow = []
        #print("loop i is ",i)
        for j in range(0,100,10-overlap):
            xend = j+9
            yend = i+9
            if (j>91) or (i>91):
                continue
            block = img[i:yend,j:xend]
            avgblck = np.sum(block)/100
            superrow.append(avgblck)
        #print(superrow)
        #superimg = np.vstack((superimg,superrow))
        if(superrow == []):
            continue
        try:
            #print("in try superimg ",superimg)
            #print("in try superrow ",superrow)
            superimg = np.vstack((superimg,superrow))
        except:
            #print("in except")
            superimg = superrow
            #print(superimg)
        
    print(superimg)


superPix('735nm.png',5)