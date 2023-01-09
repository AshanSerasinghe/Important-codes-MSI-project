import cv2 as cv

img = cv.imread('D:\openCVfiles/lena.png', cv.IMREAD_UNCHANGED)        # change the path
print(img.shape)
cv.imshow("Image1", img)


resized = cv.resize(img, (32 , 32), interpolation = cv.INTER_AREA)     # dims should be a tuple. 
                                                                       # number of chanels remains same

#@@@@@@@interpolation@@@@@@@@
#[optional] flag that takes one of the following methods. INTER_NEAREST
#– a nearest-neighbor interpolation INTER_LINEAR – a bilinear interpolation (used by default)
#INTER_AREA – resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results.
#But when the image is zoomed, it is similar to the INTER_NEAREST method.
#INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood


print(resized.shape)
cv.imshow("Image2", resized)
