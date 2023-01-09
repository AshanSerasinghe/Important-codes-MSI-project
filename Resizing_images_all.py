import cv2
import os

# rescale to 40x30
# covariance matrix may require 12*4 MB

#D:\Ebooks\projects\MSI project\data_set\poluted_croped
#D:\Ebooks\projects\MSI project\data_set\distill_croped_new
file_arr = os.listdir("D:\Ebooks\projects\MSI project\data_set\poluted_croped")


#change the cuurren working directory
#D:\Ebooks\projects\MSI project\data_set\poliuted_cropped_rescaled
#D:\Ebooks\projects\MSI project\data_set\distill_cropped_rescaled
os.chdir("D:\Ebooks\projects\MSI project\data_set\poliuted_cropped_rescaled")

#@@@@@@@interpolation@@@@@@@@
#[optional] flag that takes one of the following methods. INTER_NEAREST
#– a nearest-neighbor interpolation INTER_LINEAR – a bilinear interpolation (used by default)
#INTER_AREA – resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results.
#But when the image is zoomed, it is similar to the INTER_NEAREST method.
#INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood




for file_name in file_arr:
    #D:\Ebooks\projects\MSI project\data_set\poluted_croped
    #D:\Ebooks\projects\MSI project\data_set\distill_croped_new
    img = cv2.imread("D:\Ebooks\projects\MSI project\data_set\poluted_croped/" + file_name)
    resized = cv2.resize(img, (40 , 30), interpolation = cv2.INTER_AREA)
    cv2.imwrite("res_" + file_name , resized)




    
    
    
    

    

