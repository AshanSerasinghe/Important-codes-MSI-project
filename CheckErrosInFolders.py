# check each sample folder for erros in  images
# eg:- S1 <-- main folder 
#       30_03_2022 10-30-16 <-- sample folder 



import cv2
import os
from win32api import GetSystemMetrics
from glob import glob



pwd_main = os.getcwd()
folders = glob(pwd_main + "/*/", recursive = True)

main_folder_name = os.path.basename(pwd_main)
bands = ['000nm' , '365nm' , '405nm', '473nm', '530nm', '575nm', '621nm', '660nm','735nm','770nm','830nm','850nm','890nm','940nm']

for pwd in folders:
    file_arr = os.listdir(pwd)

    # make a directory called eye
    sub_folder_name = os.path.basename(os.path.dirname(pwd))
    file_num = 1
    start = 1
    
    if len(file_arr) !=14:
            print("===============>>> Invalid number of images in the " + main_folder_name + " " + sub_folder_name )
    else:
        for file_name in file_arr:
            if start<= file_num:
                if file_name.endswith('.jpg') or file_name.endswith('.JPG') or file_name.endswith('.png') or file_name.endswith('.PNG'):
                    if file_name != bands[file_num-1]+'.png':
                        print("file name mismatch" + main_folder_name + " " + sub_folder_name)
                        break
                    else:
                        try:
                            img = cv2.imread(pwd +'/'+ file_name)
                        except:
                            print("image cannot be read" + main_folder_name + " " + sub_folder_name)
                            break
                else:
                    print("invalid file type" + main_folder_name + " " + sub_folder_name)
                    break
            file_num+=1


