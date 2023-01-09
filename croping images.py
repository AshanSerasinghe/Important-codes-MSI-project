import cv2
import os

img = cv2.imread("D:\Ebooks\projects\Polute water detection\data_set\Original/20210911_200109.jpg")
#cv2.imshow("Image1", img)

croped = img[2200:3000 , 1300:2220]
cv2.imshow("Croped", croped)

file_arr = os.listdir("D:\Ebooks\projects\Polute water detection\data_set\Original")

#change the cuurren working directory
os.chdir("D:\Ebooks\projects\Polute water detection\data_set\Croped")

file_num = 1
file_prefix = "distill"

try:
    
    for file_name in file_arr:
        img = cv2.imread("D:\Ebooks\projects\Polute water detection\data_set\Original/" + file_name)
        croped = img[2200:3000 , 1300:2220]
        cv2.imwrite( (file_prefix + str(file_num) + ".jpg"), croped)
        file_num = file_num+1
        
except:
    print("check the file path\n")
    print("check whether there are files or folders other than images\n")
    
