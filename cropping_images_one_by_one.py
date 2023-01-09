import cv2
import os

#img = cv2.imread("D:\Ebooks\projects\Polute water detection\data_set\Original/20210911_200109.jpg")
#cv2.imshow("Image1", img)

#croped = img[2200:3000 , 1300:2220]
#cv2.imshow("Croped", croped)

file_arr = os.listdir("D:\Ebooks\projects\MSI project\data_set\poluted_original")

#change the cuurren working directory
os.chdir("D:\Ebooks\projects\MSI project\data_set\poluted_croped")

file_num = 1
file_prefix = "distill"
width = 800
height = 600


try:
    
    for file_name in file_arr:
        img = cv2.imread("D:\Ebooks\projects\MSI project\data_set\poluted_original/" + file_name)
        print(file_name , "  ", file_num)
        dims_1 = int(input("Enter center pixcel X axis:" ))
        dims_2 = int(input("Enter center pixcel Y axis:" ))
        croped = img[ (dims_2-300):(dims_2-300+height) , (dims_1-400):(dims_1-400+width)]
        cv2.imwrite( (file_prefix + str(file_num) + ".jpg"), croped)
        file_num = file_num+1
        
except:
    print("check the file path\n")
    print("check whether there are files or folders other than images\n")
    
