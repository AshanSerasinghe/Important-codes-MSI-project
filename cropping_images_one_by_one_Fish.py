import cv2
import os
from win32api import GetSystemMetrics

p = 0; q = 0

def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:      
        global p, q
        p = x; q = y
        return [x,y]
 
    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
        return [x,y]
 
    
    

pwd = os.getcwd()
file_arr = os.listdir(pwd)

# make a directory called eye
try:
    os.mkdir(pwd + '/' + 'eye_croped')
except:
    print("folder alrerady exixt")

#change the cuurren working directory
os.chdir(pwd + '/' + 'eye_croped')

file_num = 1
file_prefix = input("Input file prefix : ")#"distill"
width = 300
height = 300

start = int(input("Enter start image number : " ))

try:    
    for file_name in file_arr:
        if start<= file_num:
            if file_name.endswith('.jpg') or file_name.endswith('.JPG') or file_name.endswith('.png') or file_name.endswith('.PNG'):
                img = cv2.imread(pwd +'/'+ file_name)
                w_X = GetSystemMetrics(0) ; h_Y = GetSystemMetrics(1)
                original_width = int(img.shape[1])
                original_height = int(img.shape[0])

                fx_factor = round(w_X/original_width , 2); fy_factor = round(h_Y/original_height , 2)

                img_half = cv2.resize(img, None, fx = fx_factor, fy = fy_factor)

                # displaying the image
                cv2.imshow('image', img_half)
                
                cv2.setMouseCallback('image', click_event)

                # wait for a key to be pressed to exit
                cv2.waitKey(0)
            
                
                print(file_name , "  ", file_num)

                dims_1 = int(p/fx_factor); dims_2 = int(q/fy_factor); print(dims_1, dims_2)
                croped = img[(dims_2-int(height/2)):(dims_2+int(height/2)) , (dims_1-int(width/2)):(dims_1+int(width/2)), 0:3]
                
                cv2.imwrite( (file_prefix + str(file_num) + ".jpg"), croped)
                file_num = file_num+1
        else:
            file_num = file_num+1

except:
    print("check the file path\n")
    print("check whether there are files or folders other than images\n")
    