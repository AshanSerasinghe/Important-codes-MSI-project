import cv2
import os
from win32api import GetSystemMetrics
from glob import glob


bands = ['000nm' , '365nm' , '405nm', '473nm', '530nm', '575nm', '621nm', '660nm','735nm','770nm','830nm','850nm','890nm','940nm']
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


pwd_main = "F:\Data Sets\Rubber\Rubber\R"#os.getcwd()
folders = glob(pwd_main + "/*/", recursive = True)

main_folder_name = os.path.basename(pwd_main)

 # make a directory called eye
try:
    os.mkdir(pwd_main + '/' + 'eye_croped_samples')
except:
    print("folder alrerady exixt 1")


#pwd = folders[0]

for pwd in folders:
    file_arr = os.listdir(pwd)


    # make a directory called eye
    sub_folder_name = os.path.basename(os.path.dirname(pwd))
    try:
        os.mkdir(pwd_main + '/' + 'eye_croped_samples' + '/' + sub_folder_name )
    except:
        print("folder alrerady exixt 2")

    #change the cuurren working directory
    os.chdir(pwd_main + '/' + 'eye_croped_samples' + '/' + sub_folder_name)

    file_num = 1
    file_prefix = main_folder_name #input("Input file prefix : ")#"distill"
    width = 100 #300
    height = 100 #300

    start = 1 #int(input("Enter start image number : " ))


    try:  
        img = cv2.imread(pwd +'/'+ file_arr[8])
        w_X = GetSystemMetrics(0) ; h_Y = GetSystemMetrics(1)
        original_width = int(img.shape[1])
        original_height = int(img.shape[0])

        fx_factor = round(w_X/original_width , 2); fy_factor = round(h_Y/original_height , 2)

        img_half = cv2.resize(img, None, fx = fx_factor, fy = fy_factor)

        # if file_num == 1:
        # displaying the image
        cv2.imshow('image', img_half ) # pwd +'/'+ file_arr[8]
        
        cv2.setMouseCallback('image', click_event)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)  
        
        for file_name in file_arr:
            if start<= file_num:
                if file_name.endswith('.jpg') or file_name.endswith('.JPG') or file_name.endswith('.png') or file_name.endswith('.PNG'):
                    img = cv2.imread(pwd +'/'+ file_name, cv2.IMREAD_GRAYSCALE)
                    w_X = GetSystemMetrics(0) ; h_Y = GetSystemMetrics(1)
                    original_width = int(img.shape[1])
                    original_height = int(img.shape[0])

                    fx_factor = round(w_X/original_width , 2); fy_factor = round(h_Y/original_height , 2)

                    img_half = cv2.resize(img, None, fx = fx_factor, fy = fy_factor)

                    # if file_num == 1:
                    # displaying the image
                    # cv2.imshow('image', img_half ) # pwd +'/'+ file_arr[8]
                    
                    # cv2.setMouseCallback('image', click_event)

                    # wait for a key to be pressed to exit
                    # cv2.waitKey(0)
                
                    
                    print(file_name , " - ", file_name)

                    dims_1 = int(p/fx_factor); dims_2 = int(q/fy_factor); print(dims_1, dims_2)
                    croped = img[(dims_2-int(height/2)):(dims_2+int(height/2)) , (dims_1-int(width/2)):(dims_1+int(width/2))]
                    
                    cv2.imwrite( (str(file_name)), croped)
                    file_num = file_num+1
            else:
                file_num = file_num+1

    except:
        print("check the file path \n" , "problem at : ", sub_folder_name)
        print("check whether there are files or folders other than images\n")
