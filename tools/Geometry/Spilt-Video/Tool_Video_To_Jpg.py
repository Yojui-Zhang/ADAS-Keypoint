import cv2
import glob
import os
# import math
# from tqdm import tqdm


# pip3 install opencv-python

"""
def process_image(img, min_side = 608):
    size = img.shape
    h, w = size[0], size[1]
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv2.resize(img, (new_w, new_h),cv2.INTER_AREA) # 
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side-new_h)//2, (min_side-new_h)//2, (min_side-new_w)//2 + 1, (min_side-new_w)//2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)//2 + 1, (min_side-new_h)//2, (min_side-new_w)//2, (min_side-new_w)//2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)//2, (min_side-new_h)//2, (min_side-new_w)//2, (min_side-new_w)//2
    else:
        top, bottom, left, right = (min_side-new_h)//2 + 1, (min_side-new_h)//2, (min_side-new_w)//2 + 1, (min_side-new_w)//2
    pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0]) 
    return pad_img
"""

"""
	inputPath => 要切割的影片路徑
	outputPath => 輸出圖的資料夾路徑
	FrameCut => 幾個Frame切割一次
	FrameName => 每張圖片的前綴名稱
		ex: 如果FrameName = 'S3_Urban1'，每張圖片名稱就是 S3_Urban1_0001、S3_Urban1_0002...... 到最後一張
"""
inputPath = '../Video/output.avi' 
outputPath = '../Picture/'
FrameCut = 3  #
FrameName = 'RD-2' 
                       # [S3_Urban1]_0001

# 
if os.path.isfile(inputPath):
    print("\nVideo path => ",inputPath)
else:
    print("\n",inputPath," Not found")
    exit()
    
#   
if os.path.isdir(outputPath):
    print("Save path =>",outputPath)
else:
    os.mkdir(outputPath)
    print("Not found the path => ",outputPath,",already crtate it")    
    
files = os.path.join(inputPath)
files_grabbed = []
files_grabbed.extend(sorted(glob.iglob(files)))


for videoId in range(len(files_grabbed)):
	print("\nStart cut video <",files_grabbed[videoId],">,cut every ",FrameCut," Frames")
	raw = cv2.VideoCapture(files_grabbed[videoId])
	fIndex = 1
	fCount = 0

	while 1:
    # 
		ret,frame = raw.read()
		fCount += 1
		if (ret == True) :
			if (fCount % FrameCut) == 0:  # 
				#img_pad = process_image(frame, min_side = 608)
                #
				cv2.imwrite('%s/%s_%04d.jpg' % (outputPath,FrameName,fIndex),frame)
				fIndex += 1
		else:
			print("Finish\n")
			break
