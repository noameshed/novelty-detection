# This script takes an input video and saves a section of the individual frames

import cv2
import numpy as np
import os

loadpath = 'D:/noam_/Cornell/CS7999/Macaulay Library Birds - Videos/'
savepath = 'D:/noam_/Cornell/CS7999/Macaulay Library Birds - Frames/'
all_vids = os.listdir(loadpath)

# Capture video frames
video = all_vids[0]
cap = cv2.VideoCapture(loadpath + video)
print(video)
# Check if camera opened successfully 
if (cap.isOpened()==False):
    print("Error opening video stream or file")

# Read until video is completed
count = 0
while(cap.isOpened()) and count < 250:
    # capture frame by frame
    ret, frame = cap.read()

    if ret == True and count%5 == 0:
        name = video[:-4]+'_'+str(count)+'.jpg'
        print(name)
        cv2.imwrite(savepath + name, frame)
    
    count += 1

# When everything done, release the video capture object
cap.release()

# Close all frames
cv2.destroyAllWindows()