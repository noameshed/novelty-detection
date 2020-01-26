# This script takes an input video and saves a section of the individual frames
import cv2
import numpy as np
import os


def save_frames(video, loadpath, savepath):
    '''
    Saves every 5th frame of the video as a jpg at the savepath location
    Arguments
        video: the name of the mp4 video
        loadpath: the location of the original video
        savepath: the location where the frames will be saved
    '''
    cap = cv2.VideoCapture(loadpath+video)

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

if __name__ == "__main__":
    loadpath = 'D:/noam_/Cornell/CS7999/Macaulay Library Birds - Videos/'
    savepath = 'D:/noam_/Cornell/CS7999/Macaulay Library Birds - Frames/'

    # Create a map of video IDs to common names
    id2name = {}
    with open('vids_to_split.txt', 'r') as f:
        for line in f:
            splitline = line.strip().split('\t')
            ID = int(splitline[0])
            name = splitline[2]
            id2name[ID] = name

    all_vids = os.listdir(loadpath)
    # Capture video frames of each video
    for video in all_vids:
        ID = int(video[:-4])
        name = id2name[ID]
        print(video) 
        # Create folder to save video frames
        try:
            os.mkdir(savepath + name)
        except:
            pass

        os.mkdir(savepath + name + '/' + str(ID))
        save_frames(video, loadpath, savepath + name + '/' + str(ID)+'/')