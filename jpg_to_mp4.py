import cv2 as cv
import os

'''
    Change jpg files to mp4 videos.
    Written by Shu Wang (z5211077).
'''


def jpg_to_mp4(in_path, out_path):
    '''
        Change .jpg files from a certain directory to .mp4 video
        in_path: directory of input images
        out_path: directory of the output video
    '''

    jpg_list = os.listdir(in_path)
    
    img = cv.imread(in_path + jpg_list[0]) # read the first image
    height, width, layers = img.shape   # and get the shape of it
    
    fps = 30
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v') # write in .mp4 format
    videowriter = cv.VideoWriter(out_path, fourcc, fps, (width, height))  # create a video writer object
    
    for i in range(len(jpg_list)):
        path = in_path + str(jpg_list[i])   # get path of image 
        frame = cv.imread(path) # read the image with image name
        videowriter.write(frame) # write to video

    videowriter.release()

