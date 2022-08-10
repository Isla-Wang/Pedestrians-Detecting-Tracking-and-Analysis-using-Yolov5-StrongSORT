from distutils import text_file
import cv2 as cv
import os
import math
import numpy as np

'''
    Written by Yunseok Jang z5286005 22 / Jul / 22
    For task 2.3 and 2.4.
'''

refPt = []
cropping = False

def run_video_with_crop(frame, frame_num, id_track):
    '''
    run video with crop
    '''
    #print('video running')
    global refPt
    # if there's a region chosen by the user, draw a rectangle on the edge of the region.
    if len(refPt) == 2:
        correct_refPt()
        cv.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 1)
    # display image
    cv.imshow("video", frame)

    # if the user has chosen the region or the region that's already chosen hasn't been removed, display the ROI(Region Of Interest).
    # press 's' to undo cropping and remove ROI.
    if len(refPt) == 2:
        correct_refPt()
        roi = frame[refPt[0][1]+1:refPt[1][1], refPt[0][0]+1:refPt[1][0]]
        num_of_ppl = count_people(id_track, frame_num)
        write_num_of_ppl(roi, num_of_ppl)
        cv.imshow("Region of interest", roi)
        if cv.waitKey(1) == ord('s'):
            refPt = []
            cv.destroyWindow("Region of interest")

def crop_image(event, x, y, flags, param):
    '''
    Whenever user click and drag a mouse,
    it'll run and collect the coordinates of two points(start and end of the drag).

    It's not meant to be used manually.
    It's called by "cv.setMouseCallback("video", crop_image)" automatically when the mouse have any action.
    '''
    global refPt, cropping
    # grab references to the global variables
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if event == cv.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
	# check to see if the left mouse button was released
    elif event == cv.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
        refPt.append((x, y))
        cropping = False

def count_people(id_track, frame):
    '''
    Count the number of people in the region use chose.

    parameter: list of files of pedestrian coordinates(label)
    return: the number of people(int)
    '''
    count = 0
    for id in id_track:
        if id[frame][0] != 0:
            cen_x = math.floor(int(id[frame][0]))
            cen_y = math.floor(int(id[frame][1]))
            if in_the_region(cen_x, cen_y):
                count += 1
    return count

def in_the_region(centre_x, centre_y):
    '''
    Check if the centre point is in the region user chose.

    parameter: x coordinate of the centre(int), y coordinate of the centre(int)
    return: boolean
    '''
    global refPt
    if refPt[0][0] < centre_x and refPt[1][0] > centre_x and \
           refPt[0][1] < centre_y and refPt[1][1] > centre_y:
        return True
    return False

def write_num_of_ppl(img, num):
    '''
    Write the number of people in the given img on top-left corner.

    paramter: image(2d array), number of people(integer)
    return: x
    '''
    font = cv.FONT_HERSHEY_SIMPLEX
    org = (0, 20)
    fontScale = 0.6
    color = (255, 0, 0) # Blue color in BGR
    thickness = 2 # Line thickness of 2 px
    cv.rectangle(img, (0,0), (250, 25), (0,0,0), -1)
    img = cv.putText(img, 'number of people: ' + str(num), org, font, 
                    fontScale, color, thickness, cv.LINE_AA)

def get_labels_as_iter(coordiante_path):
    '''
    It takes label text file, break labels by frames and create 3d array that contains labels.
    Return the iterable of that array.

    parameter: empty array, path to coordiante(string)
    return: iterable
    '''
    labels = []
    frame_num = 0
    f = open(coordiante_path, "r")
    for label in f:
        label = label.split()
        if len(labels) == 0:
            labels = [[label[1:6]]]
            frame_num = int(label[0])
        elif int(label[0]) != frame_num:
            labels.append([label[1:6]])
            frame_num = int(label[0])
        else:
            labels[int(label[0])-2].append(label[1:6])
    return iter(labels)


def correct_refPt():
    '''
    If will correct reference points.
    With this function, refPt[0] will always be smaller than refPt[1],
    which means refPt[0] points top-left and pefPt[1] points bottom-right.

    parameter: x
    return: x
    '''
    global refPt
    if refPt[0][0] > refPt[1][0]:
        tmp = (refPt[0][0], refPt[1][1])
        refPt[0] = (refPt[1][0], refPt[0][1])
        refPt[1] = tmp
    if refPt[0][1] > refPt[1][1]:
        tmp = (refPt[1][0], refPt[0][1])
        refPt[0] = (refPt[0][0], refPt[1][1])
        refPt[1] = tmp

def sort_file_in_correct_order(files):
    '''
    Using .sort() only will cause a problem of having file name 10 before 9.
    Therefore in this function, we first sort it in alphabetical order first and sort it by length.
    Since python sort is stable, it will return files in numerical order (1, 2, 3, ..., 9, 10, etc)

    parameter: list of strings
    return: list of strings sorted in numerical order (if strings contain sequence of numbers)
    '''
    files.sort()
    return sorted(files, key=len)



#path = '../yolov5s_test_video_and_labels/exp20/'
#video_crop(path+"test_1.mp4", path+"test_1.txt")
