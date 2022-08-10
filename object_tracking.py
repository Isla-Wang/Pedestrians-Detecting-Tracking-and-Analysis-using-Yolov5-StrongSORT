import numpy as np
import cv2
import os
import math
import sys
import pandas as pd
import analyses
import video_crop as vc

'''
    Written by Shu Wang (z5211077). 
               Zhitong Chen (z5300114).
    Used to draw box and tracks using output of yolov5_Strongsort
'''

# helper function for calculating distance
def distance(x1, y1, x2, y2):
    # calculate distance between two points
    return math.hypot(x1 - x2, y1 - y2)

# the txt file has 10 columns, we need to use the first 6 columns
# they are:
# frame id left_top_x left_top_y width 
def detect_and_tracks(txt_filename, jpg_dir, out_dir, flag='all'):
    
    # we need to make id and frame start from 0
    # this is easier for following works
    
    # add an empty list in it, because there is no records for the frist frame
    all_label = [[]]    # change this if vhange .yaml file
    frame_label = []
    original_id_tracked = {}
    new_id = 0
    with open(txt_filename, 'r') as f:
        frame_check = 2 # change this if change .yaml file
        while True:
            line = f.readline().split()
            if not line:
                all_label.append(frame_label)
                break
            curr_frame = int(line[0])
            if curr_frame != frame_check:
                all_label.append(frame_label)
                frame_label = []
                frame_check += 1
            id = int(line[1])
            center_x = int(line[2]) + (int(line[4])//2)
            center_y = int(line[3]) + (int(line[5])//2)
            width = int(line[4])
            height = int(line[5])
            if id not in original_id_tracked.keys():
                original_id_tracked[id] = new_id
                frame_label.append([center_x, center_y, width, height, new_id])
                new_id += 1
            else:
                frame_label.append([center_x, center_y, width, height, original_id_tracked[id]])

    n_frames = len(all_label)   # number of frames
    n_ids = len(original_id_tracked)    # number of ids
    
    # go through all_label and find start and end of each id
    # the frame starts from 0
    starts = {} # key = id, value = first frame
    ends = {}   # key = id, value = last frame
    
    for i in range(n_frames):
        for j in range(len(all_label[i])):
            # loop in order to find the start of this id
            start_id = all_label[i][j][4]
            if start_id not in starts.keys():
                starts[start_id] = i
                
        # loop reversely to find the end of this id
        # since index of frames starts from 0
        # so need -1, or index will be out of range
        for j in range(len(all_label[n_frames - i - 1])):
            end_id = all_label[n_frames - i - 1][j][4]
            if end_id not in ends.keys():
                ends[end_id] = n_frames - i - 1
                
        # if both finished, stop the loop
        if len(starts) == n_ids and len(ends) == n_ids:
            break
    
    # then make a matrix
    id_tracks = np.zeros((n_ids, n_frames, 4))
    # 4 for [center_x, center_y, width, height]
    # first loop all_label and put current boxes in id_tracks
    # id_track has id starts from 0 and 
    frame = 0
    for frame in range(n_frames):
        for i in range(len(all_label[frame])):
            id = all_label[frame][i][4]
            center_x = all_label[frame][i][0]
            center_y = all_label[frame][i][1]
            width = all_label[frame][i][2]
            height = all_label[frame][i][3]
            # need id - 1 here because ndarray has index start from 0
            id_tracks[id][frame][0] = center_x
            id_tracks[id][frame][1] = center_y
            id_tracks[id][frame][2] = width
            id_tracks[id][frame][3] = height
    
    # set the distance threshold
    # small distance will abandon too many gaps
    # large distance could not detect errors
    # I choose 10
    max_dist = 10
    
    # then find gaps and fill them
    id = 0
    # use id to loop id_tracks
    while id < n_ids:
        frame = starts[id]  # starts from the first appeared frame
        while frame < ends[id]: # stop when reach the last appeared frame
            if id_tracks[id][frame].all() == 0: # if all zeros
                # (frame + 1) is the first frame of this gap
                # so frame is the frame before the gap. store it
                start_frame = frame - 1
                start_x = id_tracks[id][frame - 1][0]
                start_y = id_tracks[id][frame - 1][1]
                start_width = id_tracks[id][frame - 1][2]
                start_height = id_tracks[id][frame - 1][3]
                #print("start is: " + str(start_frame))
                # then find the end of gap
                while id_tracks[id][frame].all() == 0:
                    #print("hi")
                    frame += 1
                    
                # the loop above stop when id_tracks[id][frame].all() != 0
                # so it is the end frame. store it
                end_frame = frame
                #print("end is: " + str(end_frame))
                end_x = id_tracks[id][frame][0]
                end_y = id_tracks[id][frame][1]
                end_width = id_tracks[id][frame][2]
                end_height = id_tracks[id][frame][3]
                
                # then check if the gap exceeds the max distance
                gap_size = end_frame - start_frame
                if distance(start_x, start_y, end_x, end_y) > (gap_size * max_dist):
                    # if the gap is too large, then do not fill it
                    # just fill them with -1 to make a mark
                    j = start_frame + 1
                    while j < end_frame:
                        id_tracks[id][j][0] = -1
                        id_tracks[id][j][1] = -1
                        id_tracks[id][j][2] = -1
                        id_tracks[id][j][3] = -1
                        j += 1
                else:
                    # fill the gap by average difference
                    # calculate the average difference
                    #print(gap_size)
                    delta_x = (end_x - start_x) / gap_size
                    delta_y = (end_y - start_y) / gap_size
                    # print(start_x, end_x)
                    # use the mean of start_width and end_width to draw box
                    avg_width = (start_width + end_width) // 2
                    avg_height = (start_height + end_height) // 2
                     
                    curr_x = start_x 
                    curr_y = start_y
                    j = start_frame + 1
                    while j < end_frame:
                        curr_x = int(curr_x + delta_x)  # uodate values
                        curr_y = int(curr_y + delta_y)
                        id_tracks[id][j][0] = curr_x   # calculate x
                        id_tracks[id][j][1] = curr_y   # calculate y
                        id_tracks[id][j][2] = avg_width
                        id_tracks[id][j][3] = avg_height
                        j += 1
            frame += 1        
        id += 1
    
    '''
        I did not implement the check distance function
    '''
    
    # make unique colour for each id
    max_id = len(original_id_tracked)
    i = 0
    colours = []
    while i < max_id:
        c1 = int(np.random.choice(range(256)))
        c2 = int(np.random.choice(range(256)))
        c3 = int(np.random.choice(range(256)))
        colours.append((c1, c2, c3))
        i += 1

    # Now we could use id to get centers for each frame
    jpg_list = os.listdir(jpg_dir)   # get list of all images
    
    # now we need to draw rectangulars and lines use id_tracks
    frame = 0
    curr_ids = []
    prev_ppl = {}
    gorup_info_prev_frame = {}
    prev_groups = {}
    n_person = [0 for i in range(n_frames)] # number of persons each frame
    while frame < n_frames:
        curr_ppl = {}
        curr_groups = {}
        status_list = {}
        ids = id_tracks[:,frame,:]
        count = 0
        for ppl in ids:
            status_list[count] = 'Alone'
            if frame > 2:
                curr_ppl[count] = [ppl[0], ppl[1],ppl[2],ppl[3]]
            count += 1
        for key in curr_ppl.copy():
            if curr_ppl[key] == [0.0,0.0,0.0,0.0]:
                del curr_ppl[key]
        img_name = os.path.join(jpg_dir, jpg_list[frame]) # get path of image
        img = cv2.imread(img_name) # read image

        n_person_this_frame = 0
        # for this frame, go through each id
        id = 0
        while id < n_ids:
            if id_tracks[id][frame].all() != 0 and id_tracks[id][frame][0] != -1:
                # not empty and not gap 
                # draw a box and put id on the box
                center_x = int(id_tracks[id][frame][0])
                center_y = int(id_tracks[id][frame][1])
                width = int(id_tracks[id][frame][2])
                height = int(id_tracks[id][frame][3])
                
                status, curr_groups = analyses.dectect_sorrounding(id_tracks, frame, id, status_list, curr_groups)
                # draw box and id
                if flag == 'task_1' or flag == 'task_1_and_2' or flag == 'task_1_and_3' or flag == 'task_3' or flag == 'all':
                    cv2.rectangle(img, (center_x - width//2 , center_y - height//2), 
                                (center_x + width//2 , center_y + height//2), colours[id], 3)
                    #cv2.putText(img, str(id), (center_x, center_y - height//2), 0, 1, colours[id], 2) # put id
                    if flag != 'task_3': # we do not want trajectories for task 3
                        # also draw the track lines
                        if frame >= 2: # because no track in the first 2 frames
                            curr_frame = frame
                            prev_frame = curr_frame - 1
                            while ((id_tracks[id][prev_frame].all() != 0 and id_tracks[id][prev_frame][0] != -1)
                                and (id_tracks[id][curr_frame].all() != 0 and id_tracks[id][curr_frame][0] != -1)):
                                # draw lines
                                center_x_curr = int(id_tracks[id][curr_frame][0])
                                center_y_curr = int(id_tracks[id][curr_frame][1])
                                center_x_prev = int(id_tracks[id][prev_frame][0])
                                center_y_prev = int(id_tracks[id][prev_frame][1])
                                
                                cv2.line(img, (center_x_curr, center_y_curr), (center_x_prev, center_y_prev), colours[id], 3, 0)
                                
                                if prev_frame == 1:
                                    break # to avoid index error
                                curr_frame -= 1
                                prev_frame -= 1
                        
                # put the id into current id list
                if id not in curr_ids:
                    curr_ids.append(id)
                # then len(curr_id) is the current total number of persons
                
                # also update the number of persons in this frame
                n_person_this_frame += 1

            id += 1

        # get current total number of persons
        curr_total_n_person = len(curr_ids)

        if flag == 'task_3' or flag == 'task_1_and_3' or flag == 'all':
            # Draw bouding boxes for all the groups
            gorup_info_curr_frame = analyses.draw_groups(curr_groups, id_tracks, frame, img)
            #Detect people entering or leaving the screen
            analyses.entering_leaving_screen(prev_ppl, curr_ppl, colours, img)
            #Analyse group forming or breaking
            ppl_in_group = analyses.group_analyses(colours, img, gorup_info_curr_frame, gorup_info_prev_frame, prev_ppl, curr_ppl, curr_groups, prev_groups)
                
            #Detect people entering or leaving the screen
            analyses.entering_leaving_screen(prev_ppl, curr_ppl, colours, img)

            #Update prev_ppl and perv_groups list every 5 frames
            if frame > 2 and frame%5 == 0:
                prev_ppl = curr_ppl
                prev_groups = curr_groups
                gorup_info_prev_frame = gorup_info_curr_frame
        
        if flag == 'task_2' or flag == 'task_1_and_2' or flag == 'all':
            # add current total and number of person in this frame to the image (2.1 and 2.2)
            cv2.putText(img, "Current Total Number of Persons: " + str(curr_total_n_person), (1200, 40), 0, 1, (34, 139, 34), 2, cv2.LINE_AA)
            cv2.putText(img, "Number of Persons This Frame: " + str(n_person_this_frame), (1200, 80), 0, 1, (34, 139, 34), 2, cv2.LINE_AA)
            n_person[frame] = n_person_this_frame
        
        if flag == 'task_3' or flag == 'task_1_and_3' or flag == 'all':

            if ppl_in_group == None:
                ppl_in_group = []
            
            # add number of persons in group and alone on image
            cv2.putText(img, "Number of perople in group: " + str(len(ppl_in_group)), (1200, 120), 0, 1, (34, 139, 34), 2, cv2.LINE_AA)
            cv2.putText(img, "Number of perople alone: " + str(n_person_this_frame-len(ppl_in_group)), (1200, 160), 0, 1, (34, 139, 34), 2, cv2.LINE_AA)
        
        output_name = out_dir + jpg_list[frame]
        cv2.imwrite(output_name, img)
        # print process
        if (frame + 1) % 50 == 0:
            print(f"Finish making frame {frame + 1}/{len(jpg_list)}")
        frame += 1
    
    return all_label, n_person, id_tracks  # used for comparing

def count_person_and_compare(all_label, n_person, csv_path):
    '''
        Compare the number of persons in the output from yolov5 and strongsort
        with that after we filled gaps.
        We want to show the filling gap function do improve the performance.
    '''

    n_person_original = []  # number of persons in original txt
    diff_list = []  # list of difference
    for i in range(len(all_label)):
        n_person_original.append(len(all_label[i]))
        diff_list.append(int(n_person[i]) - int(len(all_label[i])))
    
    
    df_frame = {
        'original':  n_person_original,
        'after fill gaps': n_person,
        'difference': diff_list
    }
    
    df = pd.DataFrame(df_frame)
    df.to_csv(csv_path)
    
    return 


def view(test_out_dir, id_track, flag='all'):
    # do not include task 2
    if flag == 'task_1' or flag == 'task_3' or flag == 'task_1_and_3':
        dir = test_out_dir
        for image in os.listdir(dir):
            img = cv2.imread(os.path.join(dir, image))

            cv2.imshow('Frame', cv2.resize(img, None, fx=0.5, fy=0.5))
            cv2.waitKey(30)
    else:   # include task 2
        output_video = vc.sort_file_in_correct_order(os.listdir(test_out_dir))
        cv2.namedWindow("video")
        cv2.setMouseCallback("video", vc.crop_image)
        frame = 0
        for image in output_video:
            img = cv2.imread(os.path.join(test_out_dir, image))

            vc.run_video_with_crop(img, frame, id_track)
            frame += 1
            # press 'q' to end the video
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyWindow("video")
                print("Closing video")
                break
    return
