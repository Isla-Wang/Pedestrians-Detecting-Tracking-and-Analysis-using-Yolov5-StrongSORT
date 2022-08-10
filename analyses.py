import cv2 as cv
import math
import numpy as np

def dectect_sorrounding(id_tracks, frame, id, status_list, groups):
    center_x = id_tracks[id][frame][0]
    center_y = id_tracks[id][frame][1]
    width = id_tracks[id][frame][2]
    height = id_tracks[id][frame][3]
    ids = id_tracks[:,frame]
    #Calculate area of bounding boxes
    area = width*height
    distance_threshold = 1.5*width
    #Find all cy cx and area for other ids in side the same frame
    status = 'Alone'
    for ppl in ids:
        if ppl[0] == center_x and ppl[1] == center_y and ppl[2] == width and ppl[3] == height:
            #Same persoon
            continue

        if ppl.all() == 0 or ppl.all() == -1:
            continue
        
        if ppl[2]*ppl[3] > area: 
            area_diff = ppl[2]*ppl[3] - area
            area_threshold = 0.31*ppl[2]*ppl[3]
        else: 
            area_diff = area - ppl[2]*ppl[3] 
            area_threshold = 0.31*area       

        if area_diff < area_threshold and math.hypot(center_x-ppl[0], center_y-ppl[1]) < distance_threshold:
            target_id = get_id(id_tracks, frame, ppl[0], ppl[1], ppl[2], ppl[3])
            
            status = 'Grouped'
            status_list[id] = status
            if status_list[target_id] == 'Grouped':
                for group in groups:
                    if target_id in groups[group] and id not in groups[group]:
                        groups[group].append(id)
            else:
                status_list[target_id] = status
                groups[len(groups)] = [id, target_id]
            return status, groups

    status_list[id] = status
    return status, groups

def draw_groups(groups, id_tracks, frame, img):
    group_info = {}
    for group_id, members in groups.items():
        position_list = []
        for id in members:
            position_list.append(id_tracks[id][frame])
        
        #Find top left point
        tl_x = np.min([i[0]-i[2]//2 for i in position_list])
        tl_y = np.min([i[1]-i[3]//2 for i in position_list])

        #Find bottom right point
        br_x = np.max([i[0]+i[2]//2 for i in position_list])
        br_y = np.max([i[1]+i[3]//2 for i in position_list])

        cv.rectangle(img, (int(tl_x), int(tl_y)), (int(br_x), int(br_y)), [0,0,255], 5)
        group_info[group_id] = get_info_for_group(tl_x, tl_y, br_x, br_y)

    return group_info

def get_info_for_group(tl_x, tl_y, br_x, br_y):
    
    cr_x = (br_x - tl_x)//2
    cr_y = (br_y - tl_y)//2

    return [cr_x, cr_y, br_x - tl_x, br_y - tl_y]

def group_analyses(colours, img, curr_group_info, prev_group_info, prev_ppl, curr_ppl, curr_groups, prev_groups):
    '''
    For people in a group, determine if its moving away from its group center
    For people not in a group, determine if its moving closer to other people.
    '''
    if len(prev_group_info) == 0:
        return
    ppl_in_group=[]
    for group, members in curr_groups.items():
        for ppl in members:
            ppl_in_group.append(ppl)
        curr_group_pos = curr_group_info[group]
        if group not in prev_group_info.keys():
            continue
        prev_group_pos = prev_group_info[group]
        for ppl in members:
            if ppl not in prev_ppl or ppl not in prev_groups[group]:
                continue
            if math.hypot(curr_group_pos[0]-curr_ppl[ppl][0], curr_group_pos[1]-curr_ppl[ppl][1])>math.hypot(prev_group_pos[0]-prev_ppl[ppl][0], prev_group_pos[1]-prev_ppl[ppl][1]):
                curr_x = curr_ppl[ppl][0]
                curr_y = curr_ppl[ppl][1]
                cv.putText(img, 'BYE!', (int(curr_x), int(curr_y)), 0, 1, colours[ppl],2)
                for rest_ppl in members:
                    if rest_ppl == ppl:
                        continue
                    rest_x = curr_ppl[rest_ppl][0]
                    rest_y = curr_ppl[rest_ppl][1]
                    cv.putText(img, 'BYE!', (int(rest_x), int(rest_y)), 0, 1, colours[rest_ppl],2)

    for key in curr_ppl.keys():
        if key in ppl_in_group or key not in prev_ppl:
            continue
        for key_2 in curr_ppl.keys():
            if key_2 not in prev_ppl:
                continue

            if ppl == key_2:
                continue

            if abs(curr_ppl[key][2]*curr_ppl[key][3] - curr_ppl[key_2][2]*curr_ppl[key_2][3]) < 0.35*curr_ppl[key][2]*curr_ppl[key][3]:
                if ((math.hypot(prev_ppl[key][0]-prev_ppl[key_2][0], prev_ppl[key][1]-prev_ppl[key_2][1])>math.hypot(curr_ppl[key][0]-curr_ppl[key_2][0], curr_ppl[key][1]-curr_ppl[key_2][1]))
                and (math.hypot(curr_ppl[key][0]-curr_ppl[key_2][0], curr_ppl[key][1]-curr_ppl[key_2][1]) < 3*curr_ppl[key][2])):
                    curr_x = curr_ppl[key][0]
                    curr_y = curr_ppl[key][1]

                    curr_x_2 = curr_ppl[key_2][0]
                    curr_y_2 = curr_ppl[key_2][1]
                    cv.putText(img, 'HELLO!', (int(curr_x), int(curr_y)), 0, 1, colours[key],2)
                    cv.putText(img, 'HELLO!', (int(curr_x_2), int(curr_y_2)), 0, 1, colours[key_2],2)

    return ppl_in_group


def entering_leaving_screen(prev_ppl, curr_ppl, colours, img):
    '''
    Check if this people moves inward/outward at the edge of the image

    Return Entering/Leaving
    '''
    img_height, img_width, img_color = img.shape
    for key in curr_ppl:
        if key in prev_ppl:
            curr_x = curr_ppl[key][0]
            curr_y = curr_ppl[key][1]
            width = curr_ppl[key][2]
            height = curr_ppl[key][3]

            prev_x = prev_ppl[key][0]
            prev_y = prev_ppl[key][1]

            if in_detect_area_x(curr_x, img_width) or in_detect_area_y(curr_y,img_height):
                if math.hypot(curr_x-img_width/2, curr_y-img_height/2) < math.hypot(prev_x-img_width/2, prev_y-img_height/2):
                    cv.putText(img, 'CHEESE', (int(curr_x-width//2), int(curr_y)-int(height//2)-10), 0, 1, colours[key],2)
                    cv.rectangle(img, (int(curr_x-width//2) , int(curr_y-height//2)), 
                              (int(curr_x+width//2) , int(curr_y+height//2)), colours[key], 15)
                else:
                    cv.putText(img, 'CYALL', (int(curr_x-width//2),int(curr_y)-int(height//2)-10), 0, 1, colours[key],2)
                    cv.rectangle(img, (int(curr_x-width//2) , int(curr_y-height//2)), 
                              (int(curr_x+width//2) , int(curr_y+height//2)), colours[key], 15)

    return


def in_detect_area_x(x,img_width):
    '''
    Detect if the given x is in the detect area
    Which determine its in the edge area of the image

    Return: True/False
    '''
    left_x_bondary = 0.2*img_width
    right_x_bondary = 0.8*img_width

    if x < left_x_bondary or x > right_x_bondary:
        return True

    return False

def in_detect_area_y(y,img_height):
    '''
    Detect if the given y is in the detect area
    Which determine its in the edge area of the image

    Return: True/False
    '''
    top_y_bondary = 0.2*img_height
    bottom_y_bondary = 0.8*img_height

    if y < top_y_bondary or y > bottom_y_bondary:
        return True
    
    return False

def get_id(id_tracks, frame, center_x, center_y, width, height):
    ids = id_tracks[:,frame,:]
    count = 0
    for id in ids:
        if center_x == id[0] and center_y == id[1] and width == id[2] and height == id[3]:
            return count
        count += 1

    return -1

