import numpy as np
from PIL import ImageGrab
import cv2
import time
from numpy import ones,vstack
from numpy.linalg import lstsq
from directkeys import PressKey,ReleaseKey, W, A, S, D, SPACEBAR
from getkeys import key_check
from statistics import mean
import random

def determine_action(m1,m2):
    if m1==0 or m2==0:
        PressKey(S)
        print('Reverse Top')
        time.sleep(abs(m1)+abs(m2))
        ReleaseKey(S)
    elif -0.15<=m1<0 and -0.15<=m2<0:
        #right()
        print('Right {} {}'.format(m1,m2))
        PressKey(S)
        time.sleep(abs(max(m1,m2)))
        ReleaseKey(S)
        #ReleaseKey(D)
    elif -0.8<=m1<0.15 and -0.8<=m2<0.15:
        right()
        print('Right {} {}'.format(m1,m2))                      
        PressKey(W)
        time.sleep(abs(min(m1,m2)))
        ReleaseKey(W)
        ReleaseKey(D)
    elif 0.15>=m1>0 and 0.15>=m2>0:
        #left()
        print('Left {} {}'.format(m1,m2))
        PressKey(S)
        time.sleep(abs(max(m1,m2)))
        ReleaseKey(S)
        #ReleaseKey(A)
    elif 0.8>m1>0.15 and 0.8>m2>0.15:
        left()
        print('Left {} {}'.format(m1,m2))
        PressKey(W)
        time.sleep(abs(min(m1,m2)))
        ReleaseKey(W)
        ReleaseKey(A)
    elif (m1<=0 and m2>=0) or (m1>=0 and m2<=0):
            straight()
            print('Straight 2 {} {}'.format(m1,m2))
            time.sleep(abs(min(m1,m2)))

def no_keys():
    ReleaseKey(W)
    ReleaseKey(S)
    ReleaseKey(A)
    ReleaseKey(D)

def straight():
    ReleaseKey(S)
    PressKey(W)

def left():
    ReleaseKey(D)
    PressKey(A)

def right():
    ReleaseKey(A)
    PressKey(D)

def reverse():
    ReleaseKey(W)
    PressKey(S)

def draw_green_lines(img, l1, l2):
    cv2.line(img, (l1[0],l1[1]), (l1[2],l1[3]), color=[0,255,255], thickness=4)
    cv2.line(img, (l2[0],l2[1]), (l2[2],l2[3]), color=[0,255,255], thickness=4)
    return img

def decide_lanes(img, lines):
    try:
        ys = []  
        for i in lines:
            for ii in i:
                ys += [ii[1],ii[3]]
        min_y = min(ys)
        max_y = 600
        new_lines = []
        line_dict = {}

        for idx,i in enumerate(lines):
            for xyxy in i:
                x_coords = (xyxy[0],xyxy[2])
                y_coords = (xyxy[1],xyxy[3])
                A = vstack([x_coords,ones(len(x_coords))]).T
                m, b = lstsq(A, y_coords)[0]

                x1 = (min_y-b) / m
                x2 = (max_y-b) / m

                line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]]
                new_lines.append([int(x1), min_y, int(x2), max_y])

        final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]
            
            if len(final_lanes) == 0:
                final_lanes[m] = [ [m,b,line] ]
            else:
                found_copy = False

                for other_ms in final_lanes_copy:
                    if not found_copy:
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [ [m,b,line] ]

        line_counter = {}

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]

        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) 

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [l1_x1,l1_y1,l1_x2,l1_y2], [l2_x1,l2_y1,l2_x2,l2_y2], lane1_id, lane2_id
    except Exception as e:
        pass

def draw_white_lines(image, lines):
    for line in lines:
        coords = line[0]
        cv2.line(image, (coords[0], coords[1]), (coords[2], coords[3]), color=[255,0,0], thickness=1)

def roi(original_image):
    vertices = np.array([ [50,450], [200,250], [600,250], [750,450] ], np.int32)
    #vertices = np.array([ [100,450], [170,250], [570,250], [700,450] ], np.int32)
    mask = np.zeros_like(original_image)
    cv2.fillPoly(mask, [vertices], 255)
    masked_image = cv2.bitwise_and(original_image, mask)
    return masked_image

def process_image(image):
    try:
        original_image = image
        processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.Canny(processed_image, threshold1=100, threshold2=500)
        can_image = processed_image
        processed_image = cv2.GaussianBlur(processed_image, (5,5), 3)
        processed_image = roi(processed_image)
        
        lines = cv2.HoughLinesP(processed_image, 1, np.pi/180, 180, 20, 15)
        draw_white_lines(processed_image, lines)
        l1,l2,m1,m2 = decide_lanes(original_image, lines)        
        original_image = draw_green_lines(original_image, l1, l2)
        determine_action(m1,m2)

    except:
        pass

    return processed_image, original_image, can_image

def main():

    for i in range(4):
        print(4-i)
        time.sleep(1)

    paused = False
    last_time = time.time()

    while True:
        keys = key_check()

        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                ReleaseKey(S)
                time.sleep(1)

        if not paused:
            screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))

            processed_image, original_image, can_image = process_image(screen)

            cv2.imshow('Autopilot B&W', can_image)
            cv2.imshow('Autopilot Color', cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            if cv2.waitKey(25) & 0xFF==ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    main()