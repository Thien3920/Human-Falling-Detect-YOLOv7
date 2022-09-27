"""
This script to create .csv videos frames action annotation file.
- It will play a video frame by frame control the flow by [a] and [d]
 to play previos or next frame.
- Open the annot_file (.csv) and label each frame of video with number
 of action class.
"""

import os
import cv2
import numpy as np
import pandas as pd


class_names = ['Standing','Stand up', 'Sitting','Sit down','Lying Down','Walking','Fall Down']

video_folder = '/media/ngocthien/DATA/DO_AN_TOT_NGHIEP/DATA/Fall_detection/chute21'
annot_file_2 = '/media/ngocthien/DATA/DO_AN_TOT_NGHIEP/DATA/Fall_detection/chute21.csv'


video_list = sorted(os.listdir(video_folder))
cols = ['video', 'frame', 'label']
df = pd.DataFrame(columns=cols)


for index_video_to_play in range(len(video_list)):

    video_file = os.path.join(video_folder, video_list[index_video_to_play])
    print(os.path.basename(video_file))

    cap = cv2.VideoCapture(video_file)
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video = np.array([video_list[index_video_to_play]] * frames_count)
    frame_1 = np.arange(1, frames_count + 1)
    label = np.array([0] * frames_count)

    k = 0
    i = 0
    while True:
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        label[i-1] =k
        if ret:

            cls_name = class_names[k]

            frame = cv2.resize(frame, (0, 0), fx=2, fy=2)
            frame = cv2.putText(frame, 'Video: {}     Total_frames: {}        Frame: {}       Pose: {} '.format(video_list[index_video_to_play],frames_count,i+1, cls_name,),
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            #Show button list
            frame = cv2.putText(frame,'Back:  a',(10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            frame = cv2.putText(frame,'Dung:   0', (10, 300),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame = cv2.putText(frame,'Dung day:    1', (10, 330),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame = cv2.putText(frame,'Ngoi:    2', (10, 360),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame = cv2.putText(frame,'Ngoi xuong: 3', (10, 390),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame = cv2.putText(frame,'nam:   4', (10, 420),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame = cv2.putText(frame,'Di:   5', (10, 450),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame = cv2.putText(frame,'Nga:  6', (10, 480),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.imshow('frame', frame)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('0'): #Đứng
                i += 1
                k=0
            elif key == ord('1'): #Đứng dậy
                i += 1
                k=1
            elif key == ord('2'):#Ngồi
                i += 1
                k=2
            elif key == ord('3'):#Ngồi xuống
                i += 1
                k = 3
            elif key == ord('4'):#Nằm
                i += 1
                k = 4
            elif key == ord('5'):#Đi
                i += 1
                k = 5
            elif key == ord('6'):#Ngã
                i += 1
                k = 6
            elif key == ord('a'):#Trở lại
                i -= 1
        else:
            break
        if i <1:
            i =0
    rows = np.stack([video, frame_1, label], axis=1)
    df = df.append(pd.DataFrame(rows, columns=cols),ignore_index=True)
df.to_csv(annot_file_2,index=False)
cap.release()
cv2.destroyAllWindows()