import cv2
import numpy as np
from util import get_parking_spots_bboxes
from util import empty_or_not



def calc_diff (im1, im2) : 
    return np.abs(np.mean(im1) - np.mean(im2)) 

mask = './mask_1920_1080.png'
video_path = './parking_1920_1080_loop.mp4'


cap = cv2.VideoCapture(video_path)

msk = cv2.imread(mask,0)

connected_comps = cv2.connectedComponentsWithStats(msk, 4, cv2.CV_32S)

spots = get_parking_spots_bboxes(connected_comps)


spots_status = [None for j in spots]
diffs = [None for j in spots]
prev_frame = None
frame_nb =0 
ret = True
step = 30

while ret :
    ret ,frame = cap.read()

    if frame_nb % step == 0 and prev_frame is not None :
        for spot_index,spot in enumerate(spots) :
            x1,y1,w,h = spot

            spot_crop = frame[y1 : y1 + h, x1 : x1 + w, :]

            diffs[spot_index] = calc_diff(spot_crop,prev_frame[y1 : y1 + h, x1 : x1 + w, :])



    if frame_nb % step == 0 :
        if prev_frame is None : 
            arr = range(len(spots))
        else : 
            arr = [j for j in np.argsort(diffs) if diffs[j] > 0.4][::-1]
        for spot_index in arr :
            spot = spots[spot_index]
            x1,y1,w,h = spot

            spot_crop = frame[y1 : y1 + h, x1 : x1 + w, :]

            spot_status =empty_or_not(spot_crop)

            spots_status[spot_index] = spot_status

        prev_frame = frame.copy()

    for spot_index,spot in enumerate(spots) :
        spot_status = spots_status[spot_index]
        x1,y1,w,h = spots[spot_index]

        if spot_status :
            frame = cv2.rectangle(frame,(x1, y1),(x1 + w , y1 + h),(0,255,0),2 )
        else : 
            frame = cv2.rectangle(frame,(x1, y1),(x1 + w , y1 + h),(0,0,255),2 )

    cv2.rectangle(frame , (80,20),(550,80) , (0,0,0),-1)
    cv2.putText(frame,'Free  {} / {}'.format(str(sum(spots_status)),str(len(spots_status))),(100,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q') : 
        break

    frame_nb += 1






cap.release()
cv2.destroyWindow()