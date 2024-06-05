import cv2

from util import get_parking_spot_box
from util import check
import numpy as np

def calc_diff(im1,im2):
    return np.abs(np.mean(im1)-np.mean(im2))




masks='/Users/shrishaa/Documents/computer vision/Car_parking/data/Parking Space Counter Mask.png'
vid_path='/Users/shrishaa/Documents/computer vision/Car_parking/data/Parking spot detection tutorial/parking_1920_1080_loop.mp4'

mask=cv2.imread(masks,0)
cap=cv2.VideoCapture(vid_path)


connected_components=cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S)

spots=get_parking_spot_box(connected_components)

previousframe=None

spots_status=[None for j in spots]
diffs=[None for j in spots]
ret=True
step=30
frame_num=0

while ret:
    ret,frame=cap.read()

    if frame_num%step==0:#we add frame num so that the classifier does not classify for every single frame, the classifier classifys after 30 frames

        for spot_idx,spot in enumerate(spots):
            x1,y1,w,h=spot
            spot_crop=frame[y1:y1+h,x1:x1+w,:]
            spot_status=check(spot_crop)
            spots_status[spot_idx]=spot_status
        

    if frame_num%step==0:
        previousframe=frame.copy()



    for spot_idx,spot in enumerate(spots):
        spot_status=spots_status[spot_idx]
        x1,y1,w,h=spots[spot_idx]
        if spot_status:
            cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),2)

        else:
            cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,0,255),2)


    cv2.rectangle(frame,(80,20),(550,80),(0,0,0),-1)
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.imshow('frame',frame)

    if cv2.waitKey(25) & 0xFF==ord('q'):
        break

    frame_num+=1



cap.release()
cv2.destroyAllWindows()