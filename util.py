import pickle
from skimage.transform import resize
import numpy as np
import cv2


EMPTY=True
NOT_EMPTY=False
model=pickle.load(open("classifier.p","rb"))
def check(spot):
    flat_data = []

    img_resized = resize(spot, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = model.predict(flat_data)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY
    
def get_parking_spot_box(connected_components):
    (total_labels,label_id,values,centroid)=connected_components

    slots=[]
    coef=1
    for i in range(1,total_labels):
        x1=int(values[i,cv2.CC_STAT_LEFT]*coef)
        y1=int(values[i,cv2.CC_STAT_TOP]*coef)
        w=int(values[i,cv2.CC_STAT_WIDTH]*coef)
        h=int(values[i,cv2.CC_STAT_HEIGHT]*coef)

        slots.append([x1,y1,w,h])
    return slots