import cv2
import numpy as np
import pickle
from PIL import Image
import os
import gc
import pathlib


def writeOpticalFlow(path, outpath, filename,w,h,c):
    count=0
# 	try:
    cap = cv2.VideoCapture(path+'/'+filename)
    ret, frame1 = cap.read()

    if not frame1.any():
        return count
    filename = filename.split('.')[0]
    
    frame1 = cv2.resize(frame1, (w,h))
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    folder_u = outpath + '/u/'+filename
    pathlib.Path(folder_u).mkdir(parents=True,exist_ok=True)
    folder_v = outpath + '/v/'+filename
    pathlib.Path(folder_v).mkdir(parents=True,exist_ok=True)

    while(1):
        ret, frame2 = cap.read()
    
        if not frame2.any():
            break
        count+=1
        if count%5==0:
            print (filename+':' +str(c)+'-'+str(count))
        
        frame2 = cv2.resize(frame2, (w,h))
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
        vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
        horz = horz.astype('uint8')
        vert = vert.astype('uint8')
        
        frame_idx = 'frame'+ str(count).zfill(6)
        cv2.imwrite(os.path.join(folder_u, frame_idx +'.jpg'),horz,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
        cv2.imwrite(os.path.join(folder_v, frame_idx +'.jpg'),vert,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        prvs = next
    
    cap.release()
    cv2.destroyAllWindows()
    return count
    # except (Exception):
    # 	print ("Error in writeOpticalFlow")
    # 	return count



        

    

