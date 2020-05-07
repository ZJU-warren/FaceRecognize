from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import sys; sys.path.append('../')
import cv2
from B_Detect_Align.DetectProxy import DetectProxy
from C_Recognize.RecognizeProxy import RecognizeProxy
import DataSetLink as DLSet
import pathlib



# load all data
folder_path_set = list(pathlib.Path(DLSet.AppSet_link).iterdir())
images = []
labels = []


dp = DetectProxy()
rp = RecognizeProxy(1)

# visit all the folder
for folder_path in folder_path_set:
    # obtain the label
    label = str(folder_path).split('/')[-1]
    
    print(label)
    # load img and append it
    img_path_set = folder_path.iterdir()
    for img_path in img_path_set:
        # detect face
        img = cv2.imread(str(img_path))
        draw = dp.detect(img)
        draw = cv2.resize(draw, (150, 150))
                
        images.append(rp.get_vec(draw))
        labels.append(label)

print('load all data <<<<<<<<<<<<<<<<<<<<<-----------')


# capture, dectect and recognize
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))



pause = False
draw = None
while True:
    if not pause:
        # print('capture new')
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
    
        # detect whether there is a face
        flag, draw = dp.mark(frame)
        # print('flag:', flag)
        
        # notice
        cv2.putText(draw,
                    "Press Q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
        
        # face exist
        if flag:
            pause = True
            
            # get crop
            crop, best_rectangle = dp.app_detect(frame)
            crop = cv2.resize(crop, (150, 150))
           
            vc = rp.get_vec(crop)
            idx = 0
            min_d = 1000
            for i in range(len(images)):
                d = rp.get_distance(images[i], vc, True)
                print('label:%s, d=%lf'%(labels[i], d))
                if min_d > d:
                    min_d = d
                    idx = i
                    
            cv2.putText(draw,
                    labels[idx],
                    (int(best_rectangle[0]) + 25, int(best_rectangle[3]) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
            
    cv2.imshow('frame', draw)
    
    press = cv2.waitKey(1) & 0xFF
    if press == ord(' '):
     	pause = False
     	# print('continue')
     	
    if press == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
