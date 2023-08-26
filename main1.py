import cv2
from yolo_segmentation import YOLOSEG
import cvzone
from tracker import*
import numpy as np
import glob
from tracker import*
ys = YOLOSEG("yolov8s-seg.pt")


tracker=Tracker()


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
list1=[]


count=0
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

def object(img):
    overlay = img.copy()
    alpha = 0.1

    bboxes, classes, segmentations, scores = ys.detect(img)
    bbox_idx=tracker.update(bboxes)

    for bbox1,seg,l in zip(bbox_idx,segmentations,classes):
        k=class_list[l]
        if 'car' in k:
           x3,y3,x4,y4,id=bbox1
           cx=int(x3+x4)//2
           cy=int(y3+y4)//2
 
           cv2.rectangle(img, (x3, y3), (x4, y4), (255, 0, 0), 3)    
           cv2.polylines(img, [seg], True, (0, 0, 255), 4)
           cv2.circle(img,(cx,cy),4,(255,0,255),-1)
           cv2.fillPoly(overlay, [seg], (0,0,255))
           cv2.addWeighted(overlay, alpha, img, 1 - alpha, 2, img)
           cvzone.putTextRect(img, f'{id}', (x3,y3),1,1)
           list1.append(id)
    print(len(list1))        
            
path=r'C:\Users\freed\Downloads\countcars-main\countcars-main\images\*.*'
for file in glob.glob(path):
    img=cv2.imread(file)
    img=cv2.resize(img,(1020,500))
    object(img)
#    print(len(list1))
    
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   

