import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import glob
from tracker import*


model=YOLO('yolov8s.pt')
tracker=Tracker()
counter=[]
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")


list1=[]  


def object(i):
    results=model.predict(i)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            list.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(list)
#    list1=[]
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
        cv2.rectangle(i,(x3,y3),(x4,y4),(0,0,255),2)
        cvzone.putTextRect(i,f'{id}',(x3,y3),2,2)
        list1.append(id)
    print(len(list1))        
            
            



path=r'C:\Users\freed\Downloads\countcars-main\countcars-main\images\*.*'
for file in glob.glob(path):
    img=cv2.imread(file)
    img=cv2.resize(img,(1020,500))
    object(img)
    
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
