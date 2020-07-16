#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 22:34:32 2020

@author: djkim9031
"""

import cv2
import numpy as np
import imutils
from djitellopy import Tello

import threading
from threading import Thread
import time
import keyboard


exitFlag = 0

whT = 320
width = 800
height = 600
#confThresh = 0.3
#nmsThresh = 0.5 #The lower it is, the stricter its standard is, hence less bboxes

classFile = 'coco_classes.txt'
classNames = []
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)
#print(len(classNames))
    
modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfig,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layerNames = net.getLayerNames()
outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]


me = Tello()
me.connect()
me.forward_backward_velocity = 0 
me.left_right_velocity = 0
me.up_down_velocity = 0
me.yaw_velocity = 0
me.speed = 0
    
print(me.get_battery())
    
me.streamoff()
me.streamon()

val = 0


def findObjects(outputs, img):
    height,width,_= img.shape
    bbox = []
    confs = []
    class_ids = []
    
    for output in outputs:
        for detect in output:
            scores = detect[5:] #The first 5 values are x,y,w,h,confidence - the remainders are scores for 80 classes
            class_id = np.argmax(scores) #Index for maximum value in the array
            conf = scores[class_id]
            if conf > 0.3:
                cx,cy = int(detect[0]*width),int(detect[1]*height)
                w,h = int(detect[2]*width), int(detect[3]*height)
                x,y = int(cx-w/2),int(cy-h/2)
                bbox.append([x,y,w,h])
                confs.append(float(conf))
                class_ids.append(class_id)
                
    #print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, 0.5, 0.4) #Returning indices of bbox to keep
    #print(len(indices))
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(img,f'{classNames[class_ids[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        #cv2.putText(img,f'Billboard {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

#class myThread (threading.Thread):
#    global val
#    def __init__(self,code):
#        threading.Thread.__init__(self)
#        self.code = code

#    def run(self):   
#        if self.code == "motion":
#            self.motion()

#    def motion(self):
#        print(val)
#        if val=='t':
#            me.takeoff()
#            time.sleep(3)
#        elif val=='l':
#            a=me.land()
#            print("Tello Land:",a)
        
class DroneMotion(object):
    global val
    def __init__(self):
        self.thread = Thread(target=self.motion,args=())
        self.thread.daemon = True
        self.thread.start()
        
    def motion(self):
        while True:
            if val=='t':
                me.takeoff()
                time.sleep(1)
            elif val=='l':
                me.land()
                time.sleep(1)
            elif val=='w':
                me.move_forward(20)
                time.sleep(1)
            elif val=='s':
                me.move_back(20)
                time.sleep(1)
            elif val=='a':
                me.move_left(20)
                time.sleep(1)
            elif val=='d':
                me.move_right(20)
                time.sleep(1)
            elif val=='z':
                me.move_up(20)
                time.sleep(1)
            elif val=='x':
                me.move_down(20)
                time.sleep(1)
            elif val==',':
                me.rotate_counter_clockwise(30)
                time.sleep(1)
            elif val=='.':
                me.rotate_clockwise(30)
                time.sleep(1)
        
           
          
           
class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = me.get_frame_read()
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            
            self.frame = self.capture.frame
            
    
            blob = cv2.dnn.blobFromImage(self.frame,scalefactor=1/255,size=(whT,whT),mean=(0,0,0),swapRB=True,crop=False)
            net.setInput(blob)
            outputs = net.forward(outputNames)
            findObjects(outputs,self.frame)
            time.sleep(0.5)

    def show_frame(self):
        # Display frames in main program
        cv2.imshow('frame', self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)




if __name__ == '__main__':
    video_stream_widget = VideoStreamWidget()
    #thread2 = myThread("motion")
    #thread2.start()
    DroneMotion()
    while True:
        try:
            video_stream_widget.show_frame()
            
            val=0
            if keyboard.is_pressed('w'):
                val = 'w'
            elif keyboard.is_pressed('s'):
                val = 's'
            elif keyboard.is_pressed('a'):
                val = 'a'
            elif keyboard.is_pressed('d'):
                val = 'd'
            elif keyboard.is_pressed('z'):
                val = 'z'
            elif keyboard.is_pressed('x'):
                val = 'x'
            elif keyboard.is_pressed(','):
                val = ','
            elif keyboard.is_pressed('.'):
                val = '.'
            elif keyboard.is_pressed('t'):
                val = 't'
            elif keyboard.is_pressed('l'):
                val = 'l'
        except AttributeError:
            pass