# TelloDrone_ObjDetection
Tello Drone Control &amp; Object Detection


Simple personal project using Daemon multithreads for Vision processing & Drone motion control, and YOLOv3 on OpenCV DNN for object detection.

Keyboard inputs:

t: Drone take off

l: Drone landing

w: move forward by 20cm

s: move backward by 20cm

a: move left by 20cm

d: move right by 20cm

z: move upward by 20cm

x: move downward by 20cm

,: rotate ccw by 30 degrees

.: rotate cw by 30 degrees



keyboard inputs are listened in the while loop in main; while motion control & vision camera are processed in the background as threads
