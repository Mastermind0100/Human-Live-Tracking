import cv2
import argparse
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', required=True,
            help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
            help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
            help = 'path to text file containing class names')
args = ap.parse_args()

cap = cv2.VideoCapture('your-video-file-here.mp4')
ax = plt.gca()
ax.cla()

ax.set_xlim(0,200)
ax.set_ylim(0,500)

d = 500
area_list=[]
distance_list=[]

while True:
    ret,image1 = cap.read()
    Width = image1.shape[1]
    Height = image1.shape[0]
    scale = 0.00392

    classes = None
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
        net = cv2.dnn.readNet(args.weights, args.config)
        blob = cv2.dnn.blobFromImage(image1, scale, (416,416), (0,0,0), True, crop=False)
        net.setInput(blob)

    def get_output_layers(net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])
        color = COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(label)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        print(i, end = ' ')
        i = i[0]
        box = boxes[i]
        x = round(box[0]) if round(box[0])>=0 else 0
        y = round(box[1]) if round(box[1])>=0 else 0
        w = round(box[2]) if round(box[2])>=0 else 0
        h = round(box[3]) if round(box[3])>=0 else 0
        
        if str(classes[class_ids[i]]) != "person":
            continue
        cv2.rectangle(image1,(x,y),(x+w,y+h),(0,255,0),2)
        
        area = h
        if(len(area_list)==0):
            area_list.append(area)
        else:
            dist = abs(((area_list[0]/area)-1)*d)
            # if(dist>100):
            #     dist=dist-100
            distance_list.append(dist)
            if(len(distance_list)>=3):
                dist = (distance_list[-3]+distance_list[-2]+distance_list[-1])/3
                print(dist)
                y1 = (0.5*(dist))-10
                plt.plot((200-y1),(500-dist),'r.')
                plt.pause(0.005)
        
    cv2.imshow('final', image1)
    # cv2.waitKey()
    k= cv2.waitKey(30) & 0xff
    if k==27:
        break

plt.show()
cap.release()
cv2.destroyAllWindows()

