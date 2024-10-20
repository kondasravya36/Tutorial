from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2
from tkinter import ttk
import os
import tensorflow as tf
import os
import sys
from keras.models import model_from_json
import pickle
from keras.applications.inception_v3 import InceptionV3

main = tkinter.Tk()
main.title("Deep Learning based Object Detection and Recognition Framework for the Visually-Impaired")
main.geometry("1200x1200")


global filename
global ssd
global inception_model

class_labels = ['fifty', 'fivehundred', 'hundred', 'ten', 'thousand', 'twenty']

net = cv2.dnn.readNetFromCaffe("model/SSD300.txt","model/SSD300.caffemodel")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def loadInception():
    global inception_model
    with open('model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        inception_model = model_from_json(loaded_model_json)
    json_file.close()    
    inception_model.load_weights("model/model_weights.h5")
    inception_model._make_predict_function()
    pathlabel.config(text="SSD-Inception Model loaded")

def detectCurrency(filename):
    global inception_model
    image = cv2.imread(filename)
    img = cv2.resize(image, (32,32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = inception_model.predict(img)
    predict = np.argmax(preds)
    output = ""
    if np.amax(preds) > 0.98:
        output = "Currency Note Recognized as: "+class_labels[predict]
    return output 

def ssdDetection():
    global filename
    global ssd
    row = 50
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    output = detectCurrency(filename)
    text.insert(END,str(filename)+" loaded\n")
    pathlabel.config(text=str(filename)+" loaded")
    image_np = cv2.imread(filename)
    image_np = cv2.resize(image_np,(800,500))
    (h, w) = image_np.shape[:2]
    ssd = tf.Graph()
    with ssd.as_default():
        od_graphDef = tf.GraphDef()
        with tf.gfile.GFile('model/frozen_inference_graph.pb', 'rb') as file:
            serializedGraph = file.read()
            od_graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(od_graphDef, name='')
    with ssd.as_default():
        with tf.Session(graph=ssd) as sess:
            blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)),0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.2:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    print(confidence * 100)
                    if (confidence * 100) > 70:
                        label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
                        cv2.putText(image_np, label, (10, row), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                        row = row + 30
                        text.insert(END,"Detected & Recognized Objects: "+CLASSES[idx]+"\n")
                    if (confidence * 100) > 50:
                        cv2.rectangle(image_np, (startX, startY), (endX, endY),COLORS[idx], 2)
    if len(output) > 0:
        text.insert(END,output+"\n")
    text.update_idletasks()
    cv2.putText(image_np, output, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)    
    cv2.imshow('SSD Object Detection Output', image_np)
    cv2.waitKey(0)


def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('SSD-Inception V3 Accuracy & Loss Graph')
    plt.show()

font = ('times', 14, 'bold')
title = Label(main, text='Deep Learning based Object Detection and Recognition Framework for the Visually-Impaired')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
inceptionButton = Button(main, text="Generate SSD-Inception Model", command=loadInception)
inceptionButton.place(x=50,y=100)
inceptionButton.config(font=font1)

ssdButton = Button(main, text="Run SSD300 Object Detection & Classification", command=ssdDetection)
ssdButton.place(x=50,y=150)
ssdButton.config(font=font1)

graphButton = Button(main, text="Inception Accuracy & Loss Graph", command=graph)
graphButton.place(x=480,y=150)
graphButton.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=480,y=100)

font1 = ('times', 12, 'bold')
text=Text(main,height=18,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
