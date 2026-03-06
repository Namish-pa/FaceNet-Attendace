import cv2
import numpy as n
import face_recognition
import os
from datetime import datetime as d


path='Images'
imgs=[]
classNames=[]
l=os.listdir(path)
print(l)
for i in l:
    curimg=cv2.imread(f'{path}/{i}')
    imgs.append(curimg)
    classNames.append(os.path.splitext(i)[0])
print(classNames)
def findEncode(imgs):
    encodel=[]
    for i in imgs:
        i=cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(i)[0]
        encodel.append(encode)
    return encodel
def attendence(name):
    f=open('Attendence.csv','r+')
    namelisto=[]
    r=f.readlines()
    for i in r:
        entry=i.split(',')
        namelisto.append(entry[0])
    if name not in namelisto:
        t=d.now()
        time=t.strftime('%H:%M:%S')
        date=t.strftime('%d/%b/%Y')
        f.writelines(f'\n{name},{time},{date}')

encodeKnown=findEncode(imgs)
print('Done!')   #SHOWS ENCODINGS ARE FINISHED

cam=cv2.VideoCapture(0)
while True:
    success,img=cam.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    faceInCurrentFrame=face_recognition.face_locations(imgS)
    encodeCurFrame=face_recognition.face_encodings(imgS,faceInCurrentFrame)

    for enf,faceloca in zip(encodeCurFrame,faceInCurrentFrame):
        matches=face_recognition.compare_faces(encodeKnown,enf)
        facedistace=face_recognition.face_distance(encodeKnown,enf)
        #print(facedistace)
        matchIndex=n.argmin(facedistace)


        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            #print(name)
            a1,a2,a3,a4=faceloca
            a1, a2, a3, a4=a1*4,a2*4,a3*4,a4*4
            cv2.rectangle(img,(a4,a1),(a2,a3),(0,255,0),2)
            cv2.putText(img,name,(a4+6,a1-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            attendence(name)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)





