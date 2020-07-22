#Aref pourhashemi
import cv2
import numpy as np

faceXML=cv2.CascadeClassifier('face.xml')#read face xml file
leyeXML=cv2.CascadeClassifier('leye.xml')#read left eye xml file
reyeXML=cv2.CascadeClassifier('reye.xml')#read right eye xml file
smileXML= cv2.CascadeClassifier('smile.xml')#read lip(smile) xml file

eye_pic=cv2.imread("eye.png")
lip_pic=cv2.imread("lip.png")
cap=cv2.VideoCapture(0) #choose camera
while True:
    _,farme=cap.read()
    gray = cv2.cvtColor(farme,cv2.COLOR_BGR2GRAY) #change BGR--> GRAY
    faces=faceXML.detectMultiScale(gray,1.3,5) #fade detection

    for (x,y,w,h) in faces:
        x2= w//2 + x
        y2= h//2 + y
        r=w//2
        cv2.circle(farme,(x2,y2), r,(255,0,0),2) #make circle arount face
##################################################blurred the backround#######
        mask=np.zeros(farme.shape,dtype="uint8")
        cv2.circle(mask,(x2,y2), r,(255,255,255),-1)
        invmask=cv2.bitwise_not(mask)
        head=cv2.add(farme,invmask)
        mask=cv2.add(farme,mask)
        blurred=cv2.GaussianBlur(mask,(5,5),5)
        farme=cv2.bitwise_and(head,blurred)
         
###################################################end#######################
        f_gary=gray[y:y+h,x:x+w]
        farme2=farme.copy() #make copy from our camera picture

        f_color = farme[y:y+h,x:x+w]
        ff_color = farme2[y:y+h,x:x+w]
################for right eye
        reyes=reyeXML.detectMultiScale(f_color) #rihgt eye detection
        for (rex,rey,rew,reh) in reyes:
            reye = f_color[rey : rey+reh , rex : rex+rew]
            resezed_reyes=cv2.resize(reye,(200,200)) #resize right eye picture
            cv2.rectangle(f_color,(rex,rey),(rex+rew,rey+reh),(0,0,255),2)
            r_eye=cv2.resize(eye_pic,(rew,reh)) #resize animation picture
            ff_color[rey:rey+reh,rex:rex+rew]=r_eye  #change animation pic with original

###################for left eye
        leyes=leyeXML.detectMultiScale(f_color) #left eye detection
        for (lex,ley,lew,leh) in leyes:
            leye = f_color[ley : ley+leh , lex : lex+lew]
            resezed_leyes=cv2.resize(leye,(200,200)) #resize left eye picture

            cv2.imshow('left & right eyes', np.hstack([resezed_reyes,resezed_leyes])) #show left & right eye 

            cv2.rectangle(f_color,(lex,ley),(lex+lew,ley+leh),(0,0,255),2)
            l_eye=cv2.resize(eye_pic,(lew,leh)) #resize animation picture
            ff_color[ley:ley+leh,lex:lex+lew]=l_eye #change animation pic with original
            

####################for lip(smile)
        smiles=smileXML.detectMultiScale(f_gary,1.4,6) #lip(smile) detection

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(f_color, (sx, sy), (sx + sw, sy + sh), (221,15,185), 2)
            lip=cv2.resize(lip_pic,(sw,sh)) #resize animation lip pic
            ff_color[sy:sy+sh,sx:sx+sw]=lip
        cv2.imshow("change eyes & lip",farme2) #change lip pic with original
########################end
    cv2.imshow('face',farme)
    k=cv2.waitKey(27) & 0xFF
    if (k== 27):
        break

cv2.circle(farme,(x2,y2), x2,(255,0,0),10)
cv2.destroyAllWindows()
cap.release()