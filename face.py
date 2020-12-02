import cv2
import numpy as np
import face_recognition

# load the image
demoImg = face_recognition.load_image_file('Image/asif.jpg')
demoImg = cv2.cvtColor(demoImg, cv2.COLOR_BGR2RGB)

demoTest = face_recognition.load_image_file('Image/ahmed.jpg')
demoTest = cv2.cvtColor(demoTest, cv2.COLOR_BGR2RGB)

# face location
faceLoc = face_recognition.face_locations(demoImg)[0]  # only sending first element of image
encodeDemo = face_recognition.face_encodings(demoImg)[0]
cv2.rectangle(demoImg, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

# face location
faceLocTest = face_recognition.face_locations(demoTest)[0]  # only sending first element of image
encodeTest = face_recognition.face_encodings(demoTest)[0]
cv2.rectangle(demoTest, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeDemo], encodeTest)
# face distance
faceDis=face_recognition.face_distance([encodeDemo],encodeTest)
cv2.putText(demoTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
print(results,faceDis)

# print(faceLoc[0])
cv2.imshow('Asif', demoImg)
cv2.imshow('Test', demoTest)

cv2.waitKey(0)
