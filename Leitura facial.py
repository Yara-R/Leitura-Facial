import cv2 as cv
import numpy as np
import _dlib_pybind11 as dlib

cap = cv.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = detector(gray)

    i=0

    for face in faces:
        x,y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        i = i + 1
        cv.putText(frame, 'face num ' + str(i), (x-10, y-10), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)
        print(face, i)

    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
