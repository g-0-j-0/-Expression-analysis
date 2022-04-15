import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

"""img= cv2.imread("happy boy.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
prediction =DeepFace.analyze(img)
print(prediction["dominant_emotion"])
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)

for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (66, 50, 200), 6)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,
            prediction["dominant_emotion"],
            (100,200),
            font,10,
            (0,0,255),
            10,
            cv2.LINE_4);

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))"""

#Above code is limited for image completion needed for video processing/]..
a=0
h=0
cap=cv2.VideoCapture(cv2.CAP_DSHOW)
"""if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open webcam")"""
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
while True:
    font = cv2.FONT_HERSHEY_SIMPLEX
    a += 1
    ret,frame=cap.read()
    if frame is None:
        raise ValueError('Unable to get a frame!')
    print(frame)
    result= DeepFace.analyze(frame, actions=['emotion'] ,enforce_detection=False)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor =1.05, minNeighbors=5)
    if faces is None:
        cv2.putText(frame,
                    "NO FACE DETECTED",
                    (10, 150),
                    font, 5,
                    (0, 0, 255),
                    5,
                    cv2.LINE_4);
    for x, y, w, h in faces:
        h=cv2.rectangle(frame, (x, y), (x + w, y + h), (66, 50, 200), 6)
        

    cv2.putText(frame,
                result["dominant_emotion"],
                (10, 150),
                font, 5,
                (0, 0, 255),
                5,
                cv2.LINE_4);
    cv2.imshow("Original Video", frame)

    if cv2.waitKey(2) & 0xFF==ord('q'):
        break
print(a)
cap.release()
cv2.destroyAllWindows()