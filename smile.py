import cv2


face = "lbpcascade_frontalface.xml"
faceCascade = cv2.CascadeClassifier(face)


smilePath = "haarcascade_smile.xml"
smileCascade = cv2.CascadeClassifier(smilePath)

img = cv2.imread("test9.jpg")  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor= 1.1,
    minNeighbors=8,
    minSize=(55, 55),
    
)


for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

   
    smile = smileCascade.detectMultiScale(
        roi_gray,
        scaleFactor= 1.16,
        minNeighbors=35,
        minSize=(25, 25),
            )

   
    for (x2, y2, w2, h2) in smile:
        cv2.rectangle(roi_color, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)
        cv2.putText(img,'Smile',(x,y-7), 3, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('Smile', img)
c = cv2.waitKey(0)
