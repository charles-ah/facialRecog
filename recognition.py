import numpy as np
import cv2
import sys


eigenfaces = np.load("savedeigenfaces.npy")
avgface = np.load("savedavgface.npy")
print eigenfaces

image = cv2.imread(sys.argv[0])
faceCascade = cv2.CascadeClassifier(sys.argv[1])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


facevector = []
def addfaces(img):
    img = np.reshape(img,img.shape[0]*img.shape[1])
    facevector.append(img)
#print str(img.size
# Detect faces in the image
def detect(scale,img,origX,origY):
    imgH = img.shape[0]
    imgW = img.shape[1]
#    print str(imgW)
#    print str(imgH)

#    print img
    # Multiscale detection
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=scale,
        minNeighbors=1,
        minSize=(20, 20),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

    for (x,y,w,h) in faces:

        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
        addfaces(cv2.resize(gray[y:y+h,x:x+w],(20,20)))
    
detect(1.1,gray,0,0)
