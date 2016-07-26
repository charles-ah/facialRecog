import numpy as np
import cv2
import sys

out = open('savedimages.txt','a')

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]
out.write(str(imagePath)+"\n")
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

X=[]
Y=[]
W=[]
H=[]

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect faces in the image  
def detect(scale,img,origX,origY):
    
    # Read the image                                                                            
  #  img = cv2.imread(image)
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
        minSize=(1, 1),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

#    print str(imgW)
 #   imgH = np.size(img,1)
  #  print str(imgH)
        
    cv2.imshow("image",img)
    cv2.waitKey(0)
    '''
    cv2.imshow("faces",gray[0:imgH,0:imgW])
    cv2. waitKey(0)'''

    for (x,y,w,h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
        '''
        X.append(x+origX)
        Y.append(y+origY)
        W.append(w)
        H.append(h)
        '''
        out.write(str(y) + " " + str(y+h) + str(x) + " " + str(x+w) + "\n")
        gray[y:y+h, x:x+w] = np.zeros(h,w)
        #cv2.imshow("img",gray)
        #cv2.waitKey(0)
        detect(scale,gray,0,0)
        break
    cv2.imshow("faces found",img)
    out.close()

detect(1.1,gray,0,0)
cv2.waitKey(0)
    
