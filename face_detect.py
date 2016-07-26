import cv2
import sys

out = open('savedimages.txt','a')

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image



X1=0
Y1=0
w1=0
h1=0
i=1.1
# Draw a rectangle around the faces
out.write(imagePath+"\n")
faceX=[]
faceY=[]
faceH=[]
faceW=[]
while i<2: 
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=i,
        minNeighbors=1,
        minSize=(40, 40),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
#    print "debug1"
    for (x, y, w, h) in faces:
    
#        print str(x) + ' ' + str(y)
        '''    X1 = x
        Y1 = y
        w1 = w
        h1 = h'''
        if faceX.count(x) == 0 and faceY.count(y) == 0:
            faceX.append(x)
            faceY.append(y)     
            faceH.append(h)
            faceW.append(w)
            out.write(str(x)+" "+str(x+w))
            out.write(" ")
            out.write(str(y)+" "+str(y+h))
            out.write(" ")
            out.write("\n")
    print "Found {0} faces!".format(len(faces))
    print str(i)
    '''    imagenew = image[Y1:Y1+h1,X1:X1+w1]
        cv2.imshow("Face "+ str(i), imagenew)
        cv2.waitKey(0)'''
    i+=0.03       
for i in range(0,len(faceX)):
    cv2.rectangle(image, (faceX[i], faceY[i]), (faceX[i]+faceW[i], faceY[i]+faceH[i]), (0, 255, 0), 2)
cv2.namedWindow(imagePath, cv2.WINDOW_NORMAL)

cv2.imshow("Faces found", image)
cv2.waitKey(0)

out.close()
