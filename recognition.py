import numpy as np
import cv2
import sys


eigenfaces = np.load("savedeigenfaces.npy")
avgFace = np.load("savedavgface.npy")
u = np.load("savedeigenvector.npy")

theta = np.load("theta.npy")
#print eigenfaces
#print theta
names = (open("faces.txt").read()).split("\n")

out = open("results.txt",'w')

outpos = open("positive.txt","a")
outneg = open("negative.txt",'a')
#image = cv2.imread(sys.argv[1])
faceCascade = cv2.CascadeClassifier(sys.argv[1])

# Camera 0 is the integrated web cam on my computer
camera_port = 0

#Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 30

# Now we can initialize the camera capture object with the cv2.VideoCapture class.
# All it needs is the index to a camera port.
camera = cv2.VideoCapture(camera_port)

 
# Captures a single image from the camera and returns it in PIL format
def get_image():
 # read is the easiest way to get a full image out of a VideoCapture object.
 retval, im = camera.read()
 return im

for i in xrange(ramp_frames):
 temp = get_image()
#print("Taking image...")
# Take the actual image we want to keep
camera_capture = get_image()

image = camera_capture
#print image.shape
del(camera)
#cv2.imshow("pic",image)
#cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#gray =image
#print image.shape
#print cv2.resize(gray[0],(20,20))
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
        minSize=(50, 50),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

    for (x,y,w,h) in faces:
        
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
        addfaces(cv2.resize(gray[y:y+h,x:x+w],(20,20)))
    
        cv2.imshow("face",gray)
        cv2.waitKey(0)
detect(1.1,gray,0,0)

#print u.shape[1]
#print(len(facevector))
i=0
while i < len(facevector):
#    print str(i)
 #   cv2.imshow("faces",facevector[i])
  #  cv2.waitKey(0)
    arr = np.subtract(facevector[i],avgFace)
    facevector[i] = arr
    i+=1
#    i = facevector.index(arr)
#    print i
    #print str(np.sum(np.square(temparr)))
    #cv2.imshow(str(np.sum(np.square(temparr))),np.reshape(arr,(20,20)))
    #cv2.waitKey(0)
'''
    if np.sum(np.square(arr))>1000000:
        del facevector[i]
    else:
        i+=1
'''

def sigmoid(arr):
    return 1.0 / (1.0 + np.exp(-1*arr))

omega = np.dot(np.transpose(u),np.transpose(facevector))
#print eigenfaces.shape[1]

MinIndex = []
MinDist = []

for i in range(omega.shape[1]):
    temp = []
    face = omega[:,i]
    for j in range(eigenfaces.shape[1]):
        eigenface = eigenfaces[:,j]
        dist = np.sum(np.square(face - eigenface))
        temp.append(dist)
        #print dist
        #outneg.write(str(dist) +"\n")
    MinIndex.append(np.argmin(temp))
    MinDist.append(min(temp))

if len(MinDist) == 1:
    out.write(names[MinIndex[0]] + "\n")
else:
    MinDist = (1/np.std(MinDist))*(MinDist - np.sum(MinDist)/len(MinDist))
    for i in range(len(MinDist)):    
        #    print MinDist[i]*theta[1]+theta[0]`
        print sigmoid(MinDist[i]*theta[1]+theta[0])
        if (sigmoid(MinDist[i]*theta[1]+theta[0]) > 0.5):
            out.write(names[MinIndex[i]] + "\n")
        else:
            out.write("Unidentified \n")

#outneg.close()
'''
print names[MinIndex[np.argmin(MinDist)]]
if len(MinDist) != 1:
    MinDist = 1/np.std(MinDist)*(MinDist - np.average(MinDist))
    if names[MinIndex[np.argmin(MinDist)]] == "czhang2":
        outpos.write(str(MinDist[np.argmin(MinDist)])+"\n")
    else:
        outneg.write(str(MinDist[np.argmin(MinDist)])+"\n")
'''
#print MinIndex
out.close()

#print MinDist[np.argmin(MinDist)]

#outpos.write(str(MinDist[np.argmin(MinDist)])+"\n")
outpos.close()
outneg.close()
