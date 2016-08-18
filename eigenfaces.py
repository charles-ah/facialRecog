import numpy as np
import cv2
import sys

out = open('savedeigenfaces.txt','a')

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

facevector = []
def addfaces(img):
    img = np.reshape(img,img.shape[0]*img.shape[1])
    facevector.append(img)
#    print str(img.size)


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
        minSize=(20, 20),
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
        
        X.append(x+origX)
        Y.append(y+origY)
        W.append(w)
        H.append(h)

        addfaces(cv2.resize(gray[y:y+h,x:x+w],(20,20)))
        #out.write(str(y) + " " + str(y+h) + str(x) + " " + str(x+w) + "\n")
       # gray[y:y+h, x:x+w] = np.zeros(h,w)
        #cv2.imshow("img",gray)
        #cv2.waitKey(0)
       # detect(scale,gray,0,0)
       # break
    cv2.imshow("faces found",img)
    #out.close()
    cv2.waitKey(0)
detect(1.1,gray,0,0)
cv2.waitKey(0)
    
def computeAvg():
   # img = np.reshape(img,img.shape[0]*img.shape[1])
    sumvec = np.zeros(400)
    m = 1.0/len(facevector)
#    print str(m)
    for arr in facevector:
        sumvec=np.add(sumvec,arr)
#        print sumvec
    avg =m*sumvec
    #print avg
    np.save("avgface.npy",avg)
    return avg
#    avg=avg.reshape(20,20)
    
#    cv2.imshow("avg faces",avg)
#    cv2.waitKey(0)
avgFace = computeAvg()
#eigenfaces(gray[Y[1]:Y[1]+H[1],X[1]:X[1]+W[1]])    
#print facevector


i=0
while i < len(facevector):
#    print str(i)
 #   cv2.imshow("faces",facevector[i])
  #  cv2.waitKey(0)
    arr = np.subtract(facevector[i],avgFace)
    facevector[i] = arr
#    i = facevector.index(arr)
#    print i
    #print str(np.sum(np.square(temparr)))
    #cv2.imshow(str(np.sum(np.square(temparr))),np.reshape(arr,(20,20)))
    #cv2.waitKey(0) 
    if np.sum(np.square(arr))>1000000:
        del facevector[i]
    else:
        i+=1

'''        
for arr in facevector:
#    print str(np.sum(np.square(arr)))
#    print np.add(arr,avgFace)
    cv2.imshow("face",np.reshape(arr,(20,20)))
    cv2.waitKey(0)
'''


cov = np.zeros((400,400))
m = 1.0/ len(facevector)
for arr in facevector:
    cov = m*np.add(cov,np.transpose(arr) * arr)
#    print np.dot(np.transpose(arr),arr)
#print cov
eigvalue =  np.linalg.eig(cov)[0]
eigvector = np.linalg.eig(cov)[1]

#print eigvalue
#print eigvector

[U, S, V] = np.linalg.svd(cov)

U_reduce = U[1,0:len(facevector)]

u =np.transpose(facevector)*U_reduce
#print u.shape[0]
#print u.shape[1]

for i in range(u.shape[1]):
    cv2.imshow("eigenface",np.reshape(u[:,i],(20,20)))
    cv2.waitKey(0)

omega = np.transpose(u)*facevector
np.save("savedeigenfaces.npy",omega)
#for face in omega:
 #   print face
  #  face))
#out.close()
