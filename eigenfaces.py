import numpy as np
import cv2
import sys
import os

out = open('faces.txt','w')

# Get user supplied values
#imagePath = sys.argv[1]
#cascPath = sys.argv[2]
cascPath = sys.argv[1] 
#out.write(str(imagePath)+"\n")
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

X=[]
Y=[]
W=[]
H=[]


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
        cv2.imshow("img",img[y:y+h,x:x+w])
        cv2.waitKey(0)
       # detect(scale,gray,0,0)
       # break
    cv2.imshow("faces found",img)
    #out.close()
    cv2.waitKey(0)

#print os.getcwd()
for imagePath in os.listdir("/Users/CZhang/Documents/CZ/CS/MachineLearning/facialRecog/faces"):
    if imagePath.endswith(".png"):
        #print imagePath
        out.write(imagePath[0:-4]+"\n")
        image = cv2.imread("/Users/CZhang/Documents/CZ/CS/MachineLearning/facialRecog/faces/"+imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        detect(1.1,gray,0,0)
#out.close()
'''
for dir in os.listdir("/Users/CZhang/Documents/CZ/CS/MachineLearning/facialRecog/faces/att_faces"):
    for imagePath in os.listdir("/Users/CZhang/Documents/CZ/CS/MachineLearning/facialRecog/faces/att_faces/"+dir):
        if imagePath.endswith(".pgm"):
        #print imagePath
            out.write(dir + imagePath[0:-4]+"\n")
            image = cv2.imread("/Users/CZhang/Documents/CZ/CS/MachineLearning/facialRecog/faces/att_faces/"+dir+"/"+imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
           # gray = image
            detect(1.1,gray,0,0)
'''

def computeAvg():
   # img = np.reshape(img,img.shape[0]*img.shape[1])
    sumvec = np.zeros(400)
    m = 1/len(facevector)   
 #   print len(facevector)    
   
    for arr in facevector:
        sumvec=np.add(sumvec,arr)
#        print sumvec
    avg =m*sumvec
    #print avg
    np.save("savedavgface.npy",avg)
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
    cov = m*np.add(cov,np.dot(np.transpose(arr), arr))
#    print np.dot(np.transpose(arr),arr)
#print cov
eigvalue =  np.linalg.eig(cov)[0]
eigvector = np.linalg.eig(cov)[1]

#print eigvalue
#print eigvector

[U, S, V] = np.linalg.svd(cov)
print U.shape
#U_reduce = U[:,0:len(facevector)]
k=1
ProjErr=[]
while k < 400:
    U_reduce = U[:,0:k]
    omega = np.dot( np.transpose(U_reduce),np.transpose(facevector))
    appr = np.dot(U_reduce,omega)
    delta = appr - np.transpose(facevector)
    error = 0
    for i in range(len(facevector)):
        error += sum(np.square(delta[i,:]))
    Var = 0
    for i in range(len(facevector)):
        Var += sum(np.square(np.array(facevector[i])))
    error = error/Var
    ProjErr.append(error)
    k+=1
k = np.argmin(ProjErr)
print k
print ProjErr[k]
#u =np.transpose(facevector)*U_reduce
#print u.shape[0]
#print u.shape[1]
'''
for i in range(u.shape[1]):
    cv2.imshow("eigenface",np.reshape(u[:,i],(20,20)))
    cv2.waitKey(0)
'''
#print facevector
#omega = np.transpose(u)*facevector
omega = np.dot( np.transpose(U_reduce),np.transpose(facevector))

np.save("savedeigenvector.npy",U_reduce)
np.save("savedeigenfaces.npy",omega)
#for face in omega:
 #   print face
  #  face))
#out.close()
