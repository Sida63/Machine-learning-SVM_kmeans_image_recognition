import cv2
import numpy as np
img=cv2.imread("E:\\python\\imagemaltese\\0.jpg")
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT()
#sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)
a=des
print(a.shape)
for i in xrange(1200):
    #img=cv2.imread("E:\\python\\image\\"+str(i)+".jpg")
    try:
        img=cv2.imread("E:\\orginalpython\\imagemaltese\\"+str(i)+".jpg")
#img = cv2.imread('16.jpg')
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT()
#sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray,None)
        a=np.concatenate((a,des),axis=0)
        #np.concatenate(())
        #print(classified_points)
        img=cv2.drawKeypoints(gray,kp)
        cv2.imwrite("E:\\python\\siftimagemaltese\\"+"sift_keypoints"+str(i)+".jpg",img)
    except BaseException as error:
        #print("error")
        continue
temp, classified_points, means=cv2.kmeans(data=a,K=10,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
print(a.shape)
print(temp)
print(classified_points.shape)
print(means.shape)

