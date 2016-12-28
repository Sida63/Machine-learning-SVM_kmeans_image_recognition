import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw,ImageFont

'''
set up training_set, training_labels and so on.
Like initializtion
'''
accuracyresult=[]
for k_num in range(1,11):
    training_set=[]
    training_labels=[]
    svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                        svm_type = cv2.SVM_C_SVC,
                        C=2.67, gamma=5.383 )
    img=cv2.imread("E:\\python\\imagemaltese\\0.jpg")
    res=cv2.resize(img,(250,250))
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray,None)
    temptest, classified_pointstest, meanstest=cv2.kmeans(data=des,K=k_num,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
    meanstest=meanstest.reshape(-1)
    meanstest=meanstest.T
    print(meanstest.shape)
    training_set.append(meanstest)
    training_labels.append(1)

    trainData=np.float32(training_set)
    responses=np.float32(training_labels)

    '''
    produce the traning data set of cat and extract key points by SIFT and kmeans
    '''

    for i in xrange(10):
        try:
            img=cv2.imread("E:\\python\\imagecat\\"+str(i)+".jpg")
            res=cv2.resize(img,(250,250))
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT()
            kp, des = sift.detectAndCompute(gray,None)
            temptest, classified_pointstest, meanstest=cv2.kmeans(data=des,K=k_num,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
            meanstest=meanstest.reshape(-1)
            training_set.append(meanstest)
            training_labels.append(0)
            trainData=np.float32(training_set)
            responses=np.float32(training_labels)
        except BaseException as error:
            continue
        
    '''
    produce the traning data set of maltese and extract key points by SIFT and kmeans
    '''

    for i in xrange(10):
        try:
            img=cv2.imread("E:\\python\\imagemaltese\\"+str(i)+".jpg")
            res=cv2.resize(img,(250,250))
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT()
            kp, des = sift.detectAndCompute(gray,None)
            temptest, classified_pointstest, meanstest=cv2.kmeans(data=des,K=k_num,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
            meanstest=meanstest.reshape(-1)
            training_set.append(meanstest)
            training_labels.append(1)
            trainData=np.float32(training_set)
            responses=np.float32(training_labels)
        except BaseException as error:
            continue

    '''
    produce the traning data set of mountaindog and extract key points by SIFT and kmeans
    '''

    for i in xrange(10):
        try:
            img=cv2.imread("E:\\python\\imagemountaindog\\"+str(i)+".jpg")
            res=cv2.resize(img,(250,250))
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT()
            kp, des = sift.detectAndCompute(gray,None)
            temptest, classified_pointstest, meanstest=cv2.kmeans(data=des,K=k_num,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
            meanstest=meanstest.reshape(-1)
            training_set.append(meanstest)
            training_labels.append(2)
            trainData=np.float32(training_set)
            responses=np.float32(training_labels)
        except BaseException as error:
            continue

    '''
    produce the traning data set of panda and extract key points by SIFT and kmeans
    '''

    for i in xrange(10):
        try:
            img=cv2.imread("E:\\python\\imagepanda\\"+str(i)+".jpg")
            res=cv2.resize(img,(250,250))
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT()
            kp, des = sift.detectAndCompute(gray,None)
            temptest, classified_pointstest, meanstest=cv2.kmeans(data=des,K=k_num,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
            meanstest=meanstest.reshape(-1)
            training_set.append(meanstest)
            training_labels.append(3)
            trainData=np.float32(training_set)
            responses=np.float32(training_labels)
        except BaseException as error:
            continue

    '''
    produce the traning data set of tiger and extract key points by SIFT and kmeans
    '''
    for i in xrange(10):
        try:
            img=cv2.imread("E:\\python\\imagetiger\\"+str(i)+".jpg")
            res=cv2.resize(img,(250,250))
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT()
            kp, des = sift.detectAndCompute(gray,None)
            temptest, classified_pointstest, meanstest=cv2.kmeans(data=des,K=k_num,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
            meanstest=meanstest.reshape(-1)
            training_set.append(meanstest)
            training_labels.append(4)
            trainData=np.float32(training_set)
            responses=np.float32(training_labels)
        except BaseException as error:
            continue
    '''
    until this part, the training data has been created and next we will use it to train different decision boundarys through SVM
    '''
    arr= np.array(training_labels)
    print(arr.shape)
    print(trainData.shape)
    training_labels=arr
    print(trainData.shape)
    print(training_labels)
    responses=np.float32(training_labels)
    svm = cv2.SVM()
    svm.train(trainData,responses, params=svm_params)

    '''
    produce the test set
    '''
    Paths=[]
    training_set=[]
    training_labels=[]
    img=cv2.imread("E:\\python\\imagemaltese\\0.jpg")
    res=cv2.resize(img,(250,250))
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray,None)
    temptest, classified_pointstest, meanstest=cv2.kmeans(data=des,K=k_num,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
    meanstest=meanstest.reshape(-1)
    meanstest=meanstest.T
    print(meanstest.shape)
    training_set.append(meanstest)
    training_labels.append(1)
    trainData=np.float32(training_set)
    print(trainData.shape[0])
    responses=np.float32(training_labels)

    font=cv2.FONT_HERSHEY_SIMPLEX
    im = Image.open("E:\\python\\imagemaltese\\0.jpg").convert('RGBA')
    im=im.resize((250,250))
    txt=Image.new('RGBA', im.size, (0,0,0,0))
    fnt=ImageFont.truetype("c:/Windows/fonts/Tahoma.ttf", 20)
    d=ImageDraw.Draw(txt)
    d.text((txt.size[0]-240,txt.size[1]-60), "true label:1",font=fnt, fill=(0,255,0,255))
    #d.text((txt.size[0]-120,txt.size[1]-60), "test label:1",font=fnt, fill=(255,0,0,255))
    out=Image.alpha_composite(im, txt)
    #out.show()
    temppath="E:\\python\\testset\\"+"Knum_"+str(k_num)+"i_"+str(trainData.shape[0])+".jpg"
    out.save(temppath)
    Paths.append(temppath)

    for i in xrange(3):
        try:
            img=cv2.imread("E:\\python\\imagepanda\\"+str(i)+".jpg")
            res=cv2.resize(img,(250,250))
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT()
            kp, des = sift.detectAndCompute(gray,None)
            temptest, classified_pointstest, meanstest=cv2.kmeans(data=des,K=k_num,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
            meanstest=meanstest.reshape(-1)
            training_set.append(meanstest)
            training_labels.append(3)
            trainData=np.float32(training_set)
            responses=np.float32(training_labels)
            font=cv2.FONT_HERSHEY_SIMPLEX
            im = Image.open("E:\\python\\imagepanda\\"+str(i)+".jpg").convert('RGBA')
            im=im.resize((250,250))
            txt=Image.new('RGBA', im.size, (0,0,0,0))
            fnt=ImageFont.truetype("c:/Windows/fonts/Tahoma.ttf", 20)
            d=ImageDraw.Draw(txt)
            d.text((txt.size[0]-240,txt.size[1]-60), "true label:3",font=fnt, fill=(0,255,0,255))
            #d.text((txt.size[0]-120,txt.size[1]-60), "test label:1",font=fnt, fill=(255,0,0,255))
            out=Image.alpha_composite(im, txt)
            #out.show()
            temppath="E:\\python\\testset\\"+"Knum_"+str(k_num)+"i_"+str(trainData.shape[0])+".jpg"
            out.save(temppath)
            Paths.append(temppath)
        except BaseException as error:
            print("error")
            continue
    for i in xrange(3):
        try:
            img=cv2.imread("E:\\python\\imagetiger\\"+str(i)+".jpg")
            res=cv2.resize(img,(250,250))
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT()
            kp, des = sift.detectAndCompute(gray,None)
            temptest, classified_pointstest, meanstest=cv2.kmeans(data=des,K=k_num,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
            meanstest=meanstest.reshape(-1)
            training_set.append(meanstest)
            training_labels.append(4)
            trainData=np.float32(training_set)
            responses=np.float32(training_labels)
            font=cv2.FONT_HERSHEY_SIMPLEX
            im = Image.open("E:\\python\\imagetiger\\"+str(i)+".jpg").convert('RGBA')
            im=im.resize((250,250))
            txt=Image.new('RGBA', im.size, (0,0,0,0))
            fnt=ImageFont.truetype("c:/Windows/fonts/Tahoma.ttf", 20)
            d=ImageDraw.Draw(txt)
            d.text((txt.size[0]-240,txt.size[1]-60), "true label:4",font=fnt, fill=(0,255,0,255))
            #d.text((txt.size[0]-120,txt.size[1]-60), "test label:1",font=fnt, fill=(255,0,0,255))
            out=Image.alpha_composite(im, txt)
            #out.show()
            temppath="E:\\python\\testset\\"+"Knum_"+str(k_num)+"i_"+str(trainData.shape[0])+".jpg"
            out.save(temppath)
            Paths.append(temppath)
        except BaseException as error:
            continue

    result = svm.predict_all(trainData)
    rightnum=0
    print(result)
    for h in xrange(len(result)):
        temppath=Paths[h]
        im = Image.open(temppath).convert('RGBA')
        im=im.resize((250,250))
        txt=Image.new('RGBA', im.size, (0,0,0,0))
        fnt=ImageFont.truetype("c:/Windows/fonts/Tahoma.ttf", 20)
        d=ImageDraw.Draw(txt)
        d.text((txt.size[0]-120,txt.size[1]-60), "test label:"+str(int(result[h][0])),font=fnt, fill=(255,0,0,255))
        out=Image.alpha_composite(im, txt)
        #out.show()
        out.save(temppath)
        if training_labels[h]==result[h]:
            rightnum+=1
    print(rightnum)
    tempaccuracy=rightnum/(len(result)*1.0)
    accuracyresult.append(tempaccuracy)
print(accuracyresult)
accuracyresult=np.array(accuracyresult)
plt.figure(0)
plt.clf()
accuracyresultnum = np.array(accuracyresult)
kindex=[]
for index in xrange(len(accuracyresultnum)):
    kindex.append(index)
kindex=np.array(kindex)
plt.plot(kindex[:], accuracyresultnum[:], 'b', label='Test')
plt.xlabel('K-centers')
plt.ylabel("Accuracy")
plt.legend()
plt.draw()
plt.pause(0.0001)
    #svm.save('svm_data.dat')
