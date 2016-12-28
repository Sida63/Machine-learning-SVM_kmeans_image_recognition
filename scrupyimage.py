import urllib2
import urllib
import socket
import os
k=0
for t in xrange(12):
    try:
        response = urllib2.urlopen('http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02119022')  
        #html = response.read()
        i=0
        mkpath="E:\\python\\imagefox"
        flag=os.path.exists(mkpath)
        if not flag:
            os.mkdir(mkpath)
        tempk=0
        while(tempk<=k):
            html=response.readline()
            tempk+=1
        while(html!=''and i<100):
            html=response.readline()
            try:
                u = urllib2.urlopen(html,timeout=5)
                #file_size = int(u.headers['content-length']) have some problem, produced by python itself
                temp=u.headers.get('content-length')
                if temp!=None:
                    file_size=int(temp)
                    if file_size>50000 and file_size!=None:
                        data = u.read()
                        f = open(mkpath+"\\"+str(k+i)+".jpg", 'wb')
                        f.write(data)
                        f.close()
                print html
            except urllib2.HTTPError as error:
                print("error")
                continue
            except urllib2.URLError as error:
                print("error")
                continue
            except socket.timeout as error:
                print("time out")
                continue
            except socket.error as error:
                print("reconnect")
                continue
            except BaseException as error:
                print("unknown error")
            i+=1
        response.close()
        k=k+i
    except BaseException as error:
        print("unknown error")
        continue
