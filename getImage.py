import os,sys
import base64
import cv2
import numpy as np

prefix = 'images/'
if __name__ == '__main__':
    file_name = sys.argv[1]
    with open(file_name) as infile:
        for line in infile:
            # print line
            arr = line.strip().split()
            dirname = arr[0]
            imgid = arr[1]
            faceid = arr[-3]
            rect = arr[-2]
            imgbase64 = arr[-1]


            rectdata = base64.b64decode(rect)
            q = np.frombuffer(rectdata, dtype=np.float32)
            print q
            # print rectdata
            # print 'freebasemid', arr[0]
            # print 'face', arr[4]
            imgdata = base64.b64decode(imgbase64)
            # mkdir
            mid_dir = prefix + dirname
            if not os.path.exists(mid_dir):
                os.makedirs(mid_dir)
            # save file
            filename = mid_dir + '/' + imgid + '_' + faceid + '.jpg'
            print filename
            with open(filename, 'wb') as f:
                f.write(imgdata)
            img = cv2.imread(filename)
            cv2.imshow('abc', img)
            cv2.waitKey()

