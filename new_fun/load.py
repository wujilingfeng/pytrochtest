import pickle
import numpy as np
#import matplotlib.pyplot as plt
import cv2 as cv
def unpickle(filename):
    with open(filename,'rb') as fo:
        mydict = pickle.load(fo,encoding='bytes')
    return mydict
def reduct(vec,row):
    image=np.zeros((32,32,3))
    
    for k in range(0,3,1):
        for i in range(0,32,1):
            for j in range(0,32,1):
                image[i][j][k]=vec[row][i*32+j+k*1024]
#print(vec[0][i*32+j+k*1024],end=" ")
    return image.astype(int)
def get_image(data,index):
    label=[]
    label.append(data[b'labels'][index])
    data1=reduct(data[b'data'],index)
    
    return data1,label
def get_batch(data,index):
    image=np.zeros([4,32,32,3])
    label=np.zeros([4])
    for i in range(4):
        ima,la=get_image(data,index+i)
        image[i]=ima
        label[i]=la[0]
    return image,label
def show_image(image):
    image1=image[0]
    for i in range(1,image.shape[0],1):
        image1=np.concatenate((image1,image[i]),axis=1)
    #return image1
    #print(image1)
    image1=image1/255.0

    cv.namedWindow("image",cv.WINDOW_NORMAL)
    cv.imshow("image",image1)
    cv.waitKey(0)
    cv.destroyAllWindows()

