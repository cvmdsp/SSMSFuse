import numpy as np
import matplotlib.pyplot as plt
import os
import math
from tensorflow.python.ops.gen_array_ops import reshape
import MyLib as ML
import tensorflow as tf
import seaborn as sns
#tf.enable_eager_execution()



def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")


def get3band_of_tensor(outX,nbanch=0,nframe=[0,15,30]):
    X = outX[:,:,:,nframe]
    X = X[nbanch,:,:,:]
    return X

def setRange(X, maxX = 1, minX = 0):
    X = (X-minX)/(maxX - minX)
    return X


def imshow(X):
 #    X = ML.normalized(X)
    plt.close()
    X = np.maximum(X,0)
    X = np.minimum(X,1)
    plt.imshow(X[:,:,::-1])
    plt.axis('off')
    plt.show()



def imwrite(X,saveload='tempIm.JPEG'):
    plt.imsave(saveload , ML.normalized(X[:,:,::-1]))


def normalized(X):
    maxX = np.max(X)
    minX = np.min(X)
    X = (X-minX)/(maxX - minX)
    return X





		 

