
import numpy as np
from PIL import Image

former_phase = 'E:/AI/tensorflow/tensorflow/test/edgetest/HED-BSDS/'

def load_data (dirname ):
    file = open(dirname,'r')
    arrayOLines = file.readlines()
    numberOfLines = len(arrayOLines)

    srcList = []
    labelList = []
    index = 0
    for line in arrayOLines:
        temp = line.split(' ')
        srcList.append(former_phase + temp[0])

        labelList.append(former_phase + temp[1].strip('\n'))

        index += 1
    return srcList,labelList

def read_images(srcList):

    numberOfLines = len(srcList)
    dataMat = np.zeros((numberOfLines,224,224,3),dtype='float32')
    index = 0
    for v in srcList:
        img = Image.open(v)
        img = img.resize((224,224))
        dataMat[index,:] = img
        index += 1

    return dataMat

