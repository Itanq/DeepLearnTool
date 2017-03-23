#coding:utf-8

import kNN
import mnist2Text

def handWriteDigitClassify():

    # 训练数据集及标签
    mnist2Text.read_image('../caffe/data/mnist/train-images-idx3-ubyte', './train_image.txt')
    mnist2Text.read_label('../caffe/data/mnist/train-labels-idx1-ubyte', './train_label.txt')
    traingImage = kNN.img2Vector('./train_image.txt')
    traingLabel = kNN.label2Vector('./train_label.txt')

    # 测试数据集几标签
    mnist2Text.read_image('../caffe/data/mnist/t10k-images-idx3-ubyte', './test_image.txt')
    mnist2Text.read_label('../caffe/data/mnist/t10k-labels-idx1-ubyte', './test_label.txt')
    testImage = kNN.img2Vector('./test_image.txt')
    testLabel = kNN.label2Vector('./test_label.txt')

    error = 0.0
    for i in range(200):
        knnClass = kNN.classify(testImage[i], traingImage, traingLabel, 5)
        print " the kNN's classifies result is " + str(knnClass)
        print " the True is " + str(testLabel[i])
        if knnClass != testLabel[i]:
            error += 1.0
    
    print "the error rate : " + str(error/200.0)

        
if __name__ == '__main__':

    handWriteDigitClassify()
