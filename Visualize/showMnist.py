#coding:utf-8

import struct
from PIL import Image

class MNIST():
    u'''
    用于显示mnist数据集的类,mnist数据集的格式如下：

    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    
    The labels values are 0 to 9.

    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    
    The labels values are 0 to 9.

    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black). 
    '''
    def __init__(self):
        pass

    def read_train_images(self, filename, split=False):

        f = open(filename, 'rb')
        buf = f.read()
        f.close()

        index = 0
        magic,images,rows,cols = struct.unpack_from('>IIII', buf, index)
        index += struct.calcsize('>IIII')

        images = 10000
        if split:
            for i in range(images):
                image = Image.new('L', (cols, rows))
                for x in range(rows):
                    for y in range(cols):
                        image.putpixel((y,x), int(struct.unpack_from('>B', bufm, index)[0]))
                        index += struct.calcsize('>B')
                image.save("./"+str(i)+".jpg")
        else:
            for i in range(10):
                image = Image.new('L', (cols*20, rows*20))
                for j in range(20):
                    for k in range(20):
                        for x in range(rows):
                            for y in range(cols):
                                image.putpixel((y+28*j,x+28*k), int(struct.unpack_from('>B', buf, index)[0]))
                                index += struct.calcsize('>B')

                image.save("./"+str(i)+".jpg")


    def read_train_labels(self, filename):

        self.labels = open("./train_labels.txt", "wb")

        f = open(filename, 'rb')
        buf = f.read()
        f.close()

        index = 0
        magic, labels = struct.unpack_from('>II', buf, index)
        index += struct.calcsize('>II')

        for i in range(1,labels+1):
            if i % 20 == 0:
                sep = "\n"
            else:
                sep = " "
            self.labels.write(str(struct.unpack_from('>B', buf, index)[0])+sep)
            index += struct.calcsize('>B')


if __name__ == '__main__':

    mnist = MNIST()
    print "read train images..."
    mnist.read_train_images('../caffe/data/mnist/train-images-idx3-ubyte')

    print "read train lebels..."
    mnist.read_train_labels('../caffe/data/mnist/train-labels-idx1-ubyte')

