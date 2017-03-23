#coding:utf-8

import struct
from PIL import Image

def read_image(filename, dstname):
    image_text = open(dstname, "wb")

    f = open(filename, 'rb')
    buf = f.read()
    f.close()

    index = 0
    magic,images,rows,cols = struct.unpack_from(">IIII", buf, index)
    index += struct.calcsize(">IIII")

    text = ""
    for i in range(2000):
        for x in range(rows):
            for y in range(cols):
                value = int(struct.unpack_from(">B", buf, index)[0])
                if value != 0:
                    value = 1
                text += str(value)
                index += struct.calcsize(">B")
            text += "\n"

    image_text.write(text)
    image_text.close()


def read_label(filename, dstname):
    label_text = open(dstname, "wb")
    f = open(filename, 'rb')
    buf = f.read()
    f.close()

    index = 0
    magic,labels = struct.unpack_from(">II", buf, index)
    index += struct.calcsize(">II")

    text = ""
    for i in range(2000):
        text += str(struct.unpack_from(">B",buf,index)[0])
        index += struct.calcsize(">B")
    label_text.write(text)
    label_text.close()

