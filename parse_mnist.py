import numpy as np
import struct
from PIL import Image


def loadImageSet(filename):

    # Read file
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    # Get the first four integer, return a tuplie
    head = struct.unpack_from('>IIII', buffers, 0)

    # Reach the place of data begin
    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    # print(imgNum)
    width = head[2]
    # print(width)
    height = head[3]
    # print(height)

    # data -> 60000*28*28
    bits = imgNum * width * height
    # fmt format：'>47040000B'
    bitsString = '>' + str(bits) + 'B'

    # Get data，return a tuple
    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    # Reshape to array of [60000,784]
    imgs = np.reshape(imgs, [imgNum, width * height])

    return imgs, head





if __name__ == "__main__":

    file1 = 'train-images-idx3-ubyte'
  

    imgs, data_head = loadImageSet(file1)
    with open("mnist.txt","w") as f:
        for a in imgs:
            for b in a:
                f.write(str(b)+"\t")
            f.write("\n")
    # print('data_head:', data_head)
    # print(type(imgs))
    # print('imgs_array:', imgs)

  
