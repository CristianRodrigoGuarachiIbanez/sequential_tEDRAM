"""

    This prints examples of the MNIST cluttered dataset in the console

"""

import os
import numpy as np
import h5py

path = '/scratch/forch/EDRAM/datasets/mnist_cluttered_test.hdf5'

# open h5 file as 'f'
if (os.path.isfile(path)):
    print("Opening", path, "\n")
else:
    print(path, "does not exist\n")
    exit()

try:
    f = h5py.File(path, 'r')
except Exception as e:
    print(e)
    exit()

# access the data
features = f["features"]
locations = f["locations"]
labels = f["labels"]

# print random image or user selected image
while True:
    # get number of image
    i = input()
    if i == '':
        n = np.random.randint(0, 69999)
    else:
        n = int(i)
        if i == '0':
            break

    # print image
    for i in range(0, 100):
        line = ''
        for j in range(0, 100):
            # line += str(features[n,0,i,j])

            # concatenate black and white squares as pixels to build a line
            char = u"\u25A0" if features[n,0,i,j]>0 else u"\u25A1"

            # a zero indicates the center of the digit patch
            char = '0' if abs(int(locations[n,2]*100)-int((int(j) - 50.0) * 2))<=1 and abs(int(locations[n,5]*100)-int((int(i) - 50.0) * 2))<=1 else char

            line += char

        print(line)

    # print info: labels, standardized x,y coordinates, pixel x,y coordinates
    print("\n", labels[n, ...], locations[n, (2,5)], int(np.round(locations[n,2]*50-14+50)), int(np.round(locations[n,5]*50-14+50)), "\n")


f.close()
