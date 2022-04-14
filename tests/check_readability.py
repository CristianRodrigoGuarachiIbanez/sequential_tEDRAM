from __future__ import print_function
import os, time
import numpy as np
import h5py
import random

import argparse
parser = argparse.ArgumentParser(description='Checks out all the datasets in an .h5 file')
parser.add_argument("--path", help="Path of the data file", type=str, action="store", default='no file', dest="path")
parser.add_argument("--std", help="Show mean and SD of Images", type=int, action="store", default=0, dest="std")

options, unknown = parser.parse_known_args()
path = options.path
std = options.std


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

data = []

for (i, n) in enumerate(f):
    data.append(f.get(str(n)))
    l = len(data[i].shape)
    if (l==1):
        print(n, type(data[i][0]), data[i].shape,'\n\n', data[i][0, ...],'\n\n')
    elif (l==2):
        print(n, type(data[i][0,0]), data[i].shape,'\n\n', data[i][0, ...],'\n\n')
    elif (l==3):
        print(n, type(data[i][0,0,0]), data[i].shape,'\n\n', data[i][0, ...],'\n\n')
    elif (l==4):
        print(n, type(data[i][0,0,0,0]), data[i].shape,'\n\n', data[i][0, ...],'\n\n')
    else:
        print(n, data[i].shape, l,'\n\n')

print("\nEverything OK!\n")

if std:

    train_images = data[0][:60000]
    test_images = data[0][60000:]

    # normalize data
    indices = list(range(60000))
    random.shuffle(indices)
    samples = train_images[sorted(indices[:1000]), ...]

    train_mean = np.mean(samples, axis=0)
    train_sd = np.std(samples, axis=0).clip(min=0.00001)

    indices = list(range(10000))
    random.shuffle(indices)
    samples = test_images[sorted(indices[:1000]), ...]

    test_mean = np.mean(samples, axis=0)
    test_sd = np.std(samples, axis=0).clip(min=0.00001)

    print(np.min(train_mean), np.max(train_mean))
    print(train_mean)
    input("\nweiter")
    print(np.min(train_sd), np.max(train_sd))
    print(train_sd)
    input("\nweiter")
    print(np.min(test_mean), np.max(test_mean))
    print(test_mean)
    input("\nweiter")
    print(np.min(train_sd), np.max(train_sd))
    print(test_sd)
