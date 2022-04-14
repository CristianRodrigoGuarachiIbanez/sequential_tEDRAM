from __future__ import print_function
import sys, os
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt

def check_dataset(data_path=r'D:\Documents\Akademia\Aktuell\MA_IGS\data\AffectNet\AffectNet\AffectNet_validation_data.h5', vis = True, nb_images = 80):
    """
    Check validity of a generated dataset.
    """
    print("Opening", data_path)
    try:
        f = h5py.File(data_path, "r")
    except Exception as e:
        print(e)
        print("Error:", data_path, "does not exist.")
        return

    # Access the data
    X = f["X"]
    Y_lab = f["Y_lab"]
    Y_val = f["Y_val"]
    Y_ars = f["Y_ars"]
    I = f["Images"]

    # Print the shape of the data
    print("Input data:", X.shape)
    print("Output data (label):", Y_lab.shape)
    print("Output data (valence):", Y_val.shape)
    print("Output data (arousal):", Y_ars.shape)

    # Select 20 images randomly
    N = X.shape[0]
    indices = list(sorted(np.random.choice(N, nb_images, replace=False)))

    # or select 10 images of each category
    if (False):
        no_pictures = np.zeros(8)
        indices = []
        i = 1000
        l = Y_lab.shape[0]
        while (sum(no_pictures) < 80):
            i = (i+13) % l
            for c in range(8):
                if (Y_lab[i,c]==1 and no_pictures[c]<10):
                    no_pictures[c] += 1
                    indices.append(i)
                    if (no_pictures[c]==10): print('found', no_pictures[c], c, 'in', i, '!', sum(no_pictures), 'total.')

    indices = sorted(indices)

    X_sample = X[indices, ...]
    Y_lab_sample = Y_lab[indices, ...]
    Y_val_sample = Y_val[indices, ...]
    Y_ars_sample = Y_ars[indices, ...]
    I_sample = I[indices]

    # Check range
    X_min = X_sample.min()
    X_max = X_sample.max()
    print("Input range should be close to [0, 1]:", X_min, X_max)
    Y_lab_min = 8
    Y_lab_max = -1
    for lab in Y_lab_sample:
      pos = np.where(lab==1)[0][0]
      if (pos < Y_lab_min): Y_lab_min = pos
      if (pos > Y_lab_max): Y_lab_max = pos
    print("Output range should be [0, 7]:", Y_lab_min, Y_lab_max)
    print("Output range should be close to [-1, 1]:", Y_val_sample.min(), Y_val_sample.max())
    print("Output range should be close to [-1, 1]:", Y_ars_sample.min(), Y_ars_sample.max())

    # Check outputs:
    #print(indices)
    print(nb_images, "outputs:")
    print('neu hap sad sur fear dis ang con')
    print(Y_lab_sample)

    if (vis):
        # Visualize inputs
        for i in range(len(indices)):
            #plt.subplot(4, 5, i+1)
            print(I_sample[i])
            print(Y_lab_sample[i])
            print(round(Y_val_sample[i], 2), round(Y_ars_sample[i],2))
            plt.imshow(X_sample[i, :, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
            plt.show()

    f.close()

    exit()

    # or select neutral images with high valence/arousal
    no_pictures = np.zeros(6).astype(int)
    indices = np.zeros(60).astype(int)
    i = 223
    l = Y_lab.shape[0]
    while (sum(no_pictures) < 60):
        i = (i+137) % l
        #sadness
        if (Y_lab[i,0]==1 and Y_val[i]<=-0.25 and Y_ars[i]<=0 and no_pictures[0]<10):
            indices[no_pictures[0]] = i
            no_pictures[0] += 1
        #sadness2
        if (Y_lab[i,0]==1 and Y_val[i]<=-0.05 and Y_val[i]>-0.25 and Y_ars[i]<=-0.2 and no_pictures[1]<10):
            indices[no_pictures[1]+10] = i
            no_pictures[1] += 1
        #anger
        if (Y_lab[i,0]==1 and Y_val[i]<=-0.25 and Y_ars[i]>0 and no_pictures[2]<10):
            indices[no_pictures[2]+20] = i
            no_pictures[2] += 1
        #happy
        if (Y_lab[i,0]==1 and Y_val[i]>=0.25 and no_pictures[3]<10):
            indices[no_pictures[3]+30] = i
            no_pictures[3] += 1
        #fear
        if (Y_lab[i,3]==1 and Y_val[i]<0 and no_pictures[4]<10):
            indices[no_pictures[4]+40] = i
            no_pictures[4] += 1
        #neutral
        if (Y_lab[i,0]==1 and Y_val[i]>-0.25 and Y_val[i]<0.25 and Y_ars[i]>-0.2 and Y_ars[i]<0.2 and no_pictures[5]<10):
            indices[no_pictures[5]+50] = i
            no_pictures[5] += 1

    if (vis):
        # Visualize inputs
        j = 0
        for i in indices:
            #plt.subplot(4, 5, i+1)
            j += 1
            print(j, I[i])
            print(Y_lab[i])
            print(Y_val[i], Y_ars[i])
            plt.imshow(X[i, :, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
            plt.show()

    f.close()

if __name__ == "__main__":

    # Check AffectNet
    check_dataset("/scratch/facs_data/AffectNet/Manually_Annotated_Images/AffectNet_training_data_new.h5", False, 1000)