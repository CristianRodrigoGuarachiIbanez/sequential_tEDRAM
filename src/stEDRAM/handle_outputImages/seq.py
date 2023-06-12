from cython_modules.heatmaps.heat_maps.heatmap import HEATMAPS
from os import listdir
from numpy import asarray, uint8
from cv2 import imwrite, imread, IMREAD_COLOR
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sys import exit
def visualize(columns, frame):

    fig = plt.figure(figsize=(10, 10))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0.025, hspace=0.0) # set the spacing between axes.
    counter =1
    for i in range(columns):
        img = frame[:]

        ax = plt.subplot(1, columns, counter)
        counter+=1
        ax.imshow(img)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis("off")
        ax.set_aspect('equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("./images")

path = r"src/training_dataset/collisions/"
output = r"./images/"
#output2 =b"./training_dataset/scene_images/"
dir = listdir(path)
for i in range(len(dir)):
    flag = dir[i].replace("_",".").split(".")[1]
    if(flag.startswith("0")):
        img = imread(path + dir[i])
        img = img[13: 110, 60:187]
        file = output+dir[i] #"heatmap_"+str(i)+".png"
        print("savin to ->",file)
        visualize(10,img)
        #imwrite(file, img)
print(img.shape)
