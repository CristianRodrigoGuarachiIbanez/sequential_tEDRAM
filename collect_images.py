from cython_modules.heatmaps.heat_maps.heatmap import HEATMAPS
from os import listdir
from numpy import asarray, uint8
from cv2 import imwrite, imread, IMREAD_COLOR
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sys import exit
"""
path =b"./training_data/collisions/"
output = b"./images/"
output2 =b"./training_data/scene_images/"
dir = listdir(path)
masks = listdir(output)
if(len(dir)!=len(masks)):
    print("len directories {} vs len masks {}".format(len(dir), len(masks)))
    exit()
hm = HEATMAPS(output,masks,path,dir,120, 240, 13, 110, 60, 187)

heatmaps = hm.get_heatmaps(97, 127, False)
for i in range(heatmaps.shape[0]):
    file = output2.decode("utf-8")+dir[i].decode("utf-8") #"heatmap_"+str(i)+".png"
    print("savin to ->",file)
    imwrite(file, heatmaps[i])
print(heatmaps.shape)
"""

path =r"./training_data/final/"
directory = sorted(listdir(path))

images =[]
frame = []
last =[]
counter = 0
comp = str(counter)
for i in range(len(directory)):
    #print("directory" , directory[i])
    flag = directory[i].replace(".", "_").split("_")[-2]
    if(flag.startswith(comp)):
        if(len(flag)<3):
            #print(flag,counter, comp)
            #images.append(path+directory[i])
            img = imread(path+directory[i], IMREAD_COLOR)
            images.append(img)
        else:
            img = imread(path + directory[i], IMREAD_COLOR)
            last.append(img)
            #last.append(path+directory[i])
    if(len(images)==10):
        images.append(last.pop())
        frame.append(images)
        images =[]
        counter +=1
        comp = str(counter)
        #print(flag, counter, comp)

frame = asarray(frame, dtype=uint8)
print( images)
print(frame.shape)
rows = 6
columns = 10

w = 10
h = 10
fig = plt.figure(figsize=(10, 10))
#fig, ax = plt.subplots(6, 11, figsize=(6,6))
gs1 = gridspec.GridSpec(4, 4)
gs1.update(wspace=0.025, hspace=0.0) # set the spacing between axes.
columns = 11
rows = 6
counter =1
for i in range(rows):
    for j in range(columns):

        img = frame[i,j]
        # ax[i,j].imshow(img)
        # ax[i,j].set_xticks([])
        # ax[i, j].set_yticks([])
        # fig.set_figheight(15)
        # fig.set_figwidth(15)
        # fig.tight_layout(h_pad=0, w_pad=0)
        ax = plt.subplot(rows, columns, counter)
        counter+=1
        ax.imshow(img)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis("off")
        ax.set_aspect('equal')
plt.subplots_adjust(wspace=0, hspace=0)
        #plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.1,hspace=0)
#fig.tight_layout()
#plt.subplot_tool()
plt.savefig("./image_final")