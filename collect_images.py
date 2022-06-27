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


def visualize(frame, rows, columns, file="./image_final_1.png"):
    fig = plt.figure(figsize=(20, 20))
    # fig, ax = plt.subplots(6, 11, figsize=(6,6))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0.025, hspace=0.0)  # set the spacing between axes.
    # columns = 6
    # rows = 11
    counter = 1
    for i in range(rows):
        for j in range(columns):
            img = frame[i, j]
            # ax[i,j].imshow(img)
            # ax[i,j].set_xticks([])
            # ax[i, j].set_yticks([])
            # fig.set_figheight(15)
            # fig.set_figwidth(15)
            # fig.tight_layout(h_pad=0, w_pad=0)
            ax = plt.subplot(rows, columns, counter)
            counter += 1
            ax.imshow(img)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.axis("off")
            ax.set_aspect('equal')
    plt.subplots_adjust(wspace=0.1, hspace=0.0)
    # plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.1,hspace=0)
    # fig.tight_layout()
    # plt.subplot_tool()
    plt.savefig(file)

path =r"./training_data/finalDM/finaldm4/"
directory = sorted(listdir(path))
print("directory files ->", directory, len(directory))
images =[]
rows=[]
cols=[]
last = []
frame = []

counter = 0
counter2 = 0

r = 7
c=8
comp = 0
comp2 = 0
i=0
local = []
r = [11,22, 33, 44]
while(True):
    flag2= directory[i].replace(".", "_").split("_")[-3]
    flag1 = directory[i].replace(".", "_").split("_")[-2]
    print(flag2, flag1, len(local), "comp -> ", comp, comp2, "index ->", i, "total ->", len(directory))
    if(flag2.endswith(str(comp2)) and flag2.startswith(str(comp)) and flag1=="0"):# and flag.endswith(comp2) ):#and )
        print(directory[i])
        local.append(directory[i])
        comp2+=1
        if(len(local) in r):
            comp+=1
            comp2 = 0
            if(comp>3):
                break

        if(len(local)==56):
            break
        else:
            KeyboardInterrupt()
    if(i<(len(directory)-1)):
        i += 1
    else:
        i=0
        counter+=1
        if(counter >4):
            import sys
            sys.exit()
print("total -> ", local, len(local))
def sorting_7x8(local):
    local = sorted(local)
    l,l1,l2,l3,l4,l5,l6 =[],[],[],[],[],[],[]
    for i in range(len(local)):
        flag = local[i].replace(".", "_").split("_")[-2]
        if(0<=int(flag)<=7):
            l.append(local[i])
        elif(8<=int(flag)<=15):
            l1.append(local[i])
        elif(16<=int(flag)<=23):
            l2.append(local[i])
        elif (24 <= int(flag) <= 31):
            l3.append(local[i])
        elif (32 <= int(flag) <= 39):
            l4.append(local[i])
        elif (40 <= int(flag) <= 47):
            l5.append(local[i])
        elif (48 <= int(flag) <= 55):
            l6.append(local[i])
    return [sorted(l), sorted(l1), sorted(l2), sorted(l3), sorted(l4), sorted(l5), sorted(l6)]

def aorting_4x10(local):
    local = sorted(local)
    l, l1, l2, l3 = [], [], [], []
    for i in range(len(local)):
        if (0<=i<=10):
            l.append(local[i])
        elif (11<=i<= 21):
            l1.append(local[i])
        elif (22 <=i <= 32):
            l2.append(local[i])
        elif (33 <= i <= 43):
            l3.append(local[i])

    return [sorted(l), sorted(l1), sorted(l2), sorted(l3)]

ll = aorting_4x10(local)
print(len(ll[0]))
for j in range(len(ll)):
    for k in range(len(ll[0])):
        img = imread(path + ll[j][k], IMREAD_COLOR)
        images.append(img)

    if (len(images) > 0):
        rows.append(images)
        images = []
frame =asarray(rows)
print(frame.shape)
visualize(frame, 4, 11, file="./hm_"+str(comp) +str(comp2)+".png")

#frame = asarray(frame, dtype=uint8)
#print( images)
#print("frame",frame.shape)

#for j in range(frame.shape[0]):
    #visualize(frame[j], 7, 8, "img_" + str(len(frame)) + ".png")