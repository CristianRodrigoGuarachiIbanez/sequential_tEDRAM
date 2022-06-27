from matplotlib.pyplot import figure, tight_layout, subplot, imshow, xticks, yticks, suptitle, show, pause, savefig, subplots
from math import ceil
from os import chdir, makedirs
from os.path import dirname, exists
from typing import List
from numpy import asarray, ndarray, uint8, linspace, pi, sin, abs, random
def show_4Darray_matplot(plot_image:ndarray, title:str, h_plot:int, v_plot:int, index:int, path_disparity = "./training_data/disparity_maps")->None:
    '''
    function to plot a 4D numpy array
    params: plot_image      -- 4D array to plot
            title           -- title of the plot
            h_plot          -- number of images in horizontal direction
            v_plot          -- number of images in vertical direction
    '''
    path = "/home/cristian/PycharmProjects/tEDRAM/tEDRAM2/training_data/disparity_maps"

    if not exists(path_disparity):
        makedirs(path_disparity)
    chdir(path_disparity)

    splot_img_count = h_plot * v_plot
    h_step = 0
    v_step = 0
    img_count = plot_image.shape[2] * plot_image.shape[3]
    length = ceil(img_count / splot_img_count)
    print("LENGTH",length)
    for i in range(length):
        figure(figsize=(13, 10))
        tight_layout()
        print(i)
        for j in range(splot_img_count):
            if (j + i * splot_img_count) < img_count:
                subplot(v_plot, h_plot, j + 1)
                imshow(plot_image[:, :, h_step, v_step].T, cmap='gray')
                xticks([])
                yticks([])
            else:
                break
            v_step += 1
            if v_step == plot_image.shape[3]:
                v_step = 0
                h_step += 1
        suptitle(title)
        tight_layout(pad=2.5, w_pad=0.5, h_pad=1.0)

        savefig(path_disparity + "/disparity_map_"+str(index)+".png")
        show()
        pause(0.05)
def _show_4Darray_plot(plot_image:ndarray, title:str)->None:
    # settings
    nrows, ncols = 7, 8  # array of sub-plots
    figsize = [12, 14]  # figure size, inches
    rows:int = 0
    cols: int = 0
    # ax enables access to manipulate each of subplots
    fig, ax = subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    #print("fig:", fig, "ax:", len(ax), "axi", ax[0])

    for i, axi in enumerate(ax.flat):
        img = plot_image[rows,cols,:,:]
        # create subplot and append to ax
        axi.imshow(img, cmap='gray')
        rowid = i // ncols
        colid = i % ncols
        axi.set_title("Row:" + str(rowid) + ", Col:" + str(colid))
        cols +=1
        if(cols == plot_image.shape[1]):
            cols =0
            rows+=1
        #print(rows, cols)
    xticks([])
    yticks([])
    suptitle(title)
    tight_layout(pad=2.5, w_pad=0.5, h_pad=1.0)
    show()  # finally, render the plot

def show_3Darray_plot(plot_image:ndarray, title:str)->None:
    path_disparity = "/scratch/gucr/tEDRAM2/training_data/disparity_maps"
    if not exists(path_disparity):
        makedirs(path_disparity)
    chdir(path_disparity)
    # settings
    nrows, ncols = 7, 1  # array of sub-plots
    figsize = [12, 14]  # figure size, inches


    # ax enables access to manipulate each of subplots
    fig, ax = subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i, axi in enumerate(ax.flat):

        #subplot(v_plot, h_plot, i + 1)
        img = plot_image[i,:,:]
        # create subplot and append to ax
        axi.imshow(img, cmap='gray')
        rowid = i // ncols
        colid = i % ncols
        axi.set_title("Row:" + str(rowid) + ", Col:" + str(colid))
    xticks([])
    yticks([])
    suptitle(title)
    tight_layout(pad=2.5, w_pad=0.5, h_pad=1.0)
    savefig(path_disparity + "/disparity_map_" + str(index) + ".png")
    show()  # finally, render the plot