from typing import List
from numpy import ndarray, zeros, argmax, dot, asarray, mean as Mean, square, seterr, nan_to_num
import cython
seterr(divide='ignore', invalid='ignore')
class Statistics:
    hist: ndarray;
    acc: ndarray ;
    acc_avg: ndarray;
    pos: ndarray;
    zoom: ndarray ;
    mse_pos: ndarray;
    mse_zoom: ndarray;
    val_ars: ndarray;
    mse_val: ndarray;
    mse_ars: ndarray;
    acc_mean:ndarray;
    predicted_labels_avg: ndarray;
    headless:int;
    n_steps:int;
    def __init__(self, n_classes:int, headless:int, n_steps:int, use_init_matrix:int )->None:
        self.hist = zeros(n_classes, dtype='int')
        self.acc = zeros((1 if headless else n_steps, n_classes), dtype='int') # 10,6
        self.acc_avg= zeros((1 if headless else n_steps, n_classes), dtype='int')
        self.pos = zeros((n_steps + use_init_matrix, n_classes, 2), dtype='float') #10+1,6,2
        self.zoom = zeros((n_steps + use_init_matrix, n_classes, 2), dtype='float')
        self.mse_pos = zeros((n_steps + use_init_matrix, n_classes), dtype='float')
        self.mse_zoom= zeros((n_steps + use_init_matrix, n_classes), dtype='float')
        self.val_ars = zeros((n_steps, n_classes, 2), dtype='float')
        self.mse_val= zeros((n_steps, n_classes), dtype='float')
        self.mse_ars = zeros((n_steps, n_classes), dtype='float')
        self.acc_mean = zeros(1 if headless else n_steps)
        self.predicted_labels_avg = None;
        self.n_steps = n_steps; # 10
        self.headless=headless;
        self.use_init_matrix = use_init_matrix
        self.n_classes = n_classes
    def calculate_acc_avg(self, batch_size:int, Y_cla:ndarray, predicted_labels:ndarray)->None:
        i:cython.int;
        j:cython.int;
        # average predictions per step in inverted order
        for j in range(0, self.n_steps):
            k = 0
            self.predicted_labels_avg = predicted_labels[(self.n_steps - 1) * batch_size:(self.n_steps) * batch_size, :]
            for k in range(1, j + 1):
                self.predicted_labels_avg += predicted_labels[(self.n_steps - 1 - k) * batch_size:(self.n_steps - k) * batch_size, :]

            self.predicted_labels_avg /= k + 1
            for i in range(0, batch_size):
                # count correct classifications per class
                if argmax(Y_cla[i, :]) == argmax(self.predicted_labels_avg[i, :]):
                    self.acc_avg[j, :] = self.acc_avg[j, :] + Y_cla[i, :]

    def calculate_occurencies(self, batch_size:int, Y_cla:ndarray, predicted_labels:ndarray)->None:
        """

        :param batch_size: integer value -> 192
        :param Y_cla: multidimensional array
        :param predicted_labels:
        :return: None
        """
        i: cython.int
        j: cython.int
        for i in range(0, batch_size):
            # count class occurences
            self.hist = self.hist + Y_cla[i, :]
            # count correct classifications per class
            for j in range(0, 1 if(self.headless!=0) else self.n_steps):
                if (argmax(Y_cla[i, :]) == argmax(predicted_labels[i + j * batch_size, :])):
                #if(argmax(Y_cla[i,j])==argmax(predicted_labels[i,j])):
                    #print("occurencies",i, j, self.acc[j,:].shape, Y_cla[i,:].shape )
                    self.acc[j, :] = self.acc[j,:] + Y_cla[i,:]
    def calculate_accuracy(self,batch_size:int, Y_cla:ndarray, Y_loc:ndarray, predicted_locations:ndarray )->None:
        # compute accuracy
        i:cython.int
        j:cython.int
        k:cython.int
        #print(self.hist, self.acc)
        for k in range(0, 1 if self.headless else self.n_steps):
            self.acc_mean[k] = dot(self.hist / batch_size, nan_to_num(self.acc[k, :] / self.hist))
        #print(self.acc_mean)

        self.hist[self.hist == 0] = 0.00000001
        self.acc = asarray(self.acc * 100 / (self.hist), dtype='int') / 100
        self.acc_avg = asarray(self.acc_avg * 100 / (self.hist), dtype='int') / 100
        self.hist[self.hist < 1] = 0
        # compute bb info per class and mse
        #print(self.hist, self.acc, self.acc_avg)
        # 10 + 1, 6
        for j in range(0, self.n_steps + self.use_init_matrix):
            for i in range(0, self.n_classes):
                # (11, 6, 2) (11, 6, 2) (11, 6) (11, 6) (8, 11, 6) (8, 10, 6) (8, 6)
                # (11, 6, 2) (11, 6, 2) (11, 6) (11, 6) (88, 6) (80, 6) (8, 6)
                print(self.pos.shape, self.zoom.shape, self.mse_pos.shape, self.mse_zoom.shape, predicted_locations.shape, Y_cla.shape, Y_loc.shape)
                self.pos[j, i, :] = Mean(predicted_locations[j * batch_size:(j + 1) * batch_size, :][Y_cla[j * batch_size:(j + 1) * batch_size, i] == 1, :][:, (2, 5)], axis=0)

                self.zoom[j, i, :] = Mean(predicted_locations[j * batch_size:(j + 1) * batch_size, :][Y_cla[j * batch_size:(j + 1) * batch_size, i] == 1, :][:, (0, 4)], axis=0)

                self.mse_pos[j, i] = Mean(square(Y_loc[Y_cla[j * batch_size:(j + 1) * batch_size, i] == 1, :][:, (2, 5)] - predicted_locations[j * batch_size:(j + 1) * batch_size, :][Y_cla[j * batch_size:(j + 1) * batch_size, i] == 1, :][:, (2, 5)]))

                self.mse_zoom[j, i] = Mean(square(Y_loc[Y_cla[j * batch_size:(j + 1) * batch_size, i] == 1, :][:, (0, 4)] - predicted_locations[j * batch_size:(j + 1) * batch_size, :][Y_cla[j * batch_size:(j + 1) * batch_size, i] == 1, :][:, (0, 4)]))

