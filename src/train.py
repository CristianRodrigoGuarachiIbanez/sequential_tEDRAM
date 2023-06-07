import os
import random

from logging import info, basicConfig, INFO
basicConfig(filemode='info.log', level=INFO, format='%(levelname)s:%(message)s')

from typing import List, Tuple, Dict
from numpy import ndarray, asarray, save, mean, std, newaxis
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau
from os.path import exists
from os import makedirs
from src.config import config, datasets
from src.batch_generator import batch_generator
from models.tedram_model import tedram_model
from dataset_tools.fileLoader import LOADER, load_dataset
from models.tedramManager import tEDRAM_TF

def train(list_params:str, gpu_id:int, dataset_id:int, model_id:int, load_path:str, save_path:str,
         batch_size:int, learning_rate:float, n_epochs:int, augment_input:bool, rotation, n_steps:int, glimpse_size,
         coarse_size:int, fc_dim:int, enc_dim:int, dec_dim:int, n_classes:int, output_mode:int, use_init_matrix:int,
         headless:int, emission_bias:int, clip_value:int, unique_emission:int, unique_glimpse:int, scale_inputs:float,
         normalize_inputs:bool,use_batch_norm:bool, dropout:int, use_weighted_loss:int,
         localisation_cost_factor:float, zoom_factor:float):

    glimpse_size = (glimpse_size, glimpse_size)
    coarse_size = (coarse_size, coarse_size)

    # train on CDS?
    if (dataset_id ==6):
        n_classes = 7 # ------ OUTPUT
    elif(model_id==2):
        n_classes = 1
    else:
        n_classes = 4 # 6 oder 4
    # train on high-res input?
    input_shape:Tuple = load_dataset(dataset_id, config)
    print("input shape: ", input_shape, "dateset ID: ", dataset_id)

    # create output directory
    save_path = './output/'+save_path+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # select a GPU
    print("\n[Info] Using GPU", gpu_id)
    if gpu_id == -1:
        print('[Error] You need to select a gpu. (e.g. python train.py --gpu=2)\n')
        exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # create the EDRAM model
    print("[Info] Creating the model...")
    tedram = tEDRAM_TF(image_shape=input_shape)
    if (load_path != '.'):
        model_path = load_path + 'model.h5'
        print("[Info] Loading the model from:", model_path,"\n")
        try:
            tedram.set_model(load_model(model_path))
        except:
            tedram.create_tedram_model(learning_rate=learning_rate, n_steps=n_steps,glimpse_size=glimpse_size, coarse_size=coarse_size, n_filters=128, filter_sizes=(3,5), n_features=fc_dim,
                                RNN_size_1=enc_dim, RNN_size_2=dec_dim, n_classes=n_classes, output_mode=output_mode, use_init_matrix=use_init_matrix,
                                emission_bias=emission_bias, clip_value=clip_value, unique_emission=unique_emission, unique_glimpse=unique_glimpse,
                                bn=use_batch_norm, dropout=dropout, use_weighted_loss=use_weighted_loss, localisation_cost_factor=localisation_cost_factor);
            tedram.model.load_weights(load_path+'model_weights.h5');
    else:
        if (model_id==1):
            tedram.create_tedram_model(learning_rate=learning_rate, n_steps=n_steps, glimpse_size=glimpse_size, coarse_size=coarse_size, n_filters=128, filter_sizes=(3, 5), n_features=fc_dim,
                                       RNN_size_1=enc_dim, RNN_size_2=dec_dim, n_classes=n_classes, output_mode=output_mode, use_init_matrix=use_init_matrix,
                                       emission_bias=emission_bias, clip_value=clip_value, unique_emission=unique_emission, unique_glimpse=unique_glimpse,
                                       bn=use_batch_norm, dropout=dropout, use_weighted_loss=use_weighted_loss, localisation_cost_factor=localisation_cost_factor);
        elif (model_id ==2):
            tedram.create_one_output_model(learning_rate=learning_rate, n_steps=n_steps, glimpse_size=glimpse_size, coarse_size=coarse_size, n_filters=128, filter_sizes=(3, 5), n_features=fc_dim,
                                       RNN_size_1=enc_dim, RNN_size_2=dec_dim, n_classes=n_classes, output_mode=output_mode, use_init_matrix=use_init_matrix,
                                       emission_bias=emission_bias, clip_value=clip_value, unique_emission=unique_emission, unique_glimpse=unique_glimpse,
                                       bn=use_batch_norm, dropout=dropout, use_weighted_loss=use_weighted_loss, localisation_cost_factor=localisation_cost_factor)
        else:

            print('[Error] Only model 1 is available!\n')
            exit()

    # model summary
    if(list_params=='all' and model_id==3):
        tedram.model.get_layer('edram_cell').summary()
    elif (list_params!='none'):
        tedram.get_model().summary()
    ###########################################################
    ######### recover the data paths load the data ###########
    ##########################################################
    data_path: str = datasets[dataset_id]
    if(dataset_id==6):
        labels_path = datasets[-2]
    elif(model_id==2):
        labels_path = datasets[-1]
    else:
        labels_path: str = datasets[0] # 0: path_labels 6 oder 4 labels

    print("\n[Info] Opening", data_path)

    # initialize the declared image and label data
    loader:LOADER = LOADER()

    ############################################
    ################ load the image data ######
    ############################################
    loader.load_image_data(data_path)
    # ------------- load labels 
    loader.load_label_data(labels_path)
    ###############################################################
    ################### specify the arm side for the labels ######
    ################### split into train and test set ############
    ###############################################################

    if(dataset_id==1):
        loader.split_dataset_nongroup(dataset='scene_img', dataset_id=dataset_id)
    elif(dataset_id==2):
        loader.split_dataset_nongroup(dataset="disparity_arrays_s", dataset_id=dataset_id)
    elif (dataset_id == 3):
        loader.split_dataset_nongroup(dataset="disparity_arrays_56", dataset_id=dataset_id)
    elif(dataset_id ==4):
        loader.split_dataset_nongroup(dataset="disparity_arrays_54", dataset_id=dataset_id)
    elif(dataset_id ==5):
        loader.split_dataset_nongroup(dataset="binocular_images", dataset_id=dataset_id)
    elif(dataset_id==6):
        loader.split_dataset_nongroup(dataset="emotions", dataset_id=dataset_id)
    else:
        loader.split_dataset(group="binocular_image", dataset="left_img", dataset_id=dataset_id)

    train_images:ndarray=loader.get_train_images();
    test_images:ndarray=loader.get_test_images();
    train_labels:ndarray=loader.get_train_labels();
    test_labels:ndarray=loader.get_test_labels()
    train_locations: ndarray=loader.get_train_locations();
    test_locations:ndarray=loader.get_test_locations()
    ##################################################################################################
    ################## define the length of the set according the predefined indices ##################
    ##################################################################################################
    # normalize input data
    if(normalize_inputs):
        indices = list(range(loader.get_n_train()))
        random.shuffle(indices)
        samples = train_images[sorted(indices[:1000]), ...]/scale_inputs

        train_mean = mean(samples, axis=0)
        train_sd = std(samples, axis=0).clip(min=0.00001)

        indices = list(range(loader.get_n_test()))
        random.shuffle(indices)
        samples = test_images[sorted(indices[:1000]), ...]/scale_inputs

        test_mean = mean(samples, axis=0)
        test_sd = std(samples, axis=0).clip(min=0.00001)
    else:
        train_mean = 0
        train_sd = 1
        test_mean = 0
        test_sd = 1

    print("[Info] Dataset Size:\n")
    print(" ",loader.get_n_train(), "training examples")
    print(" ", loader.get_n_test(), "test examples")

    print("\n[Info] Data Dimensions\n")
    if(len(train_images.shape)<4):
        print("  Image with 3 dimensions:", train_images.shape[0],"x", train_images.shape[1], "x", train_images.shape[2],  )
        train_images = train_images[:,newaxis,:,:]
        print("  Images: ", train_images.shape[0],"x", train_images.shape[1], "x", train_images.shape[2], "x", train_images.shape[3])
    elif(len(train_images.shape)==4):
        print("  Images:   ", train_images.shape[0],"x", train_images.shape[1], "x", train_images.shape[2], "x", train_images.shape[3],)
    elif(len(train_images.shape)==5):
        print("  Images:   ", train_images.shape[0], "x", train_images.shape[1], "x", train_images.shape[2], "x", train_images.shape[3], "x", train_images.shape[4])
    elif(len(train_images.shape)>5):
        print("  Images:   ", train_images.shape)

    print("  Labels:   ", train_labels.shape[0], "x", train_labels.shape[1])

    if(train_locations is not None):
        print("  Locations:", train_locations.shape[1],"\n")
    else:
        print("  Locations:", 6,"\n")
    #################################################
    #################### create callbacks ###########
    #################################################
    history = History()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.333, patience=1, min_lr=0.00001, verbose=0)
    checkpoint = ModelCheckpoint(filepath=save_path+'checkpoint_weights', monitor='val_loss', save_best_only=True, save_weights_only=True)

    # create data generator for data augmentation
    datagen = None
    if augment_input:
        datagen = True
    # train the model
    try:
        hist = tedram.model.fit(
            x=batch_generator(dataset_size=loader.get_n_train(), batch_size=batch_size, init_state_size=(enc_dim, dec_dim), n_steps=n_steps,
                              features=train_images, labels=train_labels, locations=train_locations, augment=datagen,
                              scale=scale_inputs, normalize=normalize_inputs, mean=train_mean, std=train_sd, mode=output_mode, mode2=use_init_matrix, mode3=headless,
                              model_id=model_id, glimpse_size=glimpse_size, zoom=zoom_factor),
            steps_per_epoch=int(loader.get_n_train() / batch_size),
            epochs=n_epochs,
            verbose=1,
            callbacks=[history, reduce_lr, checkpoint],
            validation_data=batch_generator(dataset_size=loader.get_n_test(), batch_size=batch_size, init_state_size=(enc_dim, dec_dim), n_steps=n_steps, features=test_images, labels=test_labels,
                                            locations=test_locations, augment=None, scale=scale_inputs, normalize=normalize_inputs, mean=test_mean, std=test_sd, mode=output_mode,
                                            mode2=use_init_matrix, mode3=headless, model_id=model_id, glimpse_size=glimpse_size, zoom=zoom_factor),
            validation_steps=int(loader.get_n_test() / batch_size), use_multiprocessing=False
        )

    except KeyboardInterrupt:
        pass

    # save the history
    if not(exists(save_path+'/history')):
        makedirs(save_path+'/history')
    save(save_path+'/history/history', history.history)
    for key in history.history.keys():
        save(save_path+'/history/'+str(key),history.history[key])

    # save the model
    print('\n[Info] Saving the model...')

    tedram.model.save(save_path+'/model', save_format='tf')
    tedram.model.save_weights(save_path+'/model_weights', save_format='tf')

