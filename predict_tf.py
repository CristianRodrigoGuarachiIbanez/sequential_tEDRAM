from __future__ import print_function
import os
from argparse import ArgumentParser
from numpy import ndarray, zeros, ones, array, asarray, vstack, dot, mean as Mean, square, std as Std, var, \
    empty, transpose, nan_to_num, float64, int64, load, uint8, float32, newaxis
from sequenceConstructor import SequenceConstructor
from config import config, datasets
from typing import List, Dict, Tuple, Sized, Union, TypeVar
from dataset_tools.fileLoader import LOADER, load_dataset
from models.tedramManager import tEDRAM_TF
from cython_modules.calculateACC.accuracyCalculator import AccuracyCalculator
from cython_modules.history_handler.valuesRecoverer import LossValuesRecoverer
#from cython_modules.data_augmentation.opencv_cmat import opencv_mat
from cython_modules.copy_img_arrays.copy_array import CopyNumpyArray
from visualization_tools.visualizationKit import TerminalGraphics
from dataset_tools.h5Writer import H5Writer
from cython_modules.visualizer import *
# default training parameters
_batch_size = 100
_model_id = 1
_n_steps = 10
# paths to the datasets
# python3.8 predict_tf.py --gpu=1 --data=2 --path=/scratch/gucr/tEDRAM2/outputs/output7_10/default/ --steps=10
def main(list_params: str, gpu_id: int, dataset_id: int, model_id: int, use_checkpoint_weights,
         load_path: str, batch_size: int, n_steps: int, glimpse_size: int, coarse_size: int, fc_dim: int,
         enc_dim: int, dec_dim: int, n_classes: int, clip_value: int, unique_emission: int,
         unique_glimpse: int, output_mode: int, use_init_matrix: int, headless,
         scale_inputs: float, normalize_inputs: bool, use_batch_norm: bool, dropout: int, weighting: int,
         iterations: int, zoom_factor: float):
    if (dataset_id < 5):
        n_classes = 6
    else:
        n_classes =7
    input_shape = load_dataset(dataset_id, config)
    glimpse_size: Tuple[int, int] = (glimpse_size, glimpse_size)
    coarse_size: Tuple[int, int] = (coarse_size, coarse_size)
    # ------------------------ select a GPU ----------------
    print("[Info] Using GPU", gpu_id)
    if(gpu_id == -1):
        print('[Error] You need to select a gpu. (e.g. python train.py --gpu=7)\n')
        exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # ----------------------- create the model ------------------
    print("\n[Info] Loading the model from:", load_path + ("model_weights" if use_checkpoint_weights == 0 else "checkpoint_weights"))
    print()
    # create the EDRAM model
    print("[Info] Creating the model...")
    tedram = tEDRAM_TF(image_shape=input_shape)
    if (model_id == 0):
        print('model ID 0 is not available');
    elif (model_id == 1):
        tedram.create_tedram_model(learning_rate=1, n_steps=n_steps, glimpse_size=glimpse_size, coarse_size=coarse_size, n_filters=128, filter_sizes=(3, 5),
                             n_features=fc_dim, RNN_size_1=enc_dim, RNN_size_2=dec_dim, n_classes=n_classes, output_mode=output_mode,
                             use_init_matrix=use_init_matrix, clip_value=clip_value, unique_emission=unique_emission, unique_glimpse=unique_glimpse,
                             bn=use_batch_norm, dropout=dropout, use_weighted_loss=False, localisation_cost_factor=1)
    else:
        print('[Error] Only model 1 is available!\n')
        exit()
    # -------------------- load weights -------------------------
    # loader: DataLoader = DataLoader()

    if (use_checkpoint_weights > 0):
        tedram.model.load_weights(load_path + 'checkpoint_weights')
    else:
        tedram.model.load_weights(load_path + 'model_weights')
    ###################################################################
    ############################ Loader ###############################
    ###################################################################
    loader = LOADER()
    # -------------------------- load paths -------------------------
    data_path: str = datasets[dataset_id]
    labels_path: str = datasets[0]  # 0: path_labels 6 labels
    # ------------------------- init data values ------------------------
    loader.load_image_data(data_path=data_path)
    loader.load_label_data(labels_path=labels_path)
    if (dataset_id == 1):
        loader.split_dataset_nongroup(dataset='scene_img', dataset_id=dataset_id)
    elif (dataset_id == 2):
        loader.split_dataset_nongroup(dataset="disparity_arrays_s", dataset_id=dataset_id)
    elif (dataset_id == 3):
        loader.split_dataset_nongroup(dataset="disparity_arrays_56", dataset_id=dataset_id)
    elif (dataset_id == 4):
        loader.split_dataset_nongroup(dataset="disparity_arrays_54", dataset_id=dataset_id)
    elif (dataset_id == 5):
        loader.split_dataset_nongroup(dataset="binocular_images", dataset_id=dataset_id)
    elif (dataset_id == 6):
        loader.split_dataset_nongroup(dataset="emotions", dataset_id=dataset_id)
    else:
        loader.split_dataset(group="binocular_image", dataset="left_img", dataset_id=dataset_id)

    # ----------------------  image data -----------------------
    features: ndarray = loader.get_test_images()
    print("FEATURES: ", features.shape, features.dtype, features.ndim, n_steps)
    # ---------------------  label data --------------------
    labels: ndarray = loader.get_test_labels()
    #print(len(labels))
    locations: ndarray = None;
    # ----------------------- normalize input data ------------------
    indices: SequenceConstructor;
    samples: array;
    n_test = loader.get_n_test()
    if (normalize_inputs):
        indices = SequenceConstructor(n_test, n_steps, n_steps)
        samples = indices.samples(features, 0, 1000) / scale_inputs  # [sorted(indices[:1000]), ...]/scale_inputs
        mean = Mean(samples, axis=0)
        sd = Std(samples, axis=0).clip(min=0.00001)
    else:
        mean = 0
        sd = 1
    print("[Info] Dataset Size\n")
    print(" using", iterations, "*", batch_size, "out of", n_test, "test examples")
    print("\n[Info] Data Dimensions\n")
    if (len(features.shape) < 4):
        print("  Image with 3 dimensions:", features.shape[0], "x", features.shape[1], "x", features.shape[2], )
        features = features[:, newaxis, :, :]
        print("  Images: ", features.shape[0], "x", features.shape[1], "x", features.shape[2], "x", features.shape[3])
    elif (len(features.shape) == 4):
        print("  Images:   ", features.shape[0], "x", features.shape[1], "x", features.shape[2], "x", features.shape[3], )
    elif (len(features.shape) == 5):
        print("  Images:   ", features.shape[0], "x", features.shape[1], "x", features.shape[2], "x", features.shape[3],"x", features.shape[4])
    elif (len(features.shape) > 5):
        print("  Images:   ", features.shape)

    print("  Labels:   ", labels.shape[0], "x", labels.shape[1])

    if (locations is not None):
        print("  Locations: ", locations.shape[1], "\n")
    else:
        print("  Locations: ", 6, "\n")

    # ------------------------ get sample data --------------------

    indices: SequenceConstructor = SequenceConstructor(n_test, n_steps, n_steps)
    start: int = 0;
    end: int = batch_size * iterations
    sample = asarray(indices.samples(features, start,  end), dtype=uint8)
    ##############################################################
    #################### prepare the minibatch ###################
    ##############################################################
    # ----------------------Add sequences dimension to input image ---------------------
    if(sample.ndim==4): #4 in 5 dim
        arrayCopy = CopyNumpyArray(sample,7)
        sample = arrayCopy.get_array()
        sample = asarray(sample, dtype=float)
        sample = transpose(sample, (0,1,3,4,2))

    if(sample.shape[-1]>7): # (N, 6, 120,160,1) rejected
        sample = transpose(sample, (0, 1, 3, 4, 2))

    print("COPIED:", sample.ndim, sample.shape)

    if ((scale_inputs != 1) and (scale_inputs != 0)):
        I = sample / scale_inputs
    if (normalize_inputs):
        I = (sample -mean) /sd
    else:
        I = sample
    # -------------------- transformation matrix with zoom paramters set to 1 ----------------
    A = zeros((batch_size * iterations, 6), dtype='float32')
    A[:, (0, 4)] = 1
    #########################################################
    # -------------------- initial RNN states ---------------
    #########################################################
    S1: ndarray = zeros((batch_size * iterations, enc_dim), dtype='float32')
    S2: ndarray = zeros((batch_size * iterations, dec_dim), dtype='float32')
    ##########################################################################
    ############################# biases  ####################################
    ##########################################################################

    if glimpse_size == (26, 26):
        B1 = ones((batch_size * iterations, 26, 26, 1), dtype='float32')
        B2 = ones((batch_size * iterations, 24, 24, 1), dtype='float32')
        B3 = ones((batch_size * iterations, 12, 12, 1), dtype='float32')
        B4 = ones((batch_size * iterations, 8, 8, 1), dtype='float32')
        B5 = ones((batch_size * iterations, 6, 6, 1), dtype='float32')
        B6 = ones((batch_size * iterations, 4, 4, 1), dtype='float32')
    else:
        B1 = ones((batch_size * iterations, 16, 16, 1), dtype='float32')
        B2 = ones((batch_size * iterations, 16, 16, 1), dtype='float32')
        B3 = ones((batch_size * iterations, 8, 8, 1), dtype='float32')
        B4 = ones((batch_size * iterations, 8, 8, 1), dtype='float32')
        B5 = ones((batch_size * iterations, 6, 6, 1), dtype='float32')
        B6 = ones((batch_size * iterations, 4, 4, 1), dtype='float32')
    ##########################################################################################
    # ------------------ concatenation of target outputs for every step ----------------------
    ###########################################################################################
    # print("Classification:", labels.shape)
    Y_cla: ndarray = indices.labels(labels, start, end)  # array(labels[samples, ...], dtype='float32')
    # print("Y Classification:", Y_cla.shape)
    Y_loc: ndarray = empty((0, 0))
    if (zoom_factor == 1):
        if (locations is None):
            Y_loc = zeros((batch_size * iterations, 6), dtype='float32')
            Y_loc[:, (0, 4)] = zoom_factor;
        else:
            Y_loc = indices.samples(locations, start, end);
    else:
        if (locations is None):
            Y_loc = zeros((batch_size * iterations, 6), dtype='float32')
            Y_loc[:, (0, 4)] = zoom_factor;
    # ------------------------ transpose img ------------------------------------
    if (I.shape[2] == 56 and I.shape[3] == 120):
        I = transpose(I, (0, 1, 3, 4, 2))
    elif (I.shape[2] == 7 and I.shape[3] == 120):
        I = transpose(I, (0, 1, 3, 4, 2))
        # print("from 4 dim (N, S, 7, 120, 160) will be : ", I.shape)
    elif (I.shape[2] == 7 and I.shape[3] == 8):
        pass
    inputs: Dict[str, ndarray] = dict();
    outputs: Dict[str, ndarray] = dict();
    # -------------------------------------------------------------------------------

    print("shape of images will be :", I.shape)
    if ((model_id == 1) or (model_id == 2)):
        inputs = {'input_image': I, 'input_matrix': A, 'initial_hidden_state_1': S1, 'initial_cell_state_1': S1,
                  'initial_cell_state_2': S2, 'b26': B1, 'b24': B2, 'b12': B3, 'b8': B4, 'b6': B5, 'b4': B6}
        outputs = {'classifications': Y_cla, 'localisations': Y_loc}
    elif (model_id == 3):
        inputs = {'input_image': I}
        outputs = {'classifications': Y_cla}
    # ------------------------- predicted values ------------------------
    predicted_labels, predicted_locations = tedram.model.predict(inputs, batch_size=batch_size, verbose=1)  # (192,11,6),(192,10,6)
    cla_values = Y_cla[:]
    pred_values = predicted_labels[:]

    #------------------------ reshape ------------
    """
    if (model_id==1):
        Y_cla = vstack([Y_cla[:, i, :] for i in range(0, n_steps)])  #
        if (output_mode):
            predicted_locations = vstack([predicted_locations[:,i,:] for i in range(0, n_steps+use_init_matrix)])# (192*11 , 6)
        if ((n_steps>1) and (headless==False)):
            predicted_labels = vstack([predicted_labels[:,i,:] for i in range(0, n_steps)]) # (192*10 , 6)
    """
    # ------------------------- save sample data and predictions ---------------
    ##h5file = h5py.File(load_path + 'predictions.h5', mode='w')
    group = ['features', 'locations', 'labels' ]
    dataset = [['ground_truth', 'predicted'], ['ground_truth', 'predicted'], ['ground_truth', 'predicted']]
    img = [[indices.samples(features, start, end), I], [Y_loc, predicted_locations], [Y_cla, predicted_labels]]
    h5file = H5Writer(load_path + 'predictions.h5')
    print('true',"features", indices.samples(features, start, end).shape)
    print('normalized',"features2", array(I, dtype='float32').shape)
    print('true',"locations", array(Y_loc, dtype='float32').shape)
    print('predicted', 'locations', array(predicted_locations, dtype='float32').shape)
    print('true','labels', array(Y_cla, dtype='float32').shape)
    print('predicted','labels', array(predicted_labels, dtype='float32').shape)
    h5file.saveDataIntoGroups(data=img, group=group,dataset=dataset)

    print("\n[INFO] Saved data to", load_path + 'predictions.h5', "\n")

    #################################################################
    ######################### some statistics #######################
    #################################################################
    # --------------------------- calculate Accuracy ------------------------------------
    predicted_labels = pred_values.astype(float64)
    Y_cla = cla_values.astype(int64)

    print("pred", predicted_labels.shape, "class", Y_cla.shape)
    acc = AccuracyCalculator(predicted_labels, Y_cla)
    print(acc.positive_values())
    print(acc.negative_values())
    print("ACC:", acc.calc_acc())
    print("SENSITIVITY:",acc.calc_sensit())
    print("SPECIFICITY:", acc.calc_spec())
    print("PRECISION:", acc.calc_prec())
    print("NPV:", acc.calc_NPV())
    print("FPR:", acc.calc_FPR())
    print("FNR:", acc.calc_FNR())
    print("F1:", acc.calc_f1())
    print("Matthews Correlation:", acc.calc_matthews_corr())

    path_loss = load_path + "/history/" #output7_10/default/history/" # r"/scratch/gucr2/tEDRAM2/outputs/output56_25_10/default/history/"
    print(" class loss: ")
    path = path_loss + "classifications_loss.npy"
    class_loss_data = LossValuesRecoverer(path.encode('UTF-8'))
    print(" loss data: ")
    path = path_loss + "loss.npy"
    loss_data = LossValuesRecoverer(path.encode('UTF-8'))
    print(" class categorical acc: ")
    path = path_loss+"classifications_categorical_accuracy.npy"
    class_cat_acc = LossValuesRecoverer(path.encode('UTF-8'))
    print(" history: ")
    history = load(path_loss+ "history.npy", allow_pickle=True)
    print(" loc loss: ")
    path = path_loss+"localisations_loss.npy"
    loc_loss = LossValuesRecoverer(path.encode('UTF-8'))
    print(" val cla: ")
    path = path_loss+"val_classifications_categorical_accuracy.npy"
    val_class = LossValuesRecoverer(path.encode('UTF-8'))
    print(" val cla loss: ")
    path = path_loss+"val_classifications_loss.npy"
    val_cla_loss = LossValuesRecoverer(path.encode('UTF-8'))
    print(" val loc loss: ")
    path = path_loss + "val_localisations_loss.npy"
    val_loc_loss = LossValuesRecoverer(path.encode('UTF-8'))
    print(" val loss: ")
    path = path_loss + "val_loss.npy"
    val_loss = LossValuesRecoverer(path.encode('UTF-8'))
    ####################################################################
    ########################## plot ####################################
    ####################################################################

    X = loss_data.data()
    Y = val_loss.data()
    visualise_plot(X, "Trainings- bzw. Validierungsverlust" , "Epoche", "Mittlerer quadratischer Fehler ", "./loss1.png", Y )
    #grpcs = TerminalGraphics(100,10,(0,1),(0,1))
    #grpcs.plot_config(X,Y,c=25, l="loss vs val loss ")
    X = loc_loss.data()
    Y = val_loc_loss.data()
    visualise_plot(X, "Lokalisierungsverlust: tEDRAM + Aug." , "Epoche", "Mittlerer quadratischer Fehler ", "./loss2.png", Y)
    #grpcs.plot_config(X, Y, c=50, l="loc loss vs val loc loss")

    X = class_cat_acc.data()
    Y = val_class.data()
    visualise_plot(X, "Klassifizierungsgenauigkeit: tEDRAM + Aug.", "Epoche", "Kategoriale Kreuzentropie", "./loss4.png", Y)

if __name__ == "__main__":
    # argument list
    parser = ArgumentParser(description="Generate predictions based on trained EDRAM network")
    parser.add_argument("--l", "--list_params", type=str, nargs='?', default='none', const='none',
                        dest='list_params', help="Show a parameter list")
    # high-level options
    parser.add_argument("--gpu", type=int, nargs='?', default=-1, const=7,
                        dest='gpu_id', help="Specifies the GPU.")
    parser.add_argument("--data", type=int, default=1,
                        dest='dataset_id', help="ID of the test data set or path to the dataset.")
    parser.add_argument("--model", type=int, default=_model_id,
                        dest='model_id', help="Selects model type.")
    parser.add_argument("--checkpoint", "--checkpoint_weights", type=int, default=1,
                        dest='use_checkpoint_weights', help="Whether to load checkpoint weights.")
    parser.add_argument("--path", "--load_path", type=str, default='.',
                        dest='load_path', help="Path for loading the model weights.")
    parser.add_argument("--bs", "--batch_size", type=int, default=_batch_size,
                        dest="batch_size", help="Size of each mini-batch")
    parser.add_argument("--iter", "--iterations", type=int, default=1,
                        dest="iterations", help="Number of mini-batches to process.")
    # model structure
    parser.add_argument("--steps", type=int, default=_n_steps,
                        dest="n_steps", help="Step size for digit recognition.")
    parser.add_argument("--glimpse", "--glimpse_size", "-a", type=int, default=26,
                        dest='glimpse_size', help="Window size of attention mechanism.")
    parser.add_argument("--coarse_size", type=int, default=12,
                        dest='coarse_size', help="Size of the rescaled input image for initialization of the network.")
    parser.add_argument("--fc_dim", type=int, default=1024,
                        dest="fc_dim", help="Fully connected dimension.")
    parser.add_argument("--enc_dim", type=int, default=512,
                        dest="enc_dim", help="Encoder RNN state dimension.")
    parser.add_argument("--dec_dim", type=int, default=512,
                        dest="dec_dim", help="Decoder  RNN state dimension.")
    parser.add_argument("--classes", type=int, default=6,
                        dest="n_classes", help="Number of classes for recognition.")
    parser.add_argument("--clip", "--clip_value", type=float, default=1.0,
                        dest="clip_value", help="Clips Zoom Value in Spatial Transformer.")
    parser.add_argument("--unique", "--unique_emission", type=int, default=0,
                        dest="unique_emission", help="Inserts unique emission layer")
    parser.add_argument("--unique_glimpse", type=int, default=0,
                        dest="unique_glimpse", help="Inserts unique first glimpse layer")
    # output options
    parser.add_argument("--mode", "--output_mode", type=int, default=1,
                        dest="output_mode", help="Output last step or all steps.")
    parser.add_argument("--use_init", "--use_init_matrix", type=int, default=1,
                        dest="use_init_matrix", help="Whether to use the init matrix as output.")
    parser.add_argument("--headless", type=int, default=0,
                        dest="headless", help="Whether to use a dense classifier on all timesteps in parallel.")
    # normalisation of inputs and model layers
    parser.add_argument("--scale", "--scale_inputs", type=float, default=255,
                        dest="scale_inputs", help="Scaling Factor for Input Images.")
    parser.add_argument("--normalize", "--normalize_inputs", type=int, default=0,
                        dest="normalize_inputs", help="Whether to normalize the input images.")
    parser.add_argument("--bn", "--use_batch_norm", type=int, default=1,
                        dest="use_batch_norm", help="Whether to use batch normalization.")
    parser.add_argument("--do", "--dropout", type=float, default=0,
                        dest="dropout", help="Whether to use dropout (dropout precentage).")
    # pertaining to the accuracy computation
    parser.add_argument("--weighting", type=int, nargs='?', default=0, const=1,
                        dest='weighting', help="Weighting applied for accuracy from average model predictions.")
    parser.add_argument("--zoom", "--zoom_factor", type=float, default=1,
                        dest='zoom_factor', help="Targte Zoom Factor.")
    args = parser.parse_args()


    def list_args(**args):
        for i, arg in enumerate(args.items()):
            if (i <= 2):
                if arg[1] == 'none' and i == 0:
                    break
                elif i == 0:
                    print("\n[Info] Training Parameters\n")
            else:
                print(' ' * (24 - len(arg[0])), arg[0], "=", arg[1])


    list_args(**vars(args))
    main(**vars(args))
