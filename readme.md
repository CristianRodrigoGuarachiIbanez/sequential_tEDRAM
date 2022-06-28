
# (t)EDRAM - (temporally) Enriched Deep Recurrent Visual Attention Model [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) [![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)


A TensorFlow adaptation of the temporally Enriched Deep Recurrent Visual Attention Model (tEDRAM) for visually detected collisions in the iCub-simulated 3D-environment using a self-builded data set of video sequences.

(See: https://doi.org/10.1109/WACV.2017.113 and https://julien-vitay.net/pdf/Forch2019.pdf)


Dependencies
------------
 * [tensorflow-gpu]
 * [h5py]
 * [cython]
 * [fuel] for inference

Installation
------------

Install Anaconda in *./local_directory*:

    cd ./local_directory
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

Install it in something like *./local_directory/miniconda3*, not in your AFS home. Let it append its path to your .bashrc (init). Upgrade it to be on the safe side:

	conda update -c defaults conda

Create a virtual environment (e.g. tf21) using python 3.8, activate it and install keras with tensorflow:

	conda create -n tf21 python=3.8
	conda activate tf21
	conda install tensorflow-gpu
	conda install keras
Download the repository sequential_tEDRAM in the local repository and navigate in the archive tEDRAM:

    /local_repository/sequential_tEDRAM/

Datasets
-------
## AffectNet
The raw AffectNet DB can be found [here](http://mohammadmahoor.com/affectnet/)

The file for training the network should be saved in archive training_data:

    /local_directory/sequentieal_tEDRAM/training_data/facs_data/AffectNet/AffectNet_train_data_keras.hdf5

Also, you can create the training file yourself by first compiling the full DB (~12GB) using 

    /local_directory/sequential_tEDRAM/create_dataset/generate_Anet.py

and then generating the training file (~3GB) with 
    
     /local_directory/sequential_tEDRAM/create_dataset/Anet_h5_2_h5.py

## Collision Data
The files for training the network could be found [here](http://ai.informatik.tu-chemnitz.de/gogs/gucr/collision_training_data.git) 

In order to use the training file your self, you should create a new folder with the name "training_data" in the local repository:

    /local_directory/sequential_tEDRAM/training_data/

Inside this folder, you should put the "collision_training_data.h5" file. 

Training
--------

You can train the model effectively only on GPU. To do so, check out whether the GPUs are in use:

	nvidia-smi

And run the script on a free GPU:

	python start.py --gpu=1

