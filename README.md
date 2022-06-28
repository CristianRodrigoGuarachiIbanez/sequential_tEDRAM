            
# (t)EDRAM - (temporally) Enriched Deep Recurrent Visual Attention Model [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) [![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)


A TensorFlow adaptation of the temporally Enriched Deep Recurrent Visual Attention Model (See: https://doi.org/10.1109/WACV.2017.113 and https://julien-vitay.net/pdf/Forch2019.pdf) for a set of image sequences. This Implementation has been used for:

 - emotion recognition on non chronological ordered images 
 - detecting collisions visually in the iCub-simulated 3D-environment.

Dependencies
------------
 * [tensorflow-gpu]
 * [h5py]
 * [cython]
 * [fuel] for inference


Installation
------------

Install Anaconda in */scratch/your_username*:

    cd /scratch/your_username
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

Install it in something like */scratch/your_username/miniconda3*, not in your AFS home. Let it append its path to your .bashrc (init). Upgrade it to be on the safe side:

	conda update -c defaults conda

Create a virtual environment (e.g. tf21) using python 3.8, activate it and install keras with tensorflow:

	conda create -n tf21 python=3.8
	conda activate tf21
	conda install tensorflow-gpu
	conda install keras

If you are working on *globus* / *gyrus* / *tractus*, install the CUDA toolkit 10.1 instead:

	conda install tensorflow-gpu cudatoolkit=10.1 cudnn cupti
	conda install keras


Datasets
-------
## AffectNet
The raw AffectNet DB can be found under: 

    /scratch/facs_data/AffectNet/Manually_Annotated_Images/

The file for training the network should be in: 

    /scratch/facs_data/AffectNet/AffectNet_train_data_keras.hdf5

You can create the training file yourself by first compiling the full DB (~12GB) using *generate_Anet.py* and then generating the training file (~3GB) with *Anet_h5_2_h5.py*.

## Collision Data
The files for training the network could be found here: 

    /scratch/collision_data/collision_training_data.h5 

and 

    /scratch/collision_data/binocular_training_data.h5 

In order to use the training file your self you should create a new folder with the name "tEDRAM2/training_data" in your local repository.
Inside this folder, you should put the "collision_training_data.h5" file. 

Training
--------

You can train the model effectively only on GPU. To do so, check out whether the GPUs are in use:

	nvidia-smi

And run the script on a free GPU:

	python start.py --gpu=1