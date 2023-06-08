#!/bin/bash
cd /mnt/c/Users/cristian.guarachi/PycharmProjects/sequential_tEDRAM

DIR=dependencies/bin/activate
if [ -d "$DIR" ];
then
    source "$DIR"
else
    #python3.10 -m venv dependencies
    #source "$DIR"
    #python3.10 -m pip install numpy
    echo "$DIR does not exist"
fi

cd /mnt/c/Users/cristian.guarachi/PycharmProjects/sequential_tEDRAM/src

python3.10 train.py
