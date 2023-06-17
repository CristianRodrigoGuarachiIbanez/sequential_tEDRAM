#!/bin/bash

usage() { echo "Usage: $0 [-s <src>] [-t <test>]" 1>&2; exit 1; }
# for linux
DIR_UBUNTU=/home/cristian/PycharmProjects/tEDRAM/sequential_tEDRAM
# fro windows wsl
DIR_WIN=/mnt/c/Users/cristian.guarachi/PycharmProjects/sequential_tEDRAM

if [ -d "$DIR_UBUNTU" ];
then
  cd "$DIR_UBUNTU"
else
  cd "$DIR_WIN"
fi

DIR_=$PWD

DIR="$DIR_/dependencies/bin"
if [ ! -d "$DIR" ];
then
    echo "not virtual environment"
    python3.10 -m venv "$DIR_/dependencies"
fi
# rm -rf "$DIR"
# python3 -m venv "$DIR_/dependencies"
source "$DIR/activate"

python3 -m pip install --upgrade pip

python3 -m pip install -e ./