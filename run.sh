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
    python3.10 -m venv dependencies
fi
# rm -rf "$DIR"
# python3 -m venv "$DIR_/dependencies"
source "$DIR/activate"

python3 -m pip install --upgrade pip
python3 -m pip install -e ./

test=0
src=0

while getopts st option; do
  case "${option}" in
    s) src=1;;
    t) test=1;;
    *) usage;;
  esac
done

if [ ${src} -eq 1 ]; then
  search_dir="$DIR_/src/sEDRAM"
  cd "$DIR_/src/stEDRAM"
  python3 start.py --gpu=1
elif [ ${test} -eq 1 ]; then
  search_dir="$DIR_/test"
  cd "$DIR_/test"
  for entry in "$search_dir"/*.py
  do
    echo "$entry"
    python3 "$entry"
  done
else
  cd "$DIR_/src/evaluation"
fi


# python3.10 train.py

pwd
