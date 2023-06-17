#!/bin/bash

usage() { echo "Usage: $0 [-s <src>] [-t <test>]" 1>&2; exit 1; }

DIR_=$PWD

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
  usage
fi

