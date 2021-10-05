#!/usr/bin/env bash

source=$1
target=$2
arch=$3
data_dir=$4

if [ $# -ne 4 ]
 then
   echo "Arguments error: <source> <target> <arch> <data_dir>"
   exit 1
fi


python3 examples/train_baseline.py -ds ${source} -dt ${target} -a ${arch} \
--logs-dir logs/${arch}_strong_baseline/${source}-TO-${target} --data-dir ${data_dir} --use-xbm

