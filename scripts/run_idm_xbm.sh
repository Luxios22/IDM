#!/usr/bin/env bash

source=$1
target=$2
arch=$3
stage=$4
mu1=$5
mu2=$6
mu3=$7
data_dir=$8


if [ $# -ne 8 ]
 then
   echo "Arguments error: <source> <target> <arch> <stage> <mu1> <mu2> <mu3> <data_dir>"
   exit 1
fi


python3 examples/train_idm.py -ds ${source} -dt ${target} -a ${arch} \
--logs-dir logs/${arch}_xbm/${source}-TO-${target} --data-dir ${data_dir} \
--use-xbm --stage ${stage} --mu1 ${mu1} --mu2 ${mu2} --mu3 ${mu3}


