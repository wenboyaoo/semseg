#!/bin/sh

## uncomment for slurm
##SBATCH -p gpu
##SBATCH --gres=gpu:1
##SBATCH -c 10

export PYTHONPATH=./
PYTHON=python

dataset=$1
exp_name=$2
exp_dir=exp/
result_dir=${exp_dir}/result
config=config.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${result_dir}
cp tool/test.sh tool/test.py ${config} ${exp_dir}

export PYTHONPATH=./
$PYTHON -u ${exp_dir}/test.py \
  --config=${config} \
  2>&1 | tee ${result_dir}/test-$now.log
