#!/bin/bash 
#SBATCH --partition=gpuq 
#SBATCH --qos=gpu 
#SBATCH --job-name=gpu_basics 
#SBATCH --output=gpu_basics.%j.out 
#SBATCH --error=gpu_basics.%j.out 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1g.10gb:1 
#SBATCH --mem=16GB 
#SBATCH --export=ALL 
#SBATCH --time=0-01:00:00 
#
set echo 
umask 0022 
# to see ID and state of GPUs assigned
nvidia-smi 

## Load the necessary modules

module load gnu10

module load python

python -m venv mlp



source activate ~/mlp/bin/activate

cd /home/bpanigr/projects/DeepLabV3Plus-Pytorch

pip install -r requirements.txt

## Execute your script
python main.py --model deeplabv3_mobilenet --dataset cityscapes --vis_port 28333 --gpu_id 0  --lr 0.1  --crop_size 768 --batch_size 4 --output_stride 16 --total_itrs 10 --data_root ./datasets/data/cityscapes --ckpt ./checkpoints/best_deeplabv3_mobilenet_voc_os16.pth