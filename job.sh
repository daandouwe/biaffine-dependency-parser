#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=1:30:00

module load eb
# module load Python/3.6.3-foss-2017b
module load Python
module load PyTorch
module load CUDA
module load cuDNN

python3 train.py --data ../stanford-ptb
