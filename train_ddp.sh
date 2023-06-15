#!/bin/bash
#SBATCH --account=def-wanglab
#SBATCH --time=0-2:59
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100l:4





export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

# srun python main.py --init_method tcp://$MASTER_ADDR:3456 --exp_name clip_a1 --world_size $SLURM_NTASKS  --batch_size 256 --max_epochs 300 --num_workers 4
srun python $1 --init_method tcp://$MASTER_ADDR:3456 --exp_name $2 --world_size $SLURM_NTASKS  --batch_size $3 --max_epochs $4 --num_workers $5


#sbatch train_ddp_slice1.sh main.py clip_a1 256 300 4


