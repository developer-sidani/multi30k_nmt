#!/bin/bash
#SBATCH --time=110:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --job-name=nmt_train
#SBATCH --mem=40GB
#SBATCH --output=/home/asidani/logs/nmt/job_name_%j.log
#SBATCH --error=/home/asidani/logs/nmt/job_name_%j.err


###### 1 Load the module
module load nvidia/cudasdk/11.6
module load intel/python/3

function send_discord {
    python3 /home/asidani/message.py "$@"
}


echo "[SCRIPT]: Checking GPU availability"
which nvidia-smi || echo "nvidia-smi not found"
nvidia-smi || echo "Unable to run nvidia-smi"  

# Select GPU with least memory usage
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{ print NR-1 " " $1 }' | sort -k2 -n | tail -n1 | awk '{ print $1 }')
echo "[SCRIPT]: Selected GPU ID: $CUDA_VISIBLE_DEVICES"

source ~/.bashrc

source activate /home/asidani/.conda/envs/cliptrans

cd /home/asidani/multi30k_nmt

send_discord "[${SLURM_JOB_ID}]: Starting Training"

PYTHON_PATH="/home/asidani/.conda/envs/cliptrans/bin/python3"


TRAIN_PARAMS=(
    --data_dir ./data
    --model_name facebook/mbart-large-50
    --src de
    --tgt fr
    --max_length 128
    --output_dir ./models/mbart50-multi30k
    --learning_rate 1e-5
    --train_batch_size 16
    --eval_batch_size 16
    --weight_decay 0.01
    --epochs 15
    --logging_steps 500
    --save_steps 1813
    --use_cuda_if_available
    --save_base_folder ./eval_outputs/
    --comet_logging
)
$PYTHON_PATH train.py "${TRAIN_PARAMS[@]}"



send_discord "[${SLURM_JOB_ID}]: Train completed"

LOG_FILE="/home/asidani/logs/nmt/job_name_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/nmt/job_name_${SLURM_JOB_ID}.err"

python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE"