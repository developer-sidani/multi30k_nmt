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

PYTHON_PATH="/home/asidani/.conda/envs/cliptrans/bin/python3"

send_discord "[${SLURM_JOB_ID}]: Starting Seq2Seq Training"

# Base parameters
BASE_PARAMS=(
    --model_name "facebook/mbart-large-50"
    --data_dir "./data"
    --max_seq_length 64
    --batch_size 16
    --eval_batch_size 16
    --learning_rate 3e-5
    --epochs 15
    --warmup_steps 500
    --lr_scheduler_type "linear"
    --weight_decay 0.01
    --eval_steps 1000
    --save_steps 1
    --num_workers 4
    --pin_memory
    --use_cuda
    --comet_logging
)

# Create a directory for saving models
SAVE_BASE="./seq2seq_models"
mkdir -p $SAVE_BASE

# Define language pairs for training
# en->de, de->en, en->fr, fr->en, en->cs, cs->en
LANGUAGE_PAIRS=(
    "en de"
    "de en"
    "en fr"
    "fr en"
    "en cs"
    "cs en"
)

# Train each language pair
for pair in "${LANGUAGE_PAIRS[@]}"; do
    read -r src_lang tgt_lang <<< "$pair"

    send_discord "[${SLURM_JOB_ID}]: Training ${src_lang}->${tgt_lang}"

    SAVE_DIR="${SAVE_BASE}/${src_lang}_${tgt_lang}"
    mkdir -p $SAVE_DIR

    # Set training parameters for this language pair
    TRAIN_PARAMS=(
        --src_lang "$src_lang"
        --tgt_lang "$tgt_lang"
        --save_base_folder "$SAVE_DIR"
    )

    # Start training
    echo "[SCRIPT]: Training ${src_lang}->${tgt_lang}"
    $PYTHON_PATH train.py "${BASE_PARAMS[@]}" "${TRAIN_PARAMS[@]}"

    # Check if training completed successfully
    if [ $? -eq 0 ]; then
        send_discord "[${SLURM_JOB_ID}]: Training ${src_lang}->${tgt_lang} completed successfully"
    else
        send_discord "[${SLURM_JOB_ID}]: Training ${src_lang}->${tgt_lang} failed"
    fi
done

send_discord "[${SLURM_JOB_ID}]: All training jobs completed"

# Send logs if notif.py script exists
LOG_FILE="/home/$USER/logs/seq2seq/train_${SLURM_JOB_ID}.log"
ERR_FILE="/home/$USER/logs/seq2seq/train_${SLURM_JOB_ID}.err"

if [ -f "/home/$USER/notif.py" ]; then
    python3 /home/$USER/notif.py "$LOG_FILE" "$ERR_FILE"
fi