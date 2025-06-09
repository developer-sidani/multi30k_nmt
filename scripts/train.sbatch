#!/bin/bash
#SBATCH --time=23:59:59
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=multi30k_nmt
#SBATCH --mem=64GB
#SBATCH --output=/home/asidani/logs/multi30k_nmt/job_name_%j.log
#SBATCH --error=/home/asidani/logs/multi30k_nmt/job_name_%j.err


function send_discord {
    python3 /home/asidani/message.py "$@"
}

source ~/.bashrc

source ~/miniconda3/bin/activate
conda activate cyclegan

echo "[SCRIPT]: Checking GPU availability"
which nvidia-smi || echo "nvidia-smi not found"
nvidia-smi || echo "Unable to run nvidia-smi"  

# Select GPU with least memory usage
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{ print NR-1 " " $1 }' | sort -k2 -n | tail -n1 | awk '{ print $1 }')
echo "[SCRIPT]: Selected GPU ID: $CUDA_VISIBLE_DEVICES"


# 3 Run the python script

echo "[SCRIPT]: Starting the run"
send_discord "[${SLURM_JOB_ID}]: Starting the run"



cd /home/asidani/multi30k_nmt/

# Configuration
DATA_DIR="./data/multi30k/data/task1/raw"
OUTPUT_BASE_DIR="./checkpoints"
EPOCHS=15
BATCH_SIZE=16
LEARNING_RATE=3e-5

# Create output directory
mkdir -p $OUTPUT_BASE_DIR

# Language pairs to train
LANGUAGES=("fr")

# Function to train bidirectional model
train_language() {
    local lang=$1
    echo "Training English <-> $lang models..."
    
    # Create output directory for this language pair
    local output_dir="$OUTPUT_BASE_DIR/en_${lang}_model"
    mkdir -p $output_dir
    
    # Train EN -> Target Language
    echo "Training EN -> $lang..."
    python train.py \
        --src en \
        --tgt $lang \
        --data_dir $DATA_DIR \
        --output_dir "${output_dir}/en_to_${lang}" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE
    
    # Train Target Language -> EN
    echo "Training $lang -> EN..."
    python train.py \
        --src $lang \
        --tgt en \
        --data_dir $DATA_DIR \
        --output_dir "${output_dir}/${lang}_to_en" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE
}

# Main training loop
send_discord "[${SLURM_JOB_ID}]: Starting training for all language pairs..."
echo "Starting training for all language pairs..."
echo "============================================"

for lang in "${LANGUAGES[@]}"; do
    echo ""
    echo "Processing language: $lang"
    send_discord "[${SLURM_JOB_ID}]: Processing language: $lang"
    echo "=========================="
    train_language $lang
    echo "Completed training for $lang"
    send_discord "[${SLURM_JOB_ID}]: Completed training for $lang"
    echo ""
done
send_discord "[${SLURM_JOB_ID}]: All training completed!"
echo "All training completed!"
echo "======================"

# Print checkpoint locations for best models
echo ""
echo "Best model checkpoints (update these paths after training):"
echo "=========================================================="
echo "# Define these variables with the actual best epoch numbers after training"
echo ""
echo "# English -> Czech"
echo "cs_en_to_cs_checkpoint='./checkpoints/en_cs_model/en_to_cs/best_model'"
echo "cs_cs_to_en_checkpoint='./checkpoints/en_cs_model/cs_to_en/best_model'"
echo ""
echo "# English -> German" 
echo "de_en_to_de_checkpoint='./checkpoints/en_de_model/en_to_de/best_model'"
echo "de_de_to_en_checkpoint='./checkpoints/en_de_model/de_to_en/best_model'"
echo ""
echo "# English -> French"
echo "fr_en_to_fr_checkpoint='./checkpoints/en_fr_model/en_to_fr/best_model'"
echo "fr_fr_to_en_checkpoint='./checkpoints/en_fr_model/fr_to_en/best_model'"
echo ""
echo "Training logs and metrics are available in Comet ML dashboard."

# Define the full path for log and error files
LOG_FILE="/home/asidani/logs/multi30k_nmt/job_name_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/multi30k_nmt/job_name_${SLURM_JOB_ID}.err"


python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE"