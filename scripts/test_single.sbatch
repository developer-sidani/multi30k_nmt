#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=multi30k_nmt_single_test
#SBATCH --mem=16GB
#SBATCH --output=/home/asidani/logs/multi30k_nmt/single_test_%j.log
#SBATCH --error=/home/asidani/logs/multi30k_nmt/single_test_%j.err

# Parameters (modify these before running)
SRC_LANG=${1:-"en"}           # Source language
TGT_LANG=${2:-"de"}           # Target language  
TEST_SPLIT=${3:-"test_2016_flickr"}  # Test split
MODEL_DIR=${4:-"./checkpoints/en_de_model/en_to_de/best_model"}  # Model directory

function send_discord {
    python3 /home/asidani/message.py "$@"
}

source ~/.bashrc
source ~/miniconda3/bin/activate
conda activate cyclegan

echo "[SCRIPT]: Testing single language pair: $SRC_LANG -> $TGT_LANG"
echo "[SCRIPT]: Test split: $TEST_SPLIT"
echo "[SCRIPT]: Model directory: $MODEL_DIR"

# Select GPU with least memory usage
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{ print NR-1 " " $1 }' | sort -k2 -n | tail -n1 | awk '{ print $1 }')
echo "[SCRIPT]: Selected GPU ID: $CUDA_VISIBLE_DEVICES"

cd /home/asidani/multi30k_nmt/

# Configuration
DATA_DIR="./data/multi30k/data/task1/raw"
OUTPUT_DIR="./eval_outputs"
BATCH_SIZE=16
MAX_LENGTH=128

# Create output directory
mkdir -p $OUTPUT_DIR

send_discord "[${SLURM_JOB_ID}]: Starting test: $SRC_LANG -> $TGT_LANG on $TEST_SPLIT"

# Run test
python test.py \
    --data_dir "$DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --test_split "$TEST_SPLIT" \
    --src "$SRC_LANG" \
    --tgt "$TGT_LANG" \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --save_base_folder "$OUTPUT_DIR" \
    --use_cuda_if_available \
    --comet_logging

if [ $? -eq 0 ]; then
    echo "Test completed successfully!"
    send_discord "[${SLURM_JOB_ID}]: ✅ Test completed: $SRC_LANG -> $TGT_LANG on $TEST_SPLIT"
else
    echo "Test failed!"
    send_discord "[${SLURM_JOB_ID}]: ❌ Test failed: $SRC_LANG -> $TGT_LANG on $TEST_SPLIT"
fi

# Send notification with log files
LOG_FILE="/home/asidani/logs/multi30k_nmt/single_test_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/multi30k_nmt/single_test_${SLURM_JOB_ID}.err"

python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE" 