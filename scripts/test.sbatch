#!/bin/bash
#SBATCH --time=23:59:59
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --job-name=multi30k_nmt_test
#SBATCH --mem=32GB
#SBATCH --output=/home/asidani/logs/multi30k_nmt/test_job_%j.log
#SBATCH --error=/home/asidani/logs/multi30k_nmt/test_job_%j.err

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

cd /home/asidani/multi30k_nmt/

echo "[SCRIPT]: Starting inference on Multi30K test sets"
send_discord "[${SLURM_JOB_ID}]: Starting inference on Multi30K test sets"

# Configuration
DATA_DIR="./data/multi30k/data/task1/raw"  # Using raw data
CHECKPOINTS_BASE_DIR="./checkpoints"
OUTPUT_DIR="./eval_outputs"
BATCH_SIZE=16
MAX_LENGTH=60

# Create output directory
mkdir -p $OUTPUT_DIR

# Test splits to evaluate
TEST_SPLITS=("test_2016_flickr" "test_2017_flickr" "test_2017_mscoco" "test_2018_flickr")

# Language pairs
LANGUAGE_PAIRS=("en fr" "fr en")

# Function to run inference for a specific model and test split
run_inference() {
    local src_lang=$1
    local tgt_lang=$2
    local test_split=$3
    local model_dir=$4
    
    echo "Running inference: $src_lang -> $tgt_lang on $test_split"
    
    # Check if model directory exists
    if [ ! -d "$model_dir" ]; then
        echo "Model directory not found: $model_dir"
        return 1
    fi
    
    # Run test script
    python test.py \
        --data_dir "$DATA_DIR" \
        --model_dir "$model_dir" \
        --test_split "$test_split" \
        --src "$src_lang" \
        --tgt "$tgt_lang" \
        --max_length $MAX_LENGTH \
        --batch_size $BATCH_SIZE \
        --save_base_folder "$OUTPUT_DIR" \
        --use_cuda_if_available \
        --comet_logging
    
    if [ $? -eq 0 ]; then
        echo "Successfully completed inference for $src_lang -> $tgt_lang on $test_split"
        return 0
    else
        echo "Error in inference for $src_lang -> $tgt_lang on $test_split"
        return 1
    fi
}

# Function to test all splits for a language pair
test_language_pair() {
    local src_lang=$1
    local tgt_lang=$2
    
    echo ""
    echo "Testing language pair: $src_lang -> $tgt_lang"
    echo "========================================"
    
    # Determine model directory
    if [ "$src_lang" = "en" ]; then
        model_dir="$CHECKPOINTS_BASE_DIR/en_${tgt_lang}_model/en_to_${tgt_lang}/best_model"
    else
        model_dir="$CHECKPOINTS_BASE_DIR/en_${src_lang}_model/${src_lang}_to_en/best_model"
    fi
    
    echo "Using model from: $model_dir"
    
    # Test on all test splits
    for test_split in "${TEST_SPLITS[@]}"; do
        echo ""
        echo "Testing on $test_split..."
        
        # Check if test files exist for this language pair and test split
        src_file="$DATA_DIR/${test_split}.${src_lang}"
        tgt_file="$DATA_DIR/${test_split}.${tgt_lang}"
        
        if [ ! -f "$src_file" ] || [ ! -f "$tgt_file" ]; then
            echo "Test files not found for $test_split ($src_lang -> $tgt_lang)"
            echo "Expected: $src_file and $tgt_file"
            continue
        fi
        
        run_inference "$src_lang" "$tgt_lang" "${test_split}" "$model_dir"
        
        if [ $? -eq 0 ]; then
            send_discord "[${SLURM_JOB_ID}]: ✅ Completed $src_lang -> $tgt_lang on $test_split"
        else
            send_discord "[${SLURM_JOB_ID}]: ❌ Failed $src_lang -> $tgt_lang on $test_split"
        fi
    done
}

# Main inference loop
echo "Starting comprehensive evaluation..."
echo "===================================="

# Test all language pairs
for pair in "${LANGUAGE_PAIRS[@]}"; do
    IFS=' ' read -r src_lang tgt_lang <<< "$pair"
    test_language_pair "$src_lang" "$tgt_lang"
done

echo ""
echo "All inference completed!"
echo "======================="
send_discord "[${SLURM_JOB_ID}]: All inference completed! Results saved to $OUTPUT_DIR"

# Create summary report
echo "Creating summary report..."
SUMMARY_FILE="$OUTPUT_DIR/inference_summary_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "Multi30K NMT Inference Summary"
    echo "=============================="
    echo "Date: $(date)"
    echo "Job ID: ${SLURM_JOB_ID}"
    echo ""
    
    echo "Test Splits Evaluated:"
    for split in "${TEST_SPLITS[@]}"; do
        echo "  - $split"
    done
    
    echo ""
    echo "Language Pairs Tested:"
    for pair in "${LANGUAGE_PAIRS[@]}"; do
        echo "  - $pair"
    done
    
    echo ""
    echo "Results Files:"
    find "$OUTPUT_DIR" -name "*.csv" -o -name "*.txt" | sort
    
} > "$SUMMARY_FILE"

echo "Summary report saved to: $SUMMARY_FILE"

# Send final notification with log files
LOG_FILE="/home/asidani/logs/multi30k_nmt/test_job_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/multi30k_nmt/test_job_${SLURM_JOB_ID}.err"

python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE" 