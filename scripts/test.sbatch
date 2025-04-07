#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --job-name=nmt_test
#SBATCH --mem=40GB
#SBATCH --output=/home/asidani/logs/nmt/test_job_%j.log
#SBATCH --error=/home/asidani/logs/nmt/test_job_%j.err

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

send_discord "[${SLURM_JOB_ID}]: Starting Seq2Seq Testing"

PYTHON_PATH="/home/asidani/.conda/envs/cliptrans/bin/python3"

# Base parameters
BASE_PARAMS=(
    --model_name "facebook/mbart-large-50"
    --data_dir "./data"
    --max_seq_length 64
    --batch_size 32
    --num_workers 4
    --pin_memory
    --use_cuda
    --comet_logging
)

# Create a directory for saving test results
RESULTS_BASE="./seq2seq_results"
mkdir -p $RESULTS_BASE

# Define test sets
TEST_SETS=("flickr2016" "flickr2017" "flickr2018" "mscoco2017")

# Define all language pairs to test
# Flickr2016: en<->de, en<->fr, en<->cs
# Flickr2017, Flickr2018, MSCOCO2017: en<->de, en<->fr

for test_set in "${TEST_SETS[@]}"; do
    echo "[SCRIPT]: Testing on $test_set"

    # Define applicable language pairs for this test set
    LANGUAGE_PAIRS=()

    if [ "$test_set" == "flickr2016" ]; then
        LANGUAGE_PAIRS=(
            "en de"
            "de en"
            "en fr"
            "fr en"
            "en cs"
            "cs en"
        )
    else
        LANGUAGE_PAIRS=(
            "en de"
            "de en"
            "en fr"
            "fr en"
        )
    fi

    # Skip test set if language files don't exist
    if [ ! -f "./data/test_${test_set}.en" ]; then
        echo "[SCRIPT]: Skipping $test_set, files not found"
        continue
    fi

    # Test each applicable language pair
    for pair in "${LANGUAGE_PAIRS[@]}"; do
        read -r src_lang tgt_lang <<< "$pair"

        # Skip if source or target file doesn't exist
        if [ ! -f "./data/test_${test_set}.${src_lang}" ] || [ ! -f "./data/test_${test_set}.${tgt_lang}" ]; then
            echo "[SCRIPT]: Skipping ${src_lang}->${tgt_lang} for $test_set, files not found"
            continue
        fi

        send_discord "[${SLURM_JOB_ID}]: Testing ${src_lang}->${tgt_lang} on $test_set"

        # Path to trained model
        MODEL_PATH="./seq2seq_models/${src_lang}_${tgt_lang}/best_${src_lang}_${tgt_lang}"
        RESULTS_DIR="${RESULTS_BASE}/${test_set}/${src_lang}_${tgt_lang}"
        mkdir -p $RESULTS_DIR

        # Set test parameters for this language pair
        TEST_PARAMS=(
            --src_lang "$src_lang"
            --tgt_lang "$tgt_lang"
            --test_set "$test_set"
            --model_path "$MODEL_PATH"
            --save_base_folder "$RESULTS_DIR"
        )

        # Start testing
        echo "[SCRIPT]: Testing ${src_lang}->${tgt_lang} on $test_set"
        $PYTHON_PATH test.py "${BASE_PARAMS[@]}" "${TEST_PARAMS[@]}"

        # Check if testing completed successfully
        if [ $? -eq 0 ]; then
            send_discord "[${SLURM_JOB_ID}]: Testing ${src_lang}->${tgt_lang} on $test_set completed successfully"
        else
            send_discord "[${SLURM_JOB_ID}]: Testing ${src_lang}->${tgt_lang} on $test_set failed"
        fi
    done
done

send_discord "[${SLURM_JOB_ID}]: All testing jobs completed"

# Send logs if notif.py script exists
LOG_FILE="/home/$USER/logs/seq2seq/test_${SLURM_JOB_ID}.log"
ERR_FILE="/home/$USER/logs/seq2seq/test_${SLURM_JOB_ID}.err"

if [ -f "/home/$USER/notif.py" ]; then
    python3 /home/$USER/notif.py "$LOG_FILE" "$ERR_FILE"
fi