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

send_discord "[${SLURM_JOB_ID}]: Starting Testing"

PYTHON_PATH="/home/asidani/.conda/envs/cliptrans/bin/python3"

LANG_PAIRS=(
    "en2de"
    "de2en"
    "en2fr"
    "fr2en"
    "en2cs"
    "cs2en"
)

send_discord "[${SLURM_JOB_ID}]: Starting testing for all language pairs: ${LANG_PAIRS[*]}"

# List of test splits to evaluate
TEST_SPLITS=(
    "test_2016_flickr"
    "test_2016_val"
    "test_2017_flickr"
    "test_2017_mscoco"
    "test_2018_flickr"
)

# Test each language pair sequentially
for LANG_PAIR in "${LANG_PAIRS[@]}"; do
    echo "[SCRIPT]: Starting testing for language pair: ${LANG_PAIR}"
    send_discord "[${SLURM_JOB_ID}]: Starting testing for ${LANG_PAIR}"

    case $LANG_PAIR in
        en2de)
            SRC="en"
            TGT="de"
            MODEL_DIR="./models/mbart50-multi30k-de"
            SAVE_FOLDER="./eval_outputs/en2de/"
            ;;
        de2en)
            SRC="de"
            TGT="en"
            MODEL_DIR="./models/mbart50-multi30k-de2en"
            SAVE_FOLDER="./eval_outputs/de2en/"
            ;;
        en2fr)
            SRC="en"
            TGT="fr"
            MODEL_DIR="./models/mbart50-multi30k-fr"
            SAVE_FOLDER="./eval_outputs/en2fr/"
            ;;
        fr2en)
            SRC="fr"
            TGT="en"
            MODEL_DIR="./models/mbart50-multi30k-fr2en"
            SAVE_FOLDER="./eval_outputs/fr2en/"
            ;;
        en2cs)
            SRC="en"
            TGT="cs"
            MODEL_DIR="./models/mbart50-multi30k-cs"
            SAVE_FOLDER="./eval_outputs/en2cs/"
            ;;
        cs2en)
            SRC="cs"
            TGT="en"
            MODEL_DIR="./models/mbart50-multi30k-cs2en"
            SAVE_FOLDER="./eval_outputs/cs2en/"
            ;;
    esac

    for TEST_SPLIT in "${TEST_SPLITS[@]}"; do
        if [ -f "./data/${TEST_SPLIT}.${SRC}" ] && [ -f "./data/${TEST_SPLIT}.${TGT}" ]; then
            echo "[SCRIPT]: Testing on ${TEST_SPLIT} (${SRC}->${TGT})"

            TEST_PARAMS=(
                --data_dir ./data
                --model_dir "${MODEL_DIR}"
                --test_split "${TEST_SPLIT}"
                --src "${SRC}"
                --tgt "${TGT}"
                --max_length 64
                --use_cuda_if_available
                --save_base_folder "${SAVE_FOLDER}${TEST_SPLIT}/"
                --comet_logging
            )

            $PYTHON_PATH test.py "${TEST_PARAMS[@]}"
            TEST_EXIT_CODE=$?

            if [ $TEST_EXIT_CODE -ne 0 ]; then
                send_discord "[${SLURM_JOB_ID}]: Error testing on ${LANG_PAIR} ${TEST_SPLIT}. Exit code: ${TEST_EXIT_CODE}"
            else
                send_discord "[${SLURM_JOB_ID}]: Successfully tested ${LANG_PAIR} on ${TEST_SPLIT}"
            fi
        else
            echo "[SCRIPT]: Skipping ${TEST_SPLIT} for ${LANG_PAIR} because files don't exist"
        fi
    done

    # Send a summary message after all test splits are completed for this language pair
    send_discord "[${SLURM_JOB_ID}]: Completed testing for ${LANG_PAIR}"
done

send_discord "[${SLURM_JOB_ID}]: All testing completed for all language pairs"

LOG_FILE="/home/asidani/logs/nmt/test_job_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/nmt/test_job_${SLURM_JOB_ID}.err"

python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE"
