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

# Define all language pairs to train
LANG_PAIRS=(
    "en2de"
    "de2en"
    "en2fr"
    "fr2en"
    "en2cs"
    "cs2en"
)

send_discord "[${SLURM_JOB_ID}]: Starting training for all language pairs: ${LANG_PAIRS[*]}"

# Train each language pair sequentially
for LANG_PAIR in "${LANG_PAIRS[@]}"; do
    echo "[SCRIPT]: Starting training for language pair: ${LANG_PAIR}"
    send_discord "[${SLURM_JOB_ID}]: Starting training for ${LANG_PAIR}"
    
    case $LANG_PAIR in
        en2de)
            TRAIN_PARAMS=(
                --data_dir ./data
                --model_name facebook/mbart-large-50
                --src en
                --tgt de
                --max_length 64
                --output_dir ./models/mbart50-multi30k-de
                --learning_rate 5e-5
                --train_batch_size 16
                --eval_batch_size 16
                --weight_decay 0.01
                --epochs 15
                --logging_steps 500
                --save_steps 1
                --use_cuda_if_available
                --save_base_folder ./eval_outputs/en2de/
                --comet_logging
            )
            ;;
        de2en)
            TRAIN_PARAMS=(
                --data_dir ./data
                --model_name facebook/mbart-large-50
                --src de
                --tgt en
                --max_length 64
                --output_dir ./models/mbart50-multi30k-de2en
                --learning_rate 5e-5
                --train_batch_size 16
                --eval_batch_size 16
                --weight_decay 0.01
                --epochs 15
                --logging_steps 500
                --save_steps 1
                --use_cuda_if_available
                --save_base_folder ./eval_outputs/de2en/
                --comet_logging
            )
            ;;
        en2fr)
            TRAIN_PARAMS=(
                --data_dir ./data
                --model_name facebook/mbart-large-50
                --src en
                --tgt fr
                --max_length 64
                --output_dir ./models/mbart50-multi30k-fr
                --learning_rate 5e-5
                --train_batch_size 16
                --eval_batch_size 16
                --weight_decay 0.01
                --epochs 15
                --logging_steps 500
                --save_steps 1
                --use_cuda_if_available
                --save_base_folder ./eval_outputs/en2fr/
                --comet_logging
            )
            ;;
        fr2en)
            TRAIN_PARAMS=(
                --data_dir ./data
                --model_name facebook/mbart-large-50
                --src fr
                --tgt en
                --max_length 64
                --output_dir ./models/mbart50-multi30k-fr2en
                --learning_rate 5e-5
                --train_batch_size 16
                --eval_batch_size 16
                --weight_decay 0.01
                --epochs 15
                --logging_steps 500
                --save_steps 1
                --use_cuda_if_available
                --save_base_folder ./eval_outputs/fr2en/
                --comet_logging
            )
            ;;
        en2cs)
            TRAIN_PARAMS=(
                --data_dir ./data
                --model_name facebook/mbart-large-50
                --src en
                --tgt cs
                --max_length 64
                --output_dir ./models/mbart50-multi30k-cs
                --learning_rate 5e-5
                --train_batch_size 16
                --eval_batch_size 16
                --weight_decay 0.01
                --epochs 15
                --logging_steps 500
                --save_steps 1
                --use_cuda_if_available
                --save_base_folder ./eval_outputs/en2cs/
                --comet_logging
            )
            ;;
        cs2en)
            TRAIN_PARAMS=(
                --data_dir ./data
                --model_name facebook/mbart-large-50
                --src cs
                --tgt en
                --max_length 64
                --output_dir ./models/mbart50-multi30k-cs2en
                --learning_rate 5e-5
                --train_batch_size 16
                --eval_batch_size 16
                --weight_decay 0.01
                --epochs 15
                --logging_steps 500
                --save_steps 1
                --use_cuda_if_available
                --save_base_folder ./eval_outputs/cs2en/
                --comet_logging
            )
            ;;
        *)
            echo "[SCRIPT]: Invalid language pair: ${LANG_PAIR}"
            send_discord "[${SLURM_JOB_ID}]: ERROR - Invalid language pair: ${LANG_PAIR}"
            continue
            ;;
    esac

    echo "[SCRIPT]: Running training for ${LANG_PAIR} with parameters:"
    echo "${TRAIN_PARAMS[@]}"

    # Run the training for this language pair
    $PYTHON_PATH train.py "${TRAIN_PARAMS[@]}"
    
    TRAIN_EXIT_CODE=$?
    
    if [ $TRAIN_EXIT_CODE -ne 0 ]; then
        send_discord "[${SLURM_JOB_ID}]: Training for ${LANG_PAIR} failed with exit code ${TRAIN_EXIT_CODE}"
        echo "[SCRIPT]: Training for ${LANG_PAIR} failed with exit code ${TRAIN_EXIT_CODE}"
    else
        send_discord "[${SLURM_JOB_ID}]: Training for ${LANG_PAIR} completed successfully"
        echo "[SCRIPT]: Training for ${LANG_PAIR} completed successfully"
    fi
    
    # Clean up GPU memory before starting the next training
done

send_discord "[${SLURM_JOB_ID}]: All training jobs completed"

LOG_FILE="/home/asidani/logs/nmt/job_name_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/nmt/job_name_${SLURM_JOB_ID}.err"

python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE"