# mBART-50 Translation Fine-Tuning Project

An example project demonstrating how to fine-tune a seq2seq model using mBART-50 for translation tasks on multi30k dataset. This project includes training and testing scripts (`train.py` and `test.py`) with command-line arguments and Comet logging integration.

## Directory Structure

```
my_project/
├── data/
│   ├── train.en
│   ├── train.cs
│   ├── test_2016_flickr.en
│   ├── test_2016_flickr.cs
│   ├── test_2016_val.en
│   ├── test_2016_val.cs
├── scripts/
│   ├── train.sbatch
│   ├── test.sbatch
├── src/
│   ├── models/
│   │   └── seq2seq.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── metrics.py
│   │   ├── evaluator.py
│   └── comet.py
├── train.py
└── test.py
```

## Installation

1. Clone this repository.
2. Install required dependencies:
   ```bash
   conda env create -f env.yml
   ```
3. If using Comet, create a `.env` file:
   ```
   COMET_API_KEY=your_comet_api_key
   COMET_PROJECT_NAME=your_project_name
   COMET_WORKSPACE=your_workspace
   ```

## Training

To train the model:
```bash
python train.py \
  --data_dir ./data \
  --model_name facebook/mbart-large-50 \
  --src en \
  --tgt cs \
  --max_length 64 \
  --output_dir ./models/mbart50-multi30k \
  --learning_rate 1e-5 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --epochs 30 \
  --save_steps 1 \
  --use_cuda_if_available \
  --comet_logging
```

## Testing

To evaluate the trained model:
```bash
python test.py \
  --data_dir ./data \
  --model_dir ./models/mbart50-multi30k \
  --test_split test_2016_flickr \
  --src en \
  --tgt cs \
  --max_length 64 \
  --use_cuda_if_available \
  --comet_logging
```

## Command-Line Arguments

### Training Script (train.py)

| Argument                | Description |
|------------------------|-------------|
| `--data_dir`           | Path to dataset directory |
| `--model_name`         | Pretrained model name/path |
| `--src`                | Source language suffix |
| `--tgt`                | Target language suffix |
| `--max_length`         | Max sequence length |
| `--output_dir`         | Directory to save the model |
| `--learning_rate`      | Learning rate |
| `--train_batch_size`   | Training batch size |
| `--eval_batch_size`    | Evaluation batch size |
| `--epochs`             | Number of epochs |
| `--save_steps`         | Steps between saving checkpoints |
| `--use_cuda_if_available` | Enable GPU if available |
| `--comet_logging`      | Enable Comet logging |

### Testing Script (test.py)

| Argument                | Description |
|------------------------|-------------|
| `--data_dir`           | Path to dataset directory |
| `--model_dir`         | Path to trained model directory |
| `--test_split`        | Test split name |
| `--src`                | Source language suffix |
| `--tgt`                | Target language suffix |
| `--max_length`         | Max sequence length |
| `--use_cuda_if_available` | Enable GPU if available |
| `--comet_logging`      | Enable Comet logging |

## Model Saving and Loading

- The trained model is saved in `--output_dir`.
- The test script loads the model from `--model_dir`.
- Comet logs metrics and predictions if enabled.

## Example Execution

To train:
```bash
python train.py --data_dir /path/to/data --model_name facebook/mbart-large-50 ...
```

To test:
```bash
python test.py --data_dir /path/to/data --model_dir ./ckpts/mbart50-multi30k ...
```


