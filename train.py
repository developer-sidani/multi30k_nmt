# train.py
import os
import argparse
import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from src.models.seq2seq import load_seq2seq_model
from src.utils.data import load_local_dataset, preprocess_function, get_lang_code
from src.utils.metrics import compute_metrics
from src.utils.evaluator import Evaluator

def main(args):
    # Set device.
    args.device = "cuda" if torch.cuda.is_available() and args.use_cuda_if_available else "cpu"

    # Disable tokenizer parallelism to avoid deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Initialize Comet experiment if flag enabled.
    experiment = None
    if args.comet_logging:
        from src.comet import init_comet
        experiment = init_comet(args, 'train')
        print("Comet experiment initialized.")
    
    # Load dataset from data_dir using provided src and tgt file suffixes.
    datasets = load_local_dataset(args.data_dir, src_suffix=args.src, tgt_suffix=args.tgt)
    
    # Load model and tokenizer using model_name
    # Convert src and tgt to full language codes for later use
    args.src_lang = get_lang_code(args.src)
    args.tgt_lang = get_lang_code(args.tgt)
    
    model, tokenizer = load_seq2seq_model(args.model_name, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    model.to(args.device)
    
    # Preprocess datasets.
    tokenized_datasets = datasets.map(
        lambda ex: preprocess_function(ex, tokenizer, max_length=args.max_length),
        batched=True,
        remove_columns=datasets["train"].column_names
    )
    comet_callback = None
    if args.comet_logging and experiment is not None:
        from transformers.integrations import CometCallback
        comet_callback = CometCallback()
        # Manually set its experiment attribute to your pre‑existing experiment.
        comet_callback.experiment = experiment
    # Set training arguments. We use evaluation_strategy="epoch" so evaluation (on test_2016_val) is done every epoch.
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps if args.save_steps else 1000,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_steps=500,
        # load_best_model_at_end=True,
        metric_for_best_model="eval_bleu",
        greater_is_better=True,
        fp16=torch.cuda.is_available() and args.use_cuda_if_available,
        report_to=[]  # we assume Comet logging is handled separately
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
        callbacks=[comet_callback] if comet_callback is not None else None
    )
    
    # Start training.
    trainer.train()
    
    # After training, run evaluation on the validation set using our Evaluator.
    evaluator = Evaluator(model=model, tokenizer=tokenizer, args=args, experiment=experiment)
    evaluator.evaluate_split(datasets["validation"], split_name="validation", epoch=args.epochs, step=training_args.logging_steps)
    
    trainer.save_model(args.output_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune mBART-50 for translation (seq2seq)")    
    # Data paths
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the data files")
    parser.add_argument("--src", type=str, required=True, help="Source file suffix (e.g., de, fr, cs)")
    parser.add_argument("--tgt", type=str, required=True, help="Target file suffix (e.g., de, fr, cs)")
    
    
    # Training arguments
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument('--model_name', type=str, dest="model_name", help='The name of the model (e.g., "facebook/bart-base").')
    parser.add_argument("--output_dir", type=str, default="./mbart50-multi30k", help="Directory to save the model")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size per device")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--logging_steps", type=int, default=500, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Steps between saving checkpoints")  # Ensure a positive integer default
    parser.add_argument("--use_cuda_if_available", action="store_true", dest="use_cuda_if_available", default=False, help="Use GPU if available")
    
    # Evaluation and saving options
    parser.add_argument("--save_base_folder", type=str, dest="save_base_folder", default="./eval_outputs/", help="Folder to store evaluation outputs")
    parser.add_argument("--from_pretrained", type=str, dest="from_pretrained", default=None, help="Path to load pretrained checkpoint")
    
    # Comet logging
    parser.add_argument("--comet_logging", action="store_true", dest="comet_logging", default=False, help="Enable Comet logging")
    parser.add_argument("--comet_exp", type=str, dest="comet_exp", default=None, help="Comet experiment key")
    
    args = parser.parse_args()
    main(args)
