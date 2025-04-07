import os
import argparse
import logging
import numpy as np
import random
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler

from src.models import Seq2SeqModel
from src.utils import (
    NMTDataset,
    ReferenceDataset,
    load_dataset,
    load_references,
    get_language_code,
    Evaluator
)
from src.comet import setup_comet_experiment, log_metrics, log_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a Seq2Seq model for Neural Machine Translation')

    # Basic parameters
    parser.add_argument('--src_lang', type=str, required=True, help='Source language code (e.g., en, de)')
    parser.add_argument('--tgt_lang', type=str, required=True, help='Target language code (e.g., en, de)')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory with dataset files')
    parser.add_argument('--max_samples_train', type=int, default=None, help='Max number of training examples')
    parser.add_argument('--max_samples_eval', type=int, default=None, help='Max number of evaluation examples')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='facebook/mbart-large-50',
                        help='Pretrained model name or path')
    parser.add_argument('--max_seq_length', type=int, default=64, help='Maximum sequence length')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps for the scheduler')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear',
                        choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant',
                                 'constant_with_warmup'],
                        help='Type of learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW optimizer')

    # Evaluation parameters
    parser.add_argument('--n_references', type=int, default=1, help='Number of references per input for evaluation')
    parser.add_argument('--eval_steps', type=int, default=1000, help='Evaluate every X steps')
    parser.add_argument('--save_steps', type=int, default=1, help='Save checkpoint every X epochs')

    # Hardware parameters
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true', help='Use pin memory for data loading')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')

    # Output parameters
    parser.add_argument('--save_base_folder', type=str, required=True, help='Base folder to save model checkpoints')
    parser.add_argument('--from_pretrained', type=str, default=None, help='Path to pretrained model checkpoint')

    # Logging parameters
    parser.add_argument('--comet_logging', action='store_true', help='Enable Comet ML logging')
    parser.add_argument('--comet_key', type=str, default=None, help='Comet ML API key')
    parser.add_argument('--comet_workspace', type=str, default=None, help='Comet ML workspace')
    parser.add_argument('--comet_project_name', type=str, default=None, help='Comet ML project name')
    parser.add_argument('--comet_exp', type=str, default=None, help='Comet ML experiment key to continue')

    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()

    # Set device
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    args.device = device

    # Create output directory
    os.makedirs(args.save_base_folder, exist_ok=True)

    # Print arguments
    logger.info("Arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")

    # Set up Comet ML experiment
    experiment_name = f"seq2seq_{args.src_lang}_{args.tgt_lang}"
    experiment = setup_comet_experiment(
        args,
        experiment_name=experiment_name,
        tags=[args.src_lang, args.tgt_lang, "seq2seq"],
        existing_experiment=args.comet_exp
    )

    # Load datasets
    logger.info(f"Loading datasets for {args.src_lang}-{args.tgt_lang}...")

    # Training data
    train_src_path = os.path.join(args.data_dir, f"train.{args.src_lang}")
    train_tgt_path = os.path.join(args.data_dir, f"train.{args.tgt_lang}")

    train_src, train_tgt = load_dataset(
        train_src_path,
        train_tgt_path,
        max_samples=args.max_samples_train
    )

    # Validation data
    val_src_path = os.path.join(args.data_dir, f"test_2016_val.{args.src_lang}")
    val_tgt_path = os.path.join(args.data_dir, f"test_2016_val.{args.tgt_lang}")

    val_src, val_tgt = load_dataset(
        val_src_path,
        val_tgt_path,
        max_samples=args.max_samples_eval
    )

    # Create datasets
    train_dataset = NMTDataset(train_src, train_tgt)
    val_dataset = NMTDataset(val_src, val_tgt)

    # Create reference dataset for evaluation
    val_references = [[tgt] for tgt in val_tgt]  # Single reference per source
    ref_dataset = ReferenceDataset(val_src, val_references)

    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    ref_loader = DataLoader(
        ref_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=ReferenceDataset.collate_fn
    )

    # Initialize model
    logger.info(f"Initializing model: {args.model_name}")

    # Convert language codes
    src_lang_code = get_language_code(args.src_lang)
    tgt_lang_code = get_language_code(args.tgt_lang)

    model = Seq2SeqModel(
        model_name_or_path=args.model_name,
        pretrained_path=args.from_pretrained,
        max_seq_length=args.max_seq_length,
        src_lang=src_lang_code,
        tgt_lang=tgt_lang_code
    )

    model.model.to(device)

    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    num_training_steps = args.epochs * len(train_loader)
    logger.info(f"Total training steps: {num_training_steps}")

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )

    # Initialize evaluator
    evaluator = Evaluator(model, args, experiment)

    # Training loop
    logger.info("Starting training...")
    global_step = 0
    best_val_bleu = 0.0
    train_losses = []

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        model.train_mode()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        epoch_loss = 0.0

        for batch_idx, (src_batch, tgt_batch) in enumerate(progress_bar):
            # Forward pass with loss computation
            _, _, loss = model(src_batch, tgt_batch, device=device)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

            # Log metrics
            if experiment:
                log_metrics(
                    experiment,
                    {"loss": loss.item()},
                    step=global_step,
                    epoch=epoch,
                    phase="train"
                )

            train_losses.append(loss.item())
            global_step += 1

            # Evaluate periodically
            if global_step % args.eval_steps == 0:
                model.eval_mode()
                val_metrics = evaluator.evaluate(
                    ref_loader,
                    phase="validation",
                    epoch=epoch,
                    step=global_step,
                    src_lang=args.src_lang,
                    tgt_lang=args.tgt_lang
                )
                model.train_mode()

                # Save best model
                if val_metrics["bleu"] > best_val_bleu:
                    best_val_bleu = val_metrics["bleu"]
                    logger.info(f"New best BLEU score: {best_val_bleu:.4f}")

                    # Save best model
                    best_model_path = f"{args.save_base_folder}/best_{args.src_lang}_{args.tgt_lang}"
                    os.makedirs(best_model_path, exist_ok=True)
                    model.save_model(best_model_path)

                    if experiment:
                        log_model(experiment, best_model_path, step=global_step, epoch=epoch)

        # End of epoch evaluation
        model.eval_mode()
        val_metrics = evaluator.evaluate(
            ref_loader,
            phase="validation",
            epoch=epoch,
            step=global_step,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang
        )
        model.train_mode()

        # Save model checkpoint at specified intervals
        if (epoch + 1) % args.save_steps == 0:
            checkpoint_path = f"{args.save_base_folder}/epoch_{epoch + 1}_{args.src_lang}_{args.tgt_lang}"
            os.makedirs(checkpoint_path, exist_ok=True)
            model.save_model(checkpoint_path)

            # Save optimizer state
            optimizer_state = {
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_step': global_step
            }
            torch.save(optimizer_state, f"{checkpoint_path}/optimizer.pt")

            # Save training history
            history = {
                'train_losses': train_losses,
                'best_val_bleu': best_val_bleu,
                'args': vars(args)
            }
            with open(f"{checkpoint_path}/history.pickle", 'wb') as f:
                pickle.dump(history, f)

            if experiment:
                log_model(experiment, checkpoint_path, step=global_step, epoch=epoch + 1)

        logger.info(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss / len(train_loader):.4f}")

    # Save final model
    logger.info("Training completed. Saving final model...")
    final_model_path = f"{args.save_base_folder}/final_{args.src_lang}_{args.tgt_lang}"
    os.makedirs(final_model_path, exist_ok=True)
    model.save_model(final_model_path)

    # Final evaluation
    logger.info("Performing final evaluation...")
    final_metrics = evaluator.evaluate(
        ref_loader,
        phase="validation",
        epoch=args.epochs,
        step=global_step,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang
    )

    logger.info(f"Final BLEU score: {final_metrics['bleu']:.4f}")
    logger.info(f"Final METEOR score: {final_metrics['meteor']:.4f}")

    if experiment:
        log_model(experiment, final_model_path, step=global_step, epoch=args.epochs)
        experiment.end()

    logger.info("Training and evaluation completed!")


if __name__ == "__main__":
    main()