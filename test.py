import os
import argparse
import logging
import numpy as np
import random

import torch
from torch.utils.data import DataLoader

from src.models import Seq2SeqModel
from src.utils import (
    ReferenceDataset,
    load_dataset,
    get_language_code,
    Evaluator
)
from src.comet import setup_comet_experiment

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
    parser = argparse.ArgumentParser(description='Test a Seq2Seq model for Neural Machine Translation')

    # Basic parameters
    parser.add_argument('--src_lang', type=str, required=True, help='Source language code (e.g., en, de)')
    parser.add_argument('--tgt_lang', type=str, required=True, help='Target language code (e.g., en, de)')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory with dataset files')
    parser.add_argument('--test_set', type=str, default='flickr2016',
                        choices=['flickr2016', 'flickr2017', 'flickr2018', 'mscoco2017'],
                        help='Test set to evaluate on')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='facebook/mbart-large-50', help='Base pretrained model name')
    parser.add_argument('--max_seq_length', type=int, default=64, help='Maximum sequence length')

    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Evaluation batch size')
    parser.add_argument('--n_references', type=int, default=1, help='Number of references per input')
    parser.add_argument('--ref_path', type=str, help='Path to reference translations')

    # Hardware parameters
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true', help='Use pin memory for data loading')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')

    # Output parameters
    parser.add_argument('--save_base_folder', type=str, required=True, help='Base folder to save test results')

    # Logging parameters
    parser.add_argument('--comet_logging', action='store_true', help='Enable Comet ML logging')
    parser.add_argument('--comet_key', type=str, default=None, help='Comet ML API key')
    parser.add_argument('--comet_workspace', type=str, default=None, help='Comet ML workspace')
    parser.add_argument('--comet_project_name', type=str, default=None, help='Comet ML project name')

    return parser.parse_args()


def main():
    """Main testing function."""
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
    experiment_name = f"seq2seq_{args.src_lang}_{args.tgt_lang}_{args.test_set}"
    experiment = setup_comet_experiment(
        args,
        experiment_name=experiment_name,
        tags=[args.src_lang, args.tgt_lang, args.test_set, "seq2seq", "test"]
    )

    # Load test dataset
    logger.info(f"Loading test dataset: {args.test_set}")

    # Test data paths
    test_src_path = os.path.join(args.data_dir, f"test_{args.test_set}.{args.src_lang}")
    test_tgt_path = os.path.join(args.data_dir, f"test_{args.test_set}.{args.tgt_lang}")

    # Load source and target sentences
    test_src, test_tgt = load_dataset(test_src_path, test_tgt_path)

    # Create reference dataset
    test_references = [[tgt] for tgt in test_tgt]  # Single reference per source

    # If reference path is provided, load multiple references
    if args.ref_path and args.n_references > 1:
        # TODO: Implement loading multiple references if needed
        logger.info(f"Using {args.n_references} references for evaluation")

    ref_dataset = ReferenceDataset(test_src, test_references)

    logger.info(f"Test examples: {len(ref_dataset)}")

    # Create data loader
    ref_loader = DataLoader(
        ref_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=ReferenceDataset.collate_fn
    )

    # Initialize model
    logger.info(f"Loading model from: {args.model_path}")

    # Convert language codes
    src_lang_code = get_language_code(args.src_lang)
    tgt_lang_code = get_language_code(args.tgt_lang)

    model = Seq2SeqModel(
        model_name_or_path=args.model_name,
        pretrained_path=args.model_path,
        max_seq_length=args.max_seq_length,
        src_lang=src_lang_code,
        tgt_lang=tgt_lang_code
    )

    model.model.to(device)
    model.eval_mode()

    # Initialize evaluator
    evaluator = Evaluator(model, args, experiment)

    # Evaluate on test set
    logger.info(f"Evaluating {args.src_lang}->{args.tgt_lang} on {args.test_set}...")
    test_metrics = evaluator.evaluate(
        ref_loader,
        phase="test",
        epoch=0,
        step=0,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        test_set=args.test_set
    )

    # Print results
    logger.info(f"Test results for {args.src_lang}->{args.tgt_lang} on {args.test_set}:")
    logger.info(f"  BLEU score: {test_metrics['bleu']:.4f}")
    logger.info(f"  METEOR score: {test_metrics['meteor']:.4f}")

    if experiment:
        experiment.end()

    logger.info("Testing completed!")


if __name__ == "__main__":
    main()