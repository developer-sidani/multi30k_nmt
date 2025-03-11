# test.py
import os
import argparse
import torch
from datasets import Dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from src.utils.data import read_file, get_lang_code
from src.utils.evaluator import Evaluator

def load_test_data(data_dir: str, test_split: str, src_suffix: str, tgt_suffix: str):
    """
    Load test data from the given directory for the specified test split.
    Handles both simple file names (like "test_2016_flickr.en") and more complex ones.
    """
    src_file = os.path.join(data_dir, f"{test_split}.{src_suffix}")
    tgt_file = os.path.join(data_dir, f"{test_split}.{tgt_suffix}")
    
    # Check if files exist
    if not os.path.exists(src_file):
        raise FileNotFoundError(f"Source file not found: {src_file}")
    if not os.path.exists(tgt_file):
        raise FileNotFoundError(f"Target file not found: {tgt_file}")
        
    print(f"Loading test data from: {src_file} and {tgt_file}")
    return {"src": read_file(src_file), "tgt": read_file(tgt_file)}

def main(args):
    # Initialize Comet experiment if enabled
    experiment = None
    if args.comet_logging:
        from src.comet import init_comet
        experiment = init_comet(args, 'test')
        print("Comet experiment initialized.")

    # Load test data
    try:
        test_data = load_test_data(args.data_dir, args.test_split, args.src, args.tgt)
        test_dataset = Dataset.from_dict(test_data)
        print(f"Loaded test dataset with {len(test_dataset)} examples")
        
        # Check for empty datasets
        if len(test_dataset) == 0:
            raise ValueError(f"Test dataset is empty for split: {args.test_split}")
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        if experiment:
            experiment.log_parameter("error", str(e))
        return
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and args.use_cuda_if_available else "cpu"
    args.device = device
    print(f"Using device: {device}")
    
    # Get full language codes
    args.src_lang = get_lang_code(args.src)
    args.tgt_lang = get_lang_code(args.tgt)
    print(f"Source language code: {args.src_lang}, Target language code: {args.tgt_lang}")
    
    # Load the trained model and tokenizer
    try:
        print(f"Loading model from: {args.model_dir}")
        model = MBartForConditionalGeneration.from_pretrained(args.model_dir)
        tokenizer = MBart50TokenizerFast.from_pretrained(args.model_dir, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
        model.to(device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        if experiment:
            experiment.log_parameter("error", str(e))
        return
    
    # Create directory for evaluation outputs if it doesn't exist
    os.makedirs(args.save_base_folder, exist_ok=True)
    
    # Create an Evaluator instance and evaluate the test split
    try:
        evaluator = Evaluator(model=model, tokenizer=tokenizer, args=args, experiment=experiment)
        metrics, df = evaluator.evaluate_split(test_data, split_name=args.test_split, epoch=0, step=0)
        
        # Print summary of results
        print(f"Test Results for {args.test_split} ({args.src} -> {args.tgt}):")
        print(f"BLEU: {metrics['bleu']:.2f}")
        print(f"METEOR: {metrics['meteor']:.2f}")
        print(f"Results saved to: {args.save_base_folder}")
        
        if experiment:
            experiment.log_metric("test_bleu", metrics['bleu'])
            experiment.log_metric("test_meteor", metrics['meteor'])
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        if experiment:
            experiment.log_parameter("error", str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test mBART-50 on specified test split with Comet logging")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the data files")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the trained model")
    parser.add_argument("--test_split", type=str, required=True, help="Test split prefix (e.g., test_2016_flickr, test_2017_mscoco, etc.)")
    parser.add_argument("--src", type=str, required=True, help="Source file suffix (e.g., en, de, fr, cs)")
    parser.add_argument("--tgt", type=str, required=True, help="Target file suffix (e.g., en, de, fr, cs)")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--save_base_folder", type=str, default="./eval_outputs/", help="Folder to save evaluation outputs")
    parser.add_argument("--use_cuda_if_available", action="store_true", default=False, help="Use GPU if available")
    parser.add_argument("--comet_logging", action="store_true", default=False, help="Enable Comet logging")
    
    args = parser.parse_args()
    main(args)