# test.py
import os
import argparse
import torch
from datasets import Dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from src.utils.data import read_file, get_lang_code
from src.utils.evaluator import Evaluator

def load_test_data(data_dir: str, test_split: str, src_suffix: str, tgt_suffix: str):
    src_file = os.path.join(data_dir, f"{test_split}.{src_suffix}")
    tgt_file = os.path.join(data_dir, f"{test_split}.{tgt_suffix}")
    return {"src": read_file(src_file), "tgt": read_file(tgt_file)}

def main(args):
    # Initialize Comet experiment if enabled.
    experiment = None
    if args.comet_logging:
        from src.comet import init_comet
        experiment = init_comet(args,'test')
        print("Comet experiment initialized.")

    # Load test data.
    test_data = load_test_data(args.data_dir, args.test_split, args.src, args.tgt)
    test_dataset = Dataset.from_dict(test_data)
    
    # Load the trained model and tokenizer.
    args.src_lang = get_lang_code(args.src)
    args.tgt_lang = get_lang_code(args.tgt)
    model = MBartForConditionalGeneration.from_pretrained(args.model_dir)
    tokenizer = MBart50TokenizerFast.from_pretrained(args.model_dir, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    
    device = "cuda" if torch.cuda.is_available() and args.use_cuda_if_available else "cpu"
    model.to(device)
    
    # Create an Evaluator instance, passing the experiment if comet_logging is enabled.
    evaluator = Evaluator(model=model, tokenizer=tokenizer, args=args, experiment=experiment)
    # Evaluate the test split. Here, we use epoch=0 and step=0 for simplicity.
    evaluator.evaluate_split(test_data, split_name=args.test_split, epoch=0, step=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test mBART-50 on specified test split with Comet logging")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the data files")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the trained model")
    parser.add_argument("--test_split", type=str, required=True, help="Test split prefix (e.g., test_2016_flickr, test_2017_mscoco, etc.)")
    parser.add_argument("--src", type=str, required=True, help="Source file suffix (e.g., de, fr, cs)")
    parser.add_argument("--tgt", type=str, required=True, help="Target file suffix (e.g., de, fr, cs)")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--save_base_folder", type=str, default="./eval_outputs/", help="Folder to save evaluation outputs")
    parser.add_argument("--use_cuda_if_available", action="store_true", default=False, help="Use GPU if available")
    parser.add_argument("--comet_logging", action="store_true", default=False, help="Enable Comet logging")
    
    args = parser.parse_args()
    main(args)
