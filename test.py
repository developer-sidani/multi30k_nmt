# test.py
import os
import argparse
import torch
import pandas as pd
import numpy as np
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from datasets import load_metric
import comet_ml
from tqdm import tqdm
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_file(file_path):
    """Read lines from a text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def get_lang_code(lang_suffix):
    """Convert language suffix to mBART language code"""
    lang_codes = {
        'en': 'en_XX',
        'de': 'de_DE', 
        'fr': 'fr_XX',
        'cs': 'cs_CZ'
    }
    return lang_codes.get(lang_suffix, lang_suffix)

def load_test_data(data_dir, test_split, src_suffix, tgt_suffix):
    """Load test data from the given directory for the specified test split"""
    src_file = os.path.join(data_dir, f"{test_split}.{src_suffix}")
    tgt_file = os.path.join(data_dir, f"{test_split}.{tgt_suffix}")
    
    # Check if files exist
    if not os.path.exists(src_file):
        raise FileNotFoundError(f"Source file not found: {src_file}")
    if not os.path.exists(tgt_file):
        raise FileNotFoundError(f"Target file not found: {tgt_file}")
        
    logger.info(f"Loading test data from: {src_file} and {tgt_file}")
    
    # Read data as lists of strings
    src_data = read_file(src_file)
    tgt_data = read_file(tgt_file)
    
    logger.info(f"Loaded {len(src_data)} source sentences and {len(tgt_data)} target sentences")
    
    return {"src": src_data, "tgt": tgt_data}

def generate_translations(model, tokenizer, texts, src_lang, tgt_lang, device, max_length=512, batch_size=8):
    """Generate translations for a list of texts"""
    model.eval()
    translations = []
    
    # Set language codes
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    
    logger.info(f"Starting translation of {len(texts)} texts with batch size {batch_size}")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating translations"):
            batch_texts = texts[i:i+batch_size]
            
            logger.debug(f"Processing batch {i//batch_size + 1}, texts {i} to {i+len(batch_texts)-1}")
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                max_length=max_length, 
                truncation=True, 
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate translations
            try:
                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True
                )
                
                # Decode translations
                batch_translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                translations.extend(batch_translations)
                
                logger.debug(f"Successfully processed batch {i//batch_size + 1}, generated {len(batch_translations)} translations")
                
            except Exception as e:
                logger.error(f"Error in batch {i//batch_size + 1}: {str(e)}")
                # Add empty translations for this batch to maintain alignment
                batch_translations = ["[ERROR]"] * len(batch_texts)
                translations.extend(batch_translations)
    
    logger.info(f"Translation completed. Generated {len(translations)} translations")
    return translations

def compute_metrics(predictions, references):
    """Compute BLEU and METEOR scores"""
    # Use sacrebleu for consistency with CycleGAN implementation (returns 0-100 range)
    import evaluate
    bleu_metric = evaluate.load('sacrebleu')
    meteor_metric = evaluate.load('meteor')
    
    # Debug information
    logger.info(f"Computing metrics for {len(predictions)} predictions and {len(references)} references")
    logger.info(f"First prediction: {predictions[0] if predictions else 'None'}")
    logger.info(f"First reference: {references[0] if references else 'None'}")
    
    # Follow CycleGAN approach: compute BLEU sentence by sentence, then average
    # This matches their compute_metric function exactly
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        # For single reference: [ref] creates a list with one reference
        res = bleu_metric.compute(predictions=[pred], references=[[ref]])
        bleu_scores.append(res['score'])
    
    # Compute corpus-level METEOR
    meteor_score = meteor_metric.compute(predictions=predictions, references=references)
    
    return {
        'bleu': sum(bleu_scores) / len(bleu_scores),  # Average of sentence-level BLEU scores
        'meteor': meteor_score['meteor'] * 100  # Convert METEOR to 0-100 range for consistency
    }

def main(args):
    # Load environment variables from .env file if it exists
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize Comet experiment if enabled
    experiment = None
    if args.comet_logging:
        comet_api_key = os.getenv('COMET_API_KEY')
        comet_project = os.getenv('COMET_PROJECT_NAME') 
        comet_workspace = os.getenv('COMET_WORKSPACE')
        
        if comet_api_key:
            experiment = comet_ml.Experiment(
                api_key=comet_api_key, 
                project_name=comet_project, 
                workspace=comet_workspace
            )
            experiment.set_name(f"test-{args.src}-{args.tgt}-{args.test_split}")
            experiment.log_parameters(vars(args))
            logger.info("Comet experiment initialized.")
        else:
            logger.warning("Comet API key not found in environment variables")

    # Load test data
    try:
        test_data = load_test_data(args.data_dir, args.test_split, args.src, args.tgt)
        logger.info(f"Loaded test dataset with {len(test_data['src'])} examples")
        
        # Log some sample data for debugging
        logger.info(f"Sample source text: {test_data['src'][0] if test_data['src'] else 'None'}")
        logger.info(f"Sample target text: {test_data['tgt'][0] if test_data['tgt'] else 'None'}")
        
        # Check for empty datasets
        if len(test_data['src']) == 0 or len(test_data['tgt']) == 0:
            raise ValueError(f"Test dataset is empty for split: {args.test_split}")
            
        # Verify data integrity
        if len(test_data['src']) != len(test_data['tgt']):
            raise ValueError(f"Source and target data have different lengths: {len(test_data['src'])} vs {len(test_data['tgt'])}")
            
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        if experiment:
            experiment.log_parameter("error", str(e))
        return
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and args.use_cuda_if_available else "cpu"
    logger.info(f"Using device: {device}")
    
    # Get full language codes
    src_lang_code = get_lang_code(args.src)
    tgt_lang_code = get_lang_code(args.tgt)
    logger.info(f"Source language code: {src_lang_code}, Target language code: {tgt_lang_code}")
    
    # Load the trained model and tokenizer
    try:
        logger.info(f"Loading model from: {args.model_dir}")
        model = MBartForConditionalGeneration.from_pretrained(args.model_dir)
        tokenizer = MBart50TokenizerFast.from_pretrained(args.model_dir)
        model.to(device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        if experiment:
            experiment.log_parameter("error", str(e))
        return
    
    # Create directory for evaluation outputs if it doesn't exist
    os.makedirs(args.save_base_folder, exist_ok=True)
    
    # Generate translations
    try:
        logger.info("Starting translation generation...")
        predictions = generate_translations(
            model=model,
            tokenizer=tokenizer,
            texts=test_data['src'],
            src_lang=src_lang_code,
            tgt_lang=tgt_lang_code,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size
        )
        
        # Compute metrics
        logger.info("Computing metrics...")
        metrics = compute_metrics(predictions, test_data['tgt'])
        
        # Create results dataframe
        df = pd.DataFrame({
            f'{args.src.upper()} (source)': test_data['src'],
            f'{args.tgt.upper()} (generated)': predictions,
            f'{args.tgt.upper()} (reference)': test_data['tgt']
        })
        
        # Save results
        output_file = os.path.join(args.save_base_folder, f'test_{args.test_split}_{args.src}_{args.tgt}.csv')
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")
        
        # Print summary of results
        logger.info(f"Test Results for {args.test_split} ({args.src} -> {args.tgt}):")
        logger.info(f"BLEU: {metrics['bleu']:.4f}")
        logger.info(f"METEOR: {metrics['meteor']:.4f}")
        
        # Log to Comet ML
        if experiment:
            experiment.log_metrics({
                f'test_bleu_{args.src}_{args.tgt}': metrics['bleu'],
                f'test_meteor_{args.src}_{args.tgt}': metrics['meteor'],
                'test_samples': len(test_data['src'])
            })
            
            # Log sample translations (first 100)
            sample_df = df
            experiment.log_table(f"sample_translations_{args.test_split}.csv", tabular_data=sample_df, headers=True)
            
            logger.info("Results logged to Comet ML")
        
        # Also save metrics to a separate file
        metrics_file = os.path.join(args.save_base_folder, f'metrics_{args.test_split}_{args.src}_{args.tgt}.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Test Split: {args.test_split}\n")
            f.write(f"Language Pair: {args.src} -> {args.tgt}\n")
            f.write(f"BLEU Score: {metrics['bleu']:.4f}\n")
            f.write(f"METEOR Score: {metrics['meteor']:.4f}\n")
            f.write(f"Number of Samples: {len(test_data['src'])}\n")
        
        logger.info(f"Metrics saved to: {metrics_file}")
        
    except Exception as e:
        import traceback
        logger.error(f"Error during evaluation: {str(e)}")
        logger.error(traceback.format_exc())
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
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for translation generation")
    parser.add_argument("--save_base_folder", type=str, default="./eval_outputs/", help="Folder to save evaluation outputs")
    parser.add_argument("--use_cuda_if_available", action="store_true", default=False, help="Use GPU if available")
    parser.add_argument("--comet_logging", action="store_true", default=False, help="Enable Comet logging")
    
    args = parser.parse_args()
    main(args)