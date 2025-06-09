import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_metric
import comet_ml
from tqdm import tqdm
import argparse
from pathlib import Path
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Multi30kDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, src_lang, tgt_lang, max_length=512):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_text = str(self.src_texts[idx]).strip()
        tgt_text = str(self.tgt_texts[idx]).strip()
        
        # Set source language
        self.tokenizer.src_lang = self.src_lang
        
        # Tokenize source
        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Set target language and tokenize
        self.tokenizer.tgt_lang = self.tgt_lang
        tgt_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': src_encoding['input_ids'].squeeze(),
            'attention_mask': src_encoding['attention_mask'].squeeze(),
            'labels': tgt_encoding['input_ids'].squeeze(),
            'decoder_attention_mask': tgt_encoding['attention_mask'].squeeze()
        }

class MBartTrainer:
    def __init__(self, src_lang, tgt_lang, data_dir, output_dir, comet_api_key=None, comet_project=None, comet_workspace=None):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Language code mapping for mBART
        self.lang_codes = {
            'en': 'en_XX',
            'de': 'de_DE', 
            'fr': 'fr_XX',
            'cs': 'cs_CZ'
        }
        
        # Initialize model and tokenizer
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize metrics
        self.bleu_metric = load_metric('bleu')
        self.meteor_metric = load_metric('meteor')
        
        # Initialize Comet ML
        if comet_api_key:
            self.experiment = comet_ml.Experiment(api_key=comet_api_key, project_name=comet_project, workspace=comet_workspace)
            self.experiment.set_name(f"train-{src_lang}-{tgt_lang}")
            # Log basic parameters
            self.experiment.log_parameters({
                'src_lang': src_lang,
                'tgt_lang': tgt_lang,
                'model_name': 'facebook/mbart-large-50'
            })
        else:
            self.experiment = None
            
    def load_data(self):
        """Load training and validation data"""
        # Load training data
        train_src_path = self.data_dir / f"train.{self.src_lang}"
        train_tgt_path = self.data_dir / f"train.{self.tgt_lang}"
        
        with open(train_src_path, 'r', encoding='utf-8') as f:
            train_src = [line.strip() for line in f.readlines()]
        with open(train_tgt_path, 'r', encoding='utf-8') as f:
            train_tgt = [line.strip() for line in f.readlines()]
            
        # Load validation data from subdirectories
        # Each language has its own val_<lang> directory with reference0.0.txt and reference0.1.txt
        # reference0.0.txt is typically the target language
        # reference0.1.txt is typically the source language (English)
        
        if self.src_lang == 'en':
            # English to other language: src=English, tgt=other
            val_src_path = self.data_dir / f"val_{self.tgt_lang}" / "reference0.1.txt"  # English
            val_tgt_path = self.data_dir / f"val_{self.tgt_lang}" / "reference0.0.txt"  # Target language
        else:
            # Other language to English: src=other, tgt=English  
            val_src_path = self.data_dir / f"val_{self.src_lang}" / "reference0.0.txt"  # Source language
            val_tgt_path = self.data_dir / f"val_{self.src_lang}" / "reference0.1.txt"  # English
            
        with open(val_src_path, 'r', encoding='utf-8') as f:
            val_src = [line.strip() for line in f.readlines()]
        with open(val_tgt_path, 'r', encoding='utf-8') as f:
            val_tgt = [line.strip() for line in f.readlines()]
            
        return train_src, train_tgt, val_src, val_tgt
    
    def create_dataloaders(self, train_src, train_tgt, val_src, val_tgt, batch_size=8):
        """Create training and validation dataloaders"""
        train_dataset = Multi30kDataset(
            train_src, train_tgt, self.tokenizer, 
            self.lang_codes[self.src_lang], self.lang_codes[self.tgt_lang]
        )
        val_dataset = Multi30kDataset(
            val_src, val_tgt, self.tokenizer,
            self.lang_codes[self.src_lang], self.lang_codes[self.tgt_lang]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def generate_translation(self, texts, src_lang, tgt_lang, max_length=512):
        """Generate translations for a list of texts"""
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        
        translations = []
        
        for text in tqdm(texts, desc="Generating translations"):
            inputs = self.tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True
                )
            
            translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            translations.append(translation)
            
        return translations
    
    def compute_metrics(self, predictions, references):
        """Compute BLEU and METEOR scores"""
        # Use sacrebleu for consistency with CycleGAN implementation (returns 0-100 range)
        import evaluate
        sacrebleu_metric = evaluate.load('sacrebleu')
        
        # Follow CycleGAN approach: compute BLEU sentence by sentence, then average
        # This matches their compute_metric function exactly
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            # For single reference: [ref] creates a list with one reference
            res = sacrebleu_metric.compute(predictions=[pred], references=[[ref]])
            bleu_scores.append(res['score'])
        
        # Compute corpus-level METEOR
        meteor_score = self.meteor_metric.compute(predictions=predictions, references=references)
        
        return {
            'bleu': sum(bleu_scores) / len(bleu_scores),  # Average of sentence-level BLEU scores
            'meteor': meteor_score['meteor'] * 100  # Convert METEOR to 0-100 range for consistency
        }
    
    def evaluate(self, val_loader, epoch, step):
        """Evaluate model on validation set"""
        self.model.eval()
        
        val_src_texts = []
        val_tgt_texts = []
        val_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Get source texts for generation
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                
                # Decode source texts
                src_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                tgt_texts = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                # Generate predictions
                self.tokenizer.src_lang = self.lang_codes[self.src_lang]
                self.tokenizer.tgt_lang = self.lang_codes[self.tgt_lang]
                
                generated_tokens = self.model.generate(
                    input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[self.lang_codes[self.tgt_lang]],
                    max_length=512,
                    num_beams=5,
                    early_stopping=True
                )
                
                predictions = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                
                val_src_texts.extend(src_texts)
                val_tgt_texts.extend(tgt_texts)
                val_predictions.extend(predictions)
        
        # Compute metrics
        metrics = self.compute_metrics(val_predictions, val_tgt_texts)
        
        # Log to Comet ML
        if self.experiment:
            self.experiment.log_metrics({
                f'val_bleu_{self.src_lang}_{self.tgt_lang}': metrics['bleu'],
                f'val_meteor_{self.src_lang}_{self.tgt_lang}': metrics['meteor']
            }, step=step, epoch=epoch)
            
            # Create and log translation table
            df = pd.DataFrame({
                f'{self.src_lang.upper()} (source)': val_src_texts[:100],  # Limit to first 100 for logging
                f'{self.tgt_lang.upper()} (generated)': val_predictions[:100],
                f'{self.tgt_lang.upper()} (reference)': val_tgt_texts[:100]
            })
            
            csv_path = self.output_dir / f'val_{self.src_lang}_{self.tgt_lang}_epoch_{epoch}.csv'
            df.to_csv(csv_path, index=False)
            self.experiment.log_table(str(csv_path), tabular_data=df, headers=True)
        
        logger.info(f"Epoch {epoch}, Step {step} - BLEU: {metrics['bleu']:.4f}, METEOR: {metrics['meteor']:.4f}")
        
        return metrics
    
    def train(self, num_epochs=15, batch_size=8, learning_rate=3e-5):
        """Train the model"""
        # Log training parameters to Comet ML
        if self.experiment:
            self.experiment.log_parameters({
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'train_data_size': 'will_be_updated',
                'val_data_size': 'will_be_updated'
            })
            
        # Load data
        train_src, train_tgt, val_src, val_tgt = self.load_data()
        train_loader, val_loader = self.create_dataloaders(train_src, train_tgt, val_src, val_tgt, batch_size)
        
        # Update data sizes in Comet ML
        if self.experiment:
            self.experiment.log_parameters({
                'train_data_size': len(train_src),
                'val_data_size': len(val_src)
            })
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
        best_bleu = 0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for step, batch in enumerate(pbar):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Replace padding tokens with -100 for loss calculation
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                # Log training loss to Comet ML
                if self.experiment:
                    self.experiment.log_metric(f'train_loss_{self.src_lang}_{self.tgt_lang}', loss.item(), 
                                             step=epoch * len(train_loader) + step, epoch=epoch)
            
            # Evaluate after each epoch
            metrics = self.evaluate(val_loader, epoch + 1, epoch + 1)
            
            # Save checkpoint
            checkpoint_dir = self.output_dir / f"epoch_{epoch + 1}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            
            # Track best model
            if metrics['bleu'] > best_bleu:
                best_bleu = metrics['bleu']
                best_epoch = epoch + 1
                
                # Save best model
                best_dir = self.output_dir / "best_model"
                best_dir.mkdir(exist_ok=True)
                self.model.save_pretrained(best_dir)
                self.tokenizer.save_pretrained(best_dir)
            
            logger.info(f"Epoch {epoch+1} completed. Average loss: {total_loss/len(train_loader):.4f}")
        
        logger.info(f"Training completed. Best epoch: {best_epoch} with BLEU: {best_bleu:.4f}")
        
        return best_epoch

def main():
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Train mBART model for translation')
    parser.add_argument('--src', required=True, choices=['en', 'de', 'fr', 'cs'], help='Source language')
    parser.add_argument('--tgt', required=True, choices=['en', 'de', 'fr', 'cs'], help='Target language')
    parser.add_argument('--data_dir', required=True, help='Path to data directory')
    parser.add_argument('--output_dir', required=True, help='Path to output directory')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    
    args = parser.parse_args()
    comet_api_key = os.getenv('COMET_API_KEY')
    comet_project = os.getenv('COMET_PROJECT_NAME')
    comet_workspace = os.getenv('COMET_WORKSPACE')
    # Create trainer
    trainer = MBartTrainer(args.src, args.tgt, args.data_dir, args.output_dir, comet_api_key, comet_project, comet_workspace)
    
    # Train model
    best_epoch = trainer.train(args.epochs, args.batch_size, args.learning_rate)
    
    print(f"Best model saved at epoch {best_epoch}")

if __name__ == "__main__":
    main()