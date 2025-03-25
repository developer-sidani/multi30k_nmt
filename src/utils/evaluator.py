import os
import pickle
import numpy as np
import pandas as pd
import torch
from src.utils.metrics import compute_metrics

class Evaluator:
    def __init__(self, model, tokenizer, args, experiment=None):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.experiment = experiment

    def evaluate_split(self, dataset, split_name: str, epoch: int, step: int):
        """
        Evaluate a dataset split. Generates predictions in batches,
        computes BLEU/METEOR, and logs results.
        """
        self.model.eval()
        device = self.args.device
        
        # Ensure sources and references are lists of strings
        sources = dataset["src"] if isinstance(dataset["src"], list) else dataset["src"].tolist()
        references = dataset["tgt"] if isinstance(dataset["tgt"], list) else dataset["tgt"].tolist()

        # Create output folder if it doesn't exist
        base_path = os.path.join(self.args.save_base_folder, split_name, f"epoch_{epoch}")
        os.makedirs(base_path, exist_ok=True)
        
        print(f"Evaluating {len(sources)} examples for {split_name}")
        
        # Process in batches if dataset is large
        batch_size = self.args.eval_batch_size if hasattr(self.args, 'eval_batch_size') else 16
        num_samples = len(sources)
        predictions = []
        
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_sources = sources[i:batch_end]
            
            # Debug: print sample from batch
            if i == 0:
                print(f"Batch source example: {batch_sources[0]}")
            
            # Tokenize the batch
            inputs = self.tokenizer(batch_sources, 
                                    return_tensors="pt", 
                                    padding="max_length",
                                    truncation=True, 
                                    max_length=self.args.max_length)
            
            # Debug: print input keys
            if i == 0:
                print(f"Input keys: {inputs.keys()}")
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate translations
            with torch.no_grad():
                gen_ids = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[self.args.tgt_lang],
                    max_length=self.args.max_length
                )
            
            # Debug: print shape of generated IDs
            if i == 0:
                print(f"Generated IDs shape: {gen_ids.shape}")
            
            # Decode batch predictions
            batch_predictions = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            
            # Debug: print sample prediction
            if i == 0:
                print(f"Batch prediction example: {batch_predictions[0]}")
            
            predictions.extend(batch_predictions)
            
            print(f"Processed {batch_end}/{num_samples} examples")

        # Ensure all inputs to compute_metrics are properly formatted
        eval_pred = (predictions, references)
        
        # Compute metrics
        metrics = compute_metrics(eval_pred, self.tokenizer)
        metrics.update({"epoch": epoch, "step": step})

        # Save evaluation results
        df = pd.DataFrame({"Source": sources, "Prediction": predictions, "Reference": references})
        
        csv_path = os.path.join(base_path, f"{split_name}_epoch{epoch}_step{step}.csv")
        pickle_path = os.path.join(base_path, f"metrics_epoch{epoch}_step{step}.pickle")
        
        df.to_csv(csv_path, index=False)
        with open(pickle_path, 'wb') as f:
            pickle.dump(metrics, f)

        # Log to Comet if enabled
        if self.args.comet_logging and self.experiment is not None:
            self.experiment.log_table(csv_path, tabular_data=df, headers=True)
            for k, v in metrics.items():
                if k not in ["epoch", "step"]:
                    self.experiment.log_metric(f"{split_name}_{k}", v, step=step, epoch=epoch)
                    
        # Log some examples
        print(f"\nEvaluation on {split_name} (epoch {epoch}, step {step}):")
        print(f"BLEU = {metrics['bleu']:.2f}, METEOR = {metrics['meteor']:.2f}")
        
        # Print a few examples
        num_examples = min(3, len(df))
        print("\nExample translations:")
        for i in range(num_examples):
            print(f"Source: {df['Source'].iloc[i]}")
            print(f"Prediction: {df['Prediction'].iloc[i]}")
            print(f"Reference: {df['Reference'].iloc[i]}")
            print("-" * 50)
        
        return metrics, df