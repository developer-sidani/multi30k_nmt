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
        Evaluate a dataset split. 'dataset' is a dict with keys "src" and "tgt".
        Generates predictions, computes BLEU and METEOR, saves CSV and pickle files,
        and logs to Comet if enabled.
        """
        self.model.eval()
        device = self.args.device
        sources = dataset["src"]
        references = dataset["tgt"]
        predictions = []
        for sentence in sources:
            inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True,
                                      padding="max_length", max_length=self.args.max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                gen_ids = self.model.generate(**inputs,
                                              forced_bos_token_id=self.tokenizer.lang_code_to_id[self.args.tgt_lang],
                                              max_length=self.args.max_length)
            pred = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            predictions.append(pred)
        
        # Compute metrics using our helper.
        metrics = compute_metrics((predictions, references), self.tokenizer)
        metrics.update({"epoch": epoch, "step": step})
        
        # Create a DataFrame with Source, Prediction, and Reference.
        df = pd.DataFrame({
            "Source": sources,
            "Prediction": predictions,
            "Reference": references
        })
        
        # Save CSV and pickle.
        base_path = os.path.join(self.args.save_base_folder, split_name, f"epoch_{epoch}")
        os.makedirs(base_path, exist_ok=True)
        csv_path = os.path.join(base_path, f"{split_name}_epoch{epoch}_step{step}.csv")
        pickle_path = os.path.join(base_path, f"metrics_epoch{epoch}_step{step}.pickle")
        df.to_csv(csv_path, index=False)
        pickle.dump(metrics, open(pickle_path, "wb"))
        
        # Log to Comet if enabled.
        if self.args.comet_logging and self.experiment is not None:
            self.experiment.log_table(csv_path, tabular_data=df, headers=True)
            for k, v in metrics.items():
                if k not in ["epoch", "step"]:
                    self.experiment.log_metric(k, v, step=step, epoch=epoch)
                    
        print(f"Evaluation on {split_name} (epoch {epoch}, step {step}): BLEU={metrics['bleu']}, METEOR={metrics['meteor']}")
        return metrics, df
