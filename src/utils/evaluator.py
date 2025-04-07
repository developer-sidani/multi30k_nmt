import os
import torch
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Optional, Tuple, Union

from torch.utils.data import DataLoader
from src.utils.metrics import compute_all_metrics


class Evaluator:
    """
    Evaluation class for Seq2Seq models.
    """

    def __init__(self, model, args, experiment=None):
        """
        Initialize the evaluator.

        Args:
            model: The Seq2Seq model to evaluate
            args: Command line arguments
            experiment: Comet ML experiment for logging
        """
        self.model = model
        self.args = args
        self.experiment = experiment

    def evaluate(
            self,
            dataloader: DataLoader,
            phase: str = 'validation',
            epoch: int = 0,
            step: int = 0,
            src_lang: str = 'en',
            tgt_lang: str = 'de',
            test_set: str = 'flickr2016'
    ) -> Dict[str, float]:
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader: DataLoader containing source sentences and references
            phase: 'validation' or 'test'
            epoch: Current epoch number
            step: Current training step
            src_lang: Source language code
            tgt_lang: Target language code
            test_set: Name of the test set (e.g., 'flickr2016')

        Returns:
            Dictionary with evaluation metrics
        """
        print(f'Starting {phase} evaluation for {src_lang}->{tgt_lang} on {test_set}...')

        self.model.eval_mode()  # Set model to evaluation mode

        source_sentences = []
        references = []
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                # Unpack the batch
                src_batch, ref_batch = batch

                # Generate translations
                _, translated = self.model.translate(src_batch, device=self.args.device)

                # Store results
                source_sentences.extend(src_batch)
                references.extend(ref_batch)
                predictions.extend(translated)

        # Compute metrics
        metrics = compute_all_metrics(predictions, references)
        metrics['epoch'] = epoch
        metrics['step'] = step
        metrics['src_lang'] = src_lang
        metrics['tgt_lang'] = tgt_lang
        metrics['test_set'] = test_set

        # Create directory for results if it doesn't exist
        if phase == 'validation':
            base_path = f"{self.args.save_base_folder}/epoch_{epoch}/"
        else:
            base_path = f"{self.args.save_base_folder}/test/{test_set}/"

        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        # Save metrics
        suffix = f"{src_lang}_{tgt_lang}_{epoch}"
        if phase == 'test':
            suffix += f"_{test_set}"

        pickle.dump(metrics, open(f"{base_path}metrics_{suffix}.pickle", 'wb'))

        # Create and save dataframe with predictions
        df = pd.DataFrame()
        df['source'] = source_sentences
        df['prediction'] = predictions

        # Add references to dataframe
        for i in range(len(references[0])):
            df[f'reference_{i + 1}'] = [ref[i] if i < len(ref) else "" for ref in references]

        df.to_csv(f"{base_path}predictions_{suffix}.csv", sep=',', header=True, index=False)

        # Log to Comet ML if available
        if self.experiment:
            context = self.experiment.validate if phase == 'validation' else self.experiment.test
            with context():
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    if metric_name not in ['epoch', 'step', 'src_lang', 'tgt_lang', 'test_set']:
                        self.experiment.log_metric(
                            f"{metric_name}_{src_lang}_{tgt_lang}",
                            metric_value,
                            step=step,
                            epoch=epoch
                        )

                # Log prediction table
                self.experiment.log_table(
                    f"predictions_{src_lang}_{tgt_lang}_{test_set}.csv",
                    tabular_data=df,
                    headers=True
                )

        # Print metrics
        print(f"Evaluation results for {src_lang}->{tgt_lang} on {test_set}:")
        for metric_name, metric_value in metrics.items():
            if metric_name not in ['epoch', 'step', 'src_lang', 'tgt_lang', 'test_set']:
                print(f"  {metric_name}: {metric_value:.4f}")

        return metrics