import torch
import numpy as np
import evaluate
from typing import List, Dict, Union

# Load evaluation metrics
bleu_metric = evaluate.load('sacrebleu')
meteor_metric = evaluate.load('meteor')


def compute_bleu_score(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Compute BLEU score for the given predictions and references.

    Args:
        predictions: List of predicted translations
        references: List of lists containing reference translations

    Returns:
        Dictionary with BLEU score and related statistics
    """
    results = []

    # Compute BLEU for each prediction against its references
    for pred, refs in zip(predictions, references):
        score = bleu_metric.compute(predictions=[pred], references=[refs])
        results.append(score['score'])

    # Calculate average BLEU score
    avg_bleu = np.mean(results)

    return {
        'bleu': avg_bleu,
        'bleu_scores': results
    }


def compute_meteor_score(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Compute METEOR score for the given predictions and references.

    Args:
        predictions: List of predicted translations
        references: List of lists containing reference translations

    Returns:
        Dictionary with METEOR score and related statistics
    """
    results = []

    # Compute METEOR for each prediction against its references
    for pred, refs in zip(predictions, references):
        # Calculate METEOR against each reference and take the best score
        ref_scores = []
        for ref in refs:
            score = meteor_metric.compute(predictions=[pred], references=[ref])
            ref_scores.append(score['meteor'])

        # Use maximum score among references
        results.append(max(ref_scores) * 100)  # Convert to percentage

    # Calculate average METEOR score
    avg_meteor = np.mean(results)

    return {
        'meteor': avg_meteor,
        'meteor_scores': results
    }


def compute_all_metrics(predictions: List[str], references: List[List[str]]) -> Dict[str, Union[float, List[float]]]:
    """
    Compute all translation metrics for the given predictions and references.

    Args:
        predictions: List of predicted translations
        references: List of lists containing reference translations

    Returns:
        Dictionary with all metrics and their scores
    """
    # Compute individual metrics
    bleu_results = compute_bleu_score(predictions, references)
    meteor_results = compute_meteor_score(predictions, references)

    # Combine all metrics into a single dictionary
    metrics = {
        'bleu': bleu_results['bleu'],
        'meteor': meteor_results['meteor'],
    }

    return metrics