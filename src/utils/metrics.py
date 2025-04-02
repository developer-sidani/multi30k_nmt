import numpy as np
import evaluate

bleu_metric = evaluate.load("sacrebleu")
meteor_metric = evaluate.load("meteor")


def compute_metrics(eval_pred, tokenizer, experiment=None):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])
    meteor_result = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)

    # Create metrics dictionary
    metrics = {
        "bleu": bleu_result["score"],
        "meteor": meteor_result["meteor"]
    }

    # Log metrics to the provided Comet experiment if available
    if experiment is not None:
        experiment.log_metrics(metrics)

    return metrics
