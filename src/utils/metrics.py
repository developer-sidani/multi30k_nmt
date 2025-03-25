import numpy as np
import evaluate

bleu_metric = evaluate.load("sacrebleu")
meteor_metric = evaluate.load("meteor")

def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    
    # Ensure predictions and labels are properly formatted
    if isinstance(predictions[0], str):
        # Already decoded predictions
        decoded_preds = predictions
    else:
        # Need to decode token IDs
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
    if isinstance(labels[0], str):
        # Already decoded labels
        decoded_labels = labels
    else:
        # Need to decode token IDs
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Debug info
    print(f"Sample prediction for metrics: {decoded_preds[0]}")
    print(f"Sample reference for metrics: {decoded_labels[0]}")
    
    # Compute BLEU
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])
    
    # Compute METEOR
    meteor_result = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {"bleu": bleu_result["score"], "meteor": meteor_result["meteor"]}
