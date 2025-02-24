import numpy as np
import evaluate

bleu_metric = evaluate.load("sacrebleu")
meteor_metric = evaluate.load("meteor")

def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])
    meteor_result = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": bleu_result["score"], "meteor": meteor_result["meteor"]}
