import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from typing import List, Optional, Union


class Seq2SeqModel(nn.Module):
    """
    Sequence to sequence model for neural machine translation using pre-trained mBART model.
    """

    def __init__(
            self,
            model_name_or_path: str = "facebook/mbart-large-50",
            pretrained_path: str = None,
            max_seq_length: int = 64,
            truncation: str = "longest_first",
            padding: str = "max_length",
            src_lang: str = "en_XX",  # Full mBART language code
            tgt_lang: str = "de_DE"  # Full mBART language code
    ):
        super(Seq2SeqModel, self).__init__()

        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.truncation = truncation
        self.padding = padding
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        if pretrained_path is None:
            self.model = MBartForConditionalGeneration.from_pretrained(model_name_or_path)
            self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name_or_path)
        else:
            self.model = MBartForConditionalGeneration.from_pretrained(pretrained_path)
            self.tokenizer = MBart50TokenizerFast.from_pretrained(f"{pretrained_path}/tokenizer/")

        # Set the source language for encoding
        self.tokenizer.src_lang = self.src_lang

    def train_mode(self):
        """Set the model to training mode."""
        self.model.train()

    def eval_mode(self):
        """Set the model to evaluation mode."""
        self.model.eval()

    def forward(
            self,
            sentences: List[str],
            target_sentences: List[str] = None,
            device=None,
    ):
        """
        Forward pass through the model.

        Args:
            sentences: List of input sentences to translate
            target_sentences: Optional list of target sentences for supervised training
            device: Device to place tensors on

        Returns:
            If target_sentences is provided: output ids, decoded sentences, and loss
            Otherwise: output ids and decoded sentences
        """
        # Tokenize input sentences
        inputs = self.tokenizer(sentences,
                                truncation=self.truncation,
                                padding=self.padding,
                                max_length=self.max_seq_length,
                                return_tensors="pt")

        # If target sentences provided, compute supervised loss
        if target_sentences is not None:
            labels = self.tokenizer(target_sentences,
                                    truncation=self.truncation,
                                    padding=self.padding,
                                    max_length=self.max_seq_length,
                                    return_tensors="pt").input_ids

            # Move to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass with loss computation
            output_supervised = self.model(**inputs, labels=labels)

        # Move inputs to device
        inputs = inputs.to(device)

        # Generate translations
        output = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
            max_length=self.max_seq_length
        )

        # Decode the output
        translated_sentences = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        if target_sentences is not None:
            return output, translated_sentences, output_supervised.loss
        else:
            return output, translated_sentences

    def translate(
            self,
            sentences: List[str],
            device=None
    ):
        """
        Translate a list of sentences.

        Args:
            sentences: List of input sentences to translate
            device: Device to place tensors on

        Returns:
            List of translated sentences
        """
        inputs = self.tokenizer(sentences,
                                truncation=self.truncation,
                                padding=self.padding,
                                max_length=self.max_seq_length,
                                return_tensors="pt")

        inputs = inputs.to(device)
        output = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
            max_length=self.max_seq_length
        )

        # Decode the outputs
        translated_sentences = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return translated_sentences

    def save_model(
            self,
            path: Union[str, None]
    ):
        """
        Save the model and tokenizer to disk.

        Args:
            path: Directory to save the model to
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(f"{path}/tokenizer/")