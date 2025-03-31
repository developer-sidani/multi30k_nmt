from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def load_seq2seq_model(model_name_or_path: str, src_lang: str, tgt_lang: str):
    """
    Loads the MBartForConditionalGeneration model and MBart50TokenizerFast.
    `src_lang` and `tgt_lang` should be full language codes (e.g., "en_XX", "de_DE", "fr_XX", "cs_CZ").
    """
    model = MBartForConditionalGeneration.from_pretrained(model_name_or_path)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name_or_path)
    return model, tokenizer
