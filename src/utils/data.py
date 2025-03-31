import os
from datasets import Dataset, DatasetDict

def read_file(filepath: str):
    """
    Read a file and return its contents as a list of strings, with empty lines filtered out.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f]
            # Filter out empty lines
            lines = [line for line in lines if line.strip()]
            
            # Check if we have any content
            if not lines:
                print(f"Warning: File {filepath} is empty or contains only whitespace")
                
            return lines
    except Exception as e:
        print(f"Error reading file {filepath}: {str(e)}")
        raise

def get_lang_code(lang):
	LANG_MAP = {
				'ar': 'ar_AR', 'cs': 'cs_CZ', 'de': 'de_DE', 'en': 'en_XX', 'es': 'es_XX', 'et': 'et_EE', 
				'fi': 'fi_FI', 'fr': 'fr_XX', 'gu': 'gu_IN', 'hi': 'hi_IN', 'it': 'it_IT', 'ja': 'ja_XX', 
				'kk': 'kk_KZ', 'ko': 'ko_KR', 'lt': 'lt_LT', 'lv': 'lv_LV', 'my': 'my_MM', 'ne': 'ne_NP', 
				'nl': 'nl_XX', 'ro': 'ro_RO', 'ru': 'ru_RU', 'si': 'si_LK', 'tr': 'tr_TR', 'vi': 'vi_VN', 
				'zh': 'zh_CN', 'af': 'af_ZA'
			}
	return LANG_MAP[lang]
def load_local_dataset(data_dir: str, src_suffix: str, tgt_suffix: str, test_year: str="2016", test_dataset: str="flickr") -> DatasetDict:
    """
    Expects files to be named as:
      - train.<src_suffix>, train.<tgt_suffix>
      - test_2016_val.<src_suffix>, test_2016_val.<tgt_suffix>
      - test_2016_flickr.<src_suffix>, test_2016_flickr.<tgt_suffix>
    Returns a DatasetDict with splits: "train", "validation", "test".
    """
    train_src = read_file(os.path.join(data_dir, f"train.{src_suffix}"))
    train_tgt = read_file(os.path.join(data_dir, f"train.{tgt_suffix}"))
    val_src   = read_file(os.path.join(data_dir, f"test_2016_val.{src_suffix}"))
    val_tgt   = read_file(os.path.join(data_dir, f"test_2016_val.{tgt_suffix}"))
    test_src  = read_file(os.path.join(data_dir, f"test_2016_flickr.{src_suffix}"))
    test_tgt  = read_file(os.path.join(data_dir, f"test_2016_flickr.{tgt_suffix}"))
    
    train_dataset = Dataset.from_dict({"src": train_src, "tgt": train_tgt})
    val_dataset   = Dataset.from_dict({"src": val_src,   "tgt": val_tgt})
    test_dataset  = Dataset.from_dict({"src": test_src,  "tgt": test_tgt})
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

def preprocess_function(examples, tokenizer, max_length=128):
    """
    Tokenizes inputs (from "src") and targets (from "tgt").
    For mBART50, we encode source and target separately.
    """
    # Tokenize inputs (source text)
    model_inputs = tokenizer(examples["src"], max_length=max_length, truncation=True, padding="max_length")
    
    # Tokenize targets (target text)
    # We use standard tokenization without text_target parameter
    labels = tokenizer(examples["tgt"], max_length=max_length, truncation=True, padding="max_length")
    
    # Replace padding token id with -100 so it's ignored in the loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs