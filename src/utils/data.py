import os
from datasets import Dataset, DatasetDict

def read_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

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
    """
    inputs = examples["src"]
    targets = examples["tgt"]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
