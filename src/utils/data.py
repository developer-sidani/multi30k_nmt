import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union


class NMTDataset(Dataset):
    """
    Dataset for Neural Machine Translation with source and target sentences.
    """

    def __init__(
            self,
            source_sentences: List[str],
            target_sentences: Optional[List[str]] = None,
    ):
        """
        Initialize the dataset with source and target sentences.

        Args:
            source_sentences: List of source language sentences
            target_sentences: List of target language sentences (optional for inference)
        """
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        if self.target_sentences is not None:
            return self.source_sentences[idx], self.target_sentences[idx]
        else:
            return self.source_sentences[idx]


class ReferenceDataset(Dataset):
    """
    Dataset for evaluation with references.
    """

    def __init__(
            self,
            source_sentences: List[str],
            references: List[List[str]],
    ):
        """
        Initialize the dataset with source sentences and references.

        Args:
            source_sentences: List of source language sentences
            references: List of lists of reference translations
        """
        self.source_sentences = source_sentences
        self.references = references

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        return self.source_sentences[idx], self.references[idx]

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for reference dataset.

        Args:
            batch: List of (source, references) tuples

        Returns:
            Tuple of (sources, references)
        """
        sources, references = zip(*batch)
        return list(sources), list(references)


def load_dataset(src_path: str, tgt_path: Optional[str] = None, max_samples: Optional[int] = None) -> Tuple[
    List[str], Optional[List[str]]]:
    """
    Load source and optional target sentences from files.

    Args:
        src_path: Path to source language file
        tgt_path: Optional path to target language file
        max_samples: Maximum number of samples to load

    Returns:
        Tuple of (source_sentences, target_sentences)
    """
    with open(src_path, 'r', encoding='utf-8') as f:
        src_sentences = [line.strip() for line in f.readlines()]

    if max_samples:
        src_sentences = src_sentences[:max_samples]

    if tgt_path:
        with open(tgt_path, 'r', encoding='utf-8') as f:
            tgt_sentences = [line.strip() for line in f.readlines()]

        if max_samples:
            tgt_sentences = tgt_sentences[:max_samples]

        # Ensure source and target have the same length
        min_len = min(len(src_sentences), len(tgt_sentences))
        src_sentences = src_sentences[:min_len]
        tgt_sentences = tgt_sentences[:min_len]

        return src_sentences, tgt_sentences

    return src_sentences, None


def load_references(ref_path: str, n_references: int = 1, max_samples: Optional[int] = None) -> List[List[str]]:
    """
    Load reference translations for evaluation.

    Args:
        ref_path: Base path to references
        n_references: Number of references per source sentence
        max_samples: Maximum number of samples to load

    Returns:
        List of reference lists, where each inner list contains alternative translations
    """
    references = []

    # Load each reference file
    for i in range(n_references):
        ref_file = f"{ref_path}/reference{i}.txt"
        with open(ref_file, 'r', encoding='utf-8') as f:
            refs = [line.strip() for line in f.readlines()]

        if max_samples:
            refs = refs[:max_samples]

        # Initialize references list if this is the first file
        if i == 0:
            references = [[ref] for ref in refs]
        else:
            # Add additional references
            for j, ref in enumerate(refs):
                if j < len(references):  # Safety check
                    references[j].append(ref)

    return references


def get_language_code(lang: str) -> str:
    """
    Convert short language code to full mBART language code.

    Args:
        lang: Short language code (e.g., 'en', 'de')

    Returns:
        Full mBART language code
    """
    lang_map = {
        'ar': 'ar_AR',
        'cs': 'cs_CZ',
        'de': 'de_DE',
        'en': 'en_XX',
        'es': 'es_XX',
        'et': 'et_EE',
        'fi': 'fi_FI',
        'fr': 'fr_XX',
        'gu': 'gu_IN',
        'hi': 'hi_IN',
        'it': 'it_IT',
        'ja': 'ja_XX',
        'kk': 'kk_KZ',
        'ko': 'ko_KR',
        'lt': 'lt_LT',
        'lv': 'lv_LV',
        'my': 'my_MM',
        'ne': 'ne_NP',
        'nl': 'nl_XX',
        'ro': 'ro_RO',
        'ru': 'ru_RU',
        'si': 'si_LK',
        'tr': 'tr_TR',
        'vi': 'vi_VN',
        'zh': 'zh_CN',
        'af': 'af_ZA'
    }
    return lang_map.get(lang, 'en_XX')  # Default to English if code not found