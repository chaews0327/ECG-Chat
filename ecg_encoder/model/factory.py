import torch
from typing import List, Optional, Union

import ftfy
import html


DEFAULT_CONTEXT_LENGTH = 77  # default context length for OpenAI CLIP


def get_tokenizer(
    model_name: str = '',
    context_length: Optional[int] = None,
    **kwargs,
):
    tokenizer = HFTokenizer(
        model_name,
        context_length=context_length or DEFAULT_CONTEXT_LENGTH,
        **kwargs,
    )
    return tokenizer


class HFTokenizer:
    """HuggingFace tokenizer wrapper"""

    def __init__(
            self,
            tokenizer_name: str,
            context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH,
            strip_sep_token: bool = False,
            language: Optional[str] = None,
            **kwargs
    ):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
        set_lang_fn = getattr(self.tokenizer, 'set_src_lang_special_tokens', None)
        if callable(set_lang_fn):
            self.set_lang_fn = set_lang_fn
        if language is not None:
            self.set_language(language)
        self.context_length = context_length
        self.strip_sep_token = strip_sep_token
        self.tokenizer.add_tokens(["<s>", "</s>"])


    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)


    def clean_text(self, text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        text = " ".join(text.strip().split())    
        return text.strip()
    

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.Tensor:
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length in class init or call.'

        texts = [self.clean_text(text) for text in texts]
        
        input_ids = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors='pt',
            max_length=context_length,
            padding='max_length',
            truncation=True,
        ).input_ids

        if self.strip_sep_token:
            input_ids = torch.where(
                input_ids == self.tokenizer.sep_token_id,
                torch.zeros_like(input_ids),
                input_ids,
            )

        return input_ids
    
    def set_language(self, src_lang):
        if hasattr(self, 'set_lang_fn'):
            self.set_lang_fn(src_lang)
            
            
def get_model_preprocess_cfg(model):
    module = getattr(model, 'ecg', model)
    preprocess_cfg = getattr(module, 'preprocess_cfg', {})
    if not preprocess_cfg:
        seq_length = getattr(module, 'seq_length')
        if seq_length is not None:
            preprocess_cfg['seq_length'] = seq_length
            preprocess_cfg['duration'] = 10
            preprocess_cfg['sampling_rate'] = seq_length / 10

    return preprocess_cfg