# Copyright 2023 MosaicML byod authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenization task, convert text to tokenizer then materialize the Spark dataframe."""

import os
import time
from enum import Enum
from typing import Any, Optional, Callable, TypeVar, Union, Tuple, Dict
from copy import deepcopy

try:
    from pyspark.sql.dataframe import DataFrame
except ImportError as e:
    e.msg = get_import_exception_message(e.name, 'spark')
    raise e

#from streaming.base.converters import dataframeToMDS
from byod import MaterializeTask
from byod.utils import build_hf_dataset_wrapper, get_import_exception_message

import logging
logger = logging.getLogger(__name__)

class Tokenize(MaterializeTask):
    def __init__(self,
                 df: DataFrame,
                 task_key: str,
                 out_root: str,
                 tokenizer: str,
                 seqlen: int,
                 configs_or_yaml: Optional[Union[Dict, str]] = None,
                 *args: Any,
                 **kwargs: Any,
                 ):
        super().__init__(df, task_key, out_root, *args, **kwargs)
        """Task for tokenize a dataframe"""

        # shared cache for workers so as not to DDOS HF
        self.tokenizer_prefix = os.path.join(self.out_root, 'tokenizer_cache/')
        self.hf_datasets_cache_prefix = os.path.join(self.tokenizer_prefix, 'hf_datasets_cache/')
        self.hf_transformers_cache_prefix = os.path.join(self.tokenizer_prefix, 'hf_transformers_cache/')

        self.default_ppfn_kwargs = {
            'eos_text' : '<|endoftext|>',
            'compression' : "zstd",
            'split' : "train",
            'no_wrap' : False,
            'bos_text' : '',
            'key' : 'text', # Need to specify
        }

        self.default_mds_kwargs = {
            'compression': 'zstd:7',
            'hashes': ['sha1','xxh64'],
            'size_limit': 1<<27,
            'progress_bar':1,
            'columns':{'tokens': 'bytes'},
            'keep_local': True,
        }

        self.tokenizer = tokenizer
        self.seqlen = seqlen

        self.sanity_checks()

    def sanity_checks(self):
        super().sanity_checks()

        if self.tokenizer is None or self.seqlen is None:
            raise ValueError(f"Tokenizer task expects tokenizer, seqlen to be set")

        if self.tokenizer not in ['gpt-neox-20b', 'tiktoken']:
            raise ValueError(f"This tokenize task has not been tested other than gpt-neox-20b or tiktoken: Got {self.tokenizer}!")

        if self.tokenizer == 'gpt-neox-20b':
            predownload_tokenizer_path = os.path.join(self.tokenizer_prefix, 'gpt-neox-20b')
            assert os.path.exists(predownload_tokenizer_path) and len(os.listdir(predownload_tokenizer_path)) != 0, f"gpt-neox-20b is private. You need to download gpt-neox-20b beforehand and put in {predownload_tokenizer_path}"
            assert os.path.exists(os.path.join(predownload_tokenizer_path, 'config.json')), "Provided pretrained tokenizer is not valid, missing config.json"

        if 'text' not in self.df.columns:
            raise ValueError(f"Spark DataFrame must contain a text field column named text. Rename if necessary")


    def run(self):
        from byod import df_to_mds

        mds_kwargs = deepcopy(self.default_mds_kwargs)
        mds_kwargs['out'] = self.task_output

        ppfn_kwargs = deepcopy(self.default_ppfn_kwargs)
        ppfn_kwargs['concat_tokens'] = self.seqlen
        ppfn_kwargs['tokenizer'] = self.tokenizer
        ppfn_kwargs['HF_DATASETS_CACHE'] = self.hf_datasets_cache_prefix
        ppfn_kwargs['TRANSFORMERS_CACHE'] = self.hf_transformers_cache_prefix
        ppfn_kwargs['tokenizer_prefix'] = self.tokenizer_prefix

        df_to_mds(self.df,
                  merge_index = True,
                  mds_kwargs = mds_kwargs,
                  udf_iterable = build_hf_dataset_wrapper,
                  udf_kwargs = ppfn_kwargs)

