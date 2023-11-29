# Copyright 2023 MosaicML byod authors
# SPDX-License-Identifier: Apache-2.0

"""Filtration task, apply filters to text."""

#from blingfire import text_to_words
from datetime import datetime
from collections.abc import Iterable
from typing import Optional, Dict, Union, Any
import itertools
import os
try:
    from pyspark.sql.dataframe import DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.types import BooleanType
except ImportError as e:
    e.msg = get_import_exception_message(e.name, 'spark')
    raise e

from byod import TransformationTask
from byod.utils import build_hf_dataset_wrapper, get_import_exception_message

import logging
logger = logging.getLogger(__name__)

def text_to_words(s): # need this temporarily for testing on macw here blingfire can't be installed.
    return s

def most_frequent(ls, skiplist=[]):
    d = {}
    for l in ls:
        d[l] = d.get(l, 0) + 1

    item, freq = None, -1
    for k, v in d.items():
        if k in skiplist:
            continue
        if v > freq:
            item, freq = k, v

    return item, freq

most_frequent_english_words = [
    'the', 'of', 'and', 'to', 'a', 'in', 'is', 'that', 'was', 'it', 'for', 'on',
    'with', 'he', 'be', 'i', 'by', 'as', 'at', 'you', 'are', 'his', 'had',
    'not', 'this', 'have', 'from', 'but', 'which', 'she', 'they', 'or', 'an',
    'her', 'were', 'there', 'we', 'their', 'been', 'has', 'will', 'one', 'all',
    'would', 'can', 'if', 'who', 'more', 'when', 'said', 'do', 'what', 'about',
    'its', 'so', 'up', 'into', 'no', 'him', 'some', 'could', 'them', 'only',
    'time', 'out', 'my', 'two', 'other', 'then', 'may', 'over', 'also', 'new',
    'like', 'these', 'me', 'after', 'first', 'your', 'did', 'now', 'any',
    'people', 'than', 'should', 'very', 'most', 'see', 'where', 'just', 'made',
    'between', 'back', 'way', 'many', 'years', 'being', 'our', 'how', 'work'
]

min_len = 1
max_len = 50000
max_char_len = 500_000
alpha_ratio = 0.0
max_freq_ratio = 0.0


@F.udf(returnType=BooleanType())
def keep(s) -> bool:
    if not s:
        return False

    s = s.lower() # Fixing to text since thats what mc4 uses
    if len(s) > max_char_len:
        return False
    ws = [x for x in text_to_words(s).split() if x.isalnum()]
    return min_len <= len(ws) <= max_len

@F.udf(returnType=BooleanType())
def keep_w_freq_check(s) -> bool:
    s = s.lower() # Fixing to text since thats what mc4 uses

    if len(s) > max_char_len:
        return False

    # Check whether the most frequent character is alphabetic.
    no_whitespace = ''.join(s.split())
    if not most_frequent(no_whitespace)[0].isalpha():
        return False

    # Check that most characters are alphanumeric.
    num_alphas = len([a for a in no_whitespace if a.isalnum()])
    if float(num_alphas) / len(no_whitespace) < alpha_ratio:
        return False

    # Split into words.
    ws = [x for x in text_to_words(s).split() if x.isalnum()]

    # Ensure there are enough words.
    if len(ws) < min_len:
        return False
    if len(ws) > max_len:
        return False

    # Verify that the most frequent word isn't too frequent.
    w, freq = most_frequent(ws, most_frequent_english_words)
    frac = float(freq) / len(ws)
    if w is not None and (len(ws) > 500 and frac > 0.075) or (len(ws) <= 500 and frac > max_freq_ratio):
        return False

    return True


class Filter(TransformationTask):
    def __init__(self,
                 df: DataFrame,
                 task_key: str,
                 text_col: str = 'text',
                 min_len: int = 200,
                 max_len: int = 50000,
                 max_char_len: int = 500_000,
                 alpha_ratio: float = 0.92,
                 max_freq_ratio: float = 0.3,
                 configs_or_yaml: Optional[Union[Dict, str]] = None,
                 *args: Any,
                 **kwargs: Any,
                 ):
        super().__init__(df, task_key, *args, **kwargs)
        """Task for tokenize a dataframe"""

        self.text_col = text_col
        self.min_len = min_len
        self.max_len = max_len
        self.alpha_ratio = alpha_ratio
        self.max_freq_ratio = max_freq_ratio
        self.max_char_len = max_char_len

        self.sanity_checks()

    def sanity_checks(self):
        super().sanity_checks()

        if self.text_col not in self.df.columns:
            raise ValueError(f"Spark DataFrame must contain a text field column named text. Rename if necessary")

        assert (0 < self.min_len) and (0 < self.max_len) and (0 <= self.alpha_ratio <= 1) and (0 <= self.max_freq_ratio <= 1) and (self.max_char_len > 0), f"Got {self.min_len}, {self.max_len}, {self.alpha_ratio}, {self.max_freq_ratio}, {self.max_char_len}"

    def run(self):
        filtered_df = self.df.filter(keep(self.text_col)).cache()
        return filtered_df

