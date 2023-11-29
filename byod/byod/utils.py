# Copyright 2023 MosaicML byod authors
# SPDX-License-Identifier: Apache-2.0

"""Utility and helper functions."""

import logging
import os
import itertools
from enum import Enum
from collections.abc import Iterable
from typing import Dict, Iterator, List, Optional, Any
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset
from llmfoundry.data import ConcatTokensDataset, NoConcatDataset
from llmfoundry.tokenizers import TiktokenTokenizerWrapper
import datasets as hf_datasets
import pandas as pd
from transformers import PreTrainedTokenizerBase
from torch.utils.data import DataLoader, IterableDataset

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer
import threading
from datetime import datetime
from pytz import timezone
import pytz
import time

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    SUCCESS = 0
    FAILED = 1

class ConcatMode(Enum):
    NO_CONCAT = 'NO_CONCAT'
    CONCAT_TOKENS = 'CONCAT_TOKENS'

def hyperparameter_combinations_generator(parameters: Dict,
                                          excludes: Optional[Dict] = None) -> Iterator:
    """Iterator over combination of parameters.

    Args:
        parameters (Dict): {param name : [value1, value2, ...], param_name: value ...}
        excludes (Dict): for param in exclude, simply take the parameters[param][exclude[param]] value for it
    Returns:
        (Iterable): iterator of combination
    """
    arrays = []
    keys = []
    static = {}

    for k, v in parameters.items():
        if not isinstance(v, Iterable) or len(v) <= 1:
            static[k] = v
            continue

        if excludes is not None and k in excludes:
            static[k] = list(v)[excludes[k]] if excludes[k] is not None else list(v)[0]
            continue

        arrays.append(v)
        keys.append(k)

    d = static.copy()
    for comb in itertools.product(*arrays):
        d.update(dict(zip(keys, comb)))
        yield d.copy()


def build_hf_dataset_wrapper(df: pd.DataFrame, **args: Any)-> IterableDataset:
    """Wrapper of build_hf_dataset to set parameters

    Args:
        df (pandas.DataFrame): The input pandas DataFrame that needs to be processed.
        kwargs : Additional arguments to build_hf_dataset
    Returns:
        iterable obj
    """
    assert("HF_DATASETS_CACHE" in args.keys())
    assert("TRANSFORMERS_CACHE" in args.keys())
    assert("split" in args.keys())

    os.environ["HF_DATASETS_CACHE"] = args['HF_DATASETS_CACHE']
    os.environ["TRANSFORMERS_CACHE"] = args['TRANSFORMERS_CACHE']

    if args['concat_tokens'] is not None:
        mode = ConcatMode.CONCAT_TOKENS
        if args['tokenizer'] == 'gpt-neox-20b':
            pretrained_tokenizer_path = os.path.join(args['tokenizer_prefix'], 'gpt-neox-20b')
            tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_path,
                                                      cache_dir=os.path.join(os.environ["TRANSFORMERS_CACHE"], 'gpt-neox-20b'),
                                                      trust_remote_code=True)
        else:
            tokenizer = TiktokenTokenizerWrapper(model_name='gpt-4')

        # we will enforce length, so suppress warnings about sequences too long for the model
        tokenizer.model_max_length = int(1e30)
    else:
        mode = ConcatMode.NO_CONCAT
        tokenizer = None

    dataset = build_hf_dataset(data=df,
                               split=args['split'],
                               mode=mode,
                               max_length=args.get('concat_tokens', None),
                               bos_text=args.get('bos_text', None),
                               eos_text=args.get('eos_text', None),
                               no_wrap=args.get('no_wrap', None),
                               tokenizer=tokenizer)
    return dataset

def build_hf_dataset(
    data: pd.DataFrame,
    split: str,
    mode: ConcatMode,
    max_length: Optional[int] = None,
    bos_text: str = '',
    eos_text: str = '',
    no_wrap: bool = False,
    tokenizer: PreTrainedTokenizerBase = None,
) -> IterableDataset:
    """Build an IterableDataset over the HF C4 or pile source data.

    Args:
        data (DataFrame): subset of input data
        split (str): Split name.
        mode (ConcatMode): NO_CONCAT, or CONCAT_TOKENS
        max_length (int): The length of concatenated tokens
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        no_wrap (bool): if concatenating, whether to wrap text across `max_length` boundaries
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use
        data_subset (str): Referred to as "name" in HuggingFace datasets.load_dataset.
            Typically "all" (The Pile) or "en" (c4).

    Returns:
        An IterableDataset.
    """

    hf_dataset = hf_datasets.Dataset.from_pandas(df=data, split=split)
    if mode == ConcatMode.NO_CONCAT:
        dataset = NoConcatDataset(hf_dataset)
    else:
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(
                f'{tokenizer=} must be of type PreTrainedTokenizerBase')
        if max_length is None:
            raise ValueError(f'max_length must be set.')
        if bos_text + eos_text == '':
            test_tokens = tokenizer('test')
            if test_tokens['input_ids'][
                    0] != tokenizer.bos_token_id and test_tokens['input_ids'][
                        -1] != tokenizer.eos_token_id:
                tok_error_msg = 'This tokenizer does not insert an EOS nor BOS token. '
                tok_error_msg += 'Concatenating with this tokenizer will result in sequences being '
                tok_error_msg += 'attached without a separating token. Please use another tokenizer, '
                tok_error_msg += 'such as facebook/opt-125m, or specify EOS/BOS text with e.g. '
                tok_error_msg += '--bos_text=<|endoftext|>.'
                raise ValueError(tok_error_msg)
        dataset = ConcatTokensDataset(hf_dataset=hf_dataset,
                                      tokenizer=tokenizer,
                                      max_length=max_length,
                                      bos_text=bos_text,
                                      eos_text=eos_text,
                                      no_wrap=no_wrap)
    return dataset


class FolderObserver:
    """A wrapper class of WatchDog."""

    def __init__(self, directory: str):
        """Specify the download directory to monitor."""
        patterns = ['*']
        ignore_patterns = None
        ignore_directories = False
        case_sensitive = True
        self.average_file_size = 0

        self.file_count = 0
        self.file_size = 0

        if not os.path.exists(directory):
            os.makedirs(directory)

        self.directory = directory
        self.get_directory_info()

        self.my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns,
                                                            ignore_directories, case_sensitive)
        self.my_event_handler.on_created = self.on_created
        self.my_event_handler.on_deleted = self.on_deleted
        self.my_event_handler.on_modified = self.on_modified
        self.my_event_handler.on_moved = self.on_moved

        go_recursively = True
        self.observer = Observer()
        self.observer.schedule(self.my_event_handler, directory, recursive=go_recursively)
        self.tik = time.time()

    def start(self):
        return self.observer.start()

    def stop(self):
        return self.observer.stop()

    def join(self):
        return self.observer.join()

    def get_directory_info(self):
        self.file_count = file_count = 0
        self.file_size = total_file_size = 0
        for root, _, files in os.walk(self.directory):
            for file in files:
                file_count += 1
                file_path = os.path.join(root, file)
                total_file_size += os.path.getsize(file_path)
        self.file_count, self.file_size = file_count, total_file_size

    def on_created(self, event: Any):
        self.file_count += 1

    def on_deleted(self, event: Any):
        self.file_count -= 1

    def on_modified(self, event: Any):
        pass

    def on_moved(self, event: Any):
        pass


def monitor_directory_changes(interval: int = 5):
    """Dataset downloading monitor. Keep file counts N and file size accumulation.

    Approximate dataset size by N * avg file size.
    """

    def decorator(func: Any):

        def wrapper(repo_id: str,
                    local_dir: str,
                    #revision: Optional[str],
                    #max_workers: int,
                    #token: str,
                    #allow_patterns: Optional[List[str]],
                    *args: Any, **kwargs: Any):
            event = threading.Event()
            observer = FolderObserver(local_dir)

            def beautify(kb: int):
                mb = kb // (1024)
                gb = mb // (1024)
                if gb >= 1:
                    return str(gb) + 'GB'
                elif mb >= 1:
                    return str(mb) + 'MB'
                else:
                    return str(kb) + 'KB'

            def monitor_directory():
                observer.start()
                while not event.is_set():
                    try:
                        elapsed_time = int(time.time() - observer.tik)
                        if observer.file_size > 1e9:  # too large to keep an accurate count of the file size
                            if observer.average_file_size == 0:
                                observer.average_file_size = observer.file_size // observer.file_count
                                logger.warning(
                                    f'approximately: average file size = {beautify(observer.average_file_size//1024)}'
                                )
                            kb = observer.average_file_size * observer.file_count // 1024
                        else:
                            observer.get_directory_info()
                            b = observer.file_size
                            kb = b // 1024

                        sz = beautify(kb)
                        cnt = observer.file_count

                        if elapsed_time % 10 == 0:
                            logger.warning(
                                f'Downloaded {cnt} files, Total approx file size = {sz}, Time Elapsed: {elapsed_time} seconds.'
                            )

                        if elapsed_time > 0 and elapsed_time % 120 == 0:
                            observer.get_directory_info(
                            )  # Get the actual stats by walking through the directory
                            observer.average_file_size = observer.file_size // observer.file_count
                            logger.warning(
                                f'update average file size to {beautify(observer.average_file_size//1024)}'
                            )

                        time.sleep(1)
                    except Exception as exc:
                        # raise RuntimeError("Something bad happened") from exc
                        logger.warning(str(exc))
                        time.sleep(1)
                        continue

            monitor_thread = threading.Thread(target=monitor_directory)
            monitor_thread.start()

            try:
                result = func(repo_id, local_dir, *args, **kwargs)
                return result
            finally:
                observer.get_directory_info(
                )  # Get the actual stats by walking through the directory
                logger.warning(
                    f'Done! Downloaded {observer.file_count} files, Total file size = {beautify(observer.file_size//1024)}, Time Elapsed: {int(time.time() - observer.tik)} seconds.'
                )
                observer.stop()
                observer.join()

                event.set()
                monitor_thread.join()

        return wrapper

    return decorator

def get_import_exception_message(package_name: str, extra_deps: str) -> str:
    """Get import exception message.

    Args:
        package_name (str): Package name.

    Returns:
        str: Exception message.
    """
    return f'BYOD was installed without {package_name} support. ' + \
            f'To use {package_name} related packages with BYOD, run ' + \
            f'`pip install \'mosaicml-byod[{package_name}]\'`.'


def byod_now(strftime = False):
    now = (datetime.now(tz=pytz.utc)).astimezone(timezone('US/Pacific'))
    if strftime:
        return now.strftime("%Y-%m-%d-%H-%M-%S")
    return now

def byod_path(task_key: str, timestamp: str):
    return os.path.join(task_key, timestamp)

if __name__ == '__main__':
    hyper_params = {'A': [1, 2], 'B': ['bob', 'alice'], 'C': 5, 'D': [True, False]}

    assert (len(list(hyperparameter_combinations_generator(hyper_params))) == 8), 'sanity check failed'

    assert (len(list(hyperparameter_combinations_generator(hyper_params, take_first=['A']))) == 4), 'sanity check failed'

    print('Success')
