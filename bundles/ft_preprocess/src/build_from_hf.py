# Databricks notebook source
# MAGIC %md # Debug Only
# MAGIC <br/>
# MAGIC
# MAGIC - The cell below is for debugging only.
# MAGIC - Comment it out in workflow production.
# MAGIC - Check if the values are set as task inputs.

# COMMAND ----------

# %pip install /dbfs/xiaohan-test/mosaicml_byod-0.0.1-py3-none-any.whl
%pip install /dbfs/xiaohan-test/llm_foundry-0.3.0-py3-none-any.whl
dbutils.library.restartPython()
# COMMAND ----------

# MAGIC %md # Ingestion Task Run
# MAGIC <br/>
# MAGIC
# MAGIC - If debug dataset is not touched, skip this run as it is time consuming.
# MAGIC - We could turn it off in regression testing so ingest always run and be tested.

# COMMAND ----------

root = dbutils.widgets.get('root')
dataset_name = dbutils.widgets.get('hf_name')
split = dbutils.widgets.get('split')
proto_preprocessing_fn = dbutils.widgets.get('preprocessing_fn')
max_seq_len = dbutils.widgets.get('max_seq_len')
tokenizer_name = dbutils.widgets.get('tokenizer')

max_seq_len = int(max_seq_len[:-1]) # dabs cannot handle number param

print('root = ', root)
print('dataset_name = ', dataset_name)
print('split = ', split)
print('proto_preprocessing_fn = ', proto_preprocessing_fn)
print('max_seq_len = ', max_seq_len)
print('tokenizer_name = ', tokenizer_name)

# COMMAND ----------

import os
from llmfoundry.utils import build_tokenizer
from llmfoundry.data.finetuning import DatasetConstructor
import datasets as hf_datasets
from omegaconf import DictConfig
from typing import Any, Callable, Dict, List, Optional, Union
from transformers import PreTrainedTokenizerBase
import warnings

def _tokenize_formatted_example(
        example: Dict[str, Any],
        tokenizer: PreTrainedTokenizerBase) -> Dict[str, List[int]]:
    if ('prompt' not in example) or ('response' not in example):
        raise KeyError(
            'Unable to tokenize example because it has not been properly formatted. ' +\
            '"prompt" and "response" are required keys but at least one was missing ' +\
            f'from {example=}.'
        )
    if not isinstance(example['prompt'], str):
        raise TypeError(
            f'Unable to tokenize example because "prompt" was not a string. {example=}'
        )
    if not isinstance(example['response'], str):
        raise TypeError(
            f'Unable to tokenize example because "response" was not a string. {example=}'
        )
    return tokenizer(text=example['prompt'], text_target=example['response'])


dataset = hf_datasets.load_dataset(dataset_name, split=split)

tokenizer_kwargs = {'model_max_length': max_seq_len}
tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

dataset_constructor = DatasetConstructor()

if isinstance(proto_preprocessing_fn, dict) or isinstance(
        proto_preprocessing_fn, DictConfig):
    preprocessing_fn = dataset_constructor.get_preprocessing_fn_from_dict(
        proto_preprocessing_fn)
else:
    preprocessing_fn = dataset_constructor.get_preprocessing_fn_from_str(
        proto_preprocessing_fn, dataset_name)

def dataset_mapper(example: Dict):
    if preprocessing_fn is not None:
        example = preprocessing_fn(example)
    return _tokenize_formatted_example(example, tokenizer)

detected_cpu_count = os.cpu_count() or 1
detected_cpus_with_margin = detected_cpu_count - 8
num_cpus_to_use = max(1, detected_cpus_with_margin)

columns_to_remove = list(dataset[0].keys())
tokenized_dataset = dataset.map(
    dataset_mapper,
    batched=False,
    remove_columns=columns_to_remove,
    num_proc=num_cpus_to_use,
    desc='Tokenizing dataset',
)

pad_token_id = tokenizer.pad_token_id

def filter_long_or_empty_examples(example: Dict) -> bool:
    less_than_max_seq_len = len(example['input_ids']) < max_seq_len
    non_empty_input = len(example['input_ids']) > 0
    non_empty_labels = len(example['labels']) > 0
    non_padding_response = any(
        token_id != pad_token_id for token_id in example['labels'])
    return (less_than_max_seq_len and non_empty_input and
            non_empty_labels and non_padding_response)

filtered_dataset = tokenized_dataset.filter(
    filter_long_or_empty_examples,
    num_proc=num_cpus_to_use,
    desc='Filtering out long prompts',
)

examples_removed = len(tokenized_dataset) - len(filtered_dataset)
if examples_removed > 0:
    warnings.warn(
        f'Dropped {examples_removed} examples where the prompt was longer than {max_seq_len}, '
        +
        'the prompt or response was empty, or the response was all padding tokens.'
    )

# COMMAND ----------

# MAGIC %md # Sanity Checks
# MAGIC <br/>
# MAGIC Not Implemented Yet

# COMMAND ----------

print(f'task_output = {task_output}')
dbutils.jobs.taskValues.set(key = "task_output", value = task_output)
dbutils.jobs.taskValues.set(key = "root", value = root)

# COMMAND ----------

!ls {task_output} | wc -l
