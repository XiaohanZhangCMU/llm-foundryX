# Databricks notebook source
# MAGIC %md # Debug Only
# MAGIC <br/>
# MAGIC
# MAGIC - The cell below is for debugging only.
# MAGIC - Comment it out in workflow production.
# MAGIC - Check if the values are set as task inputs.

# COMMAND ----------

#%pip uninstall -y mosaicml-byod
#dbutils.library.restartPython()

# COMMAND ----------

#%pip install /dbfs/xiaohan-test/mosaicml_byod-0.0.1-py3-none-any.whl
%pip install git+https://github.com/XiaohanZhangCMU/llm-foundryX.git@cpt_poc
dbutils.library.restartPython()

# COMMAND ----------

import logging
import math
import os
import tempfile
from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from typing import Iterable, List, Tuple, cast

import psutil
from composer.utils import (ObjectStore, maybe_create_object_store_from_uri,
                            parse_uri)
from streaming import MDSWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from llmfoundry.data import ConcatTokensDataset
from llmfoundry.utils.data_prep_utils import (DownloadingIterable,
                                              merge_shard_groups)

import py4j
import argparse

log = logging.getLogger(__name__)
DONE_FILENAME = '.text_to_mds_conversion_done'


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert text files into MDS format, optionally concatenating and tokenizing',
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='The folder to write output to',
    )
    parser.add_argument(
        '--input_folder',
        type=str,
        required=True,
        help='The folder with text files to convert to mds',
    )
    parser.add_argument(
        '--compression',
        type=str,
        default='zstd',
        help='The compression algorithm to use for MDS writing',
    )

    parser.add_argument(
        '--concat_tokens',
        type=int,
        help='Convert text to tokens and concatenate up to this many tokens',
    )

    parser.add_argument(
        '--tokenizer',
        type=str,
        help='The name of the tokenizer to use',
    )
    parser.add_argument(
        '--bos_text',
        type=str,
        required=False,
        default=None,
        help=
        'The text to prepend to each example to separate concatenated examples',
    )
    parser.add_argument(
        '--eos_text',
        type=str,
        required=False,
        default=None,
        help=
        'The text to append to each example to separate concatenated examples',
    )
    parser.add_argument(
        '--no_wrap',
        default=False,
        action='store_true',
        help=
        'Whether to let text examples wrap across multiple training examples',
    )
    parser.add_argument(
        '--processes',
        type=int,
        required=False,
        default=min(max(psutil.cpu_count() - 2, 1), 32),
        help=
        'The number of processes to use to download and convert the dataset',
    )
    parser.add_argument(
        '--reprocess',
        type=bool,
        required=False,
        default=False,
        help='If true, reprocess the input_folder to mds format. Otherwise, ' +
        'only reprocess upon changes to the input folder or dataset creation parameters.',
    )

    parsed = parser.parse_args()

    # Make sure we have needed concat options
    if (parsed.concat_tokens is not None and
            isinstance(parsed.concat_tokens, int) and parsed.tokenizer is None):
        parser.error(
            'When setting --concat_tokens, you must specify a --tokenizer')

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''
    return parsed


def get_object_names(input_folder: str) -> List[str]:
    """Get object names from a local or remote folder.

    Args:
        input_folder (str): local or remote folder path.
    """
    object_store = maybe_create_object_store_from_uri(input_folder)
    if object_store is not None:
        _, _, folder_prefix = parse_uri(input_folder)
        names = [
            name for name in object_store.list_objects(folder_prefix)
            if name.endswith('.txt')
        ]
    else:
        # input_folder is a local folder
        names = [
            text_file for dirpath, _, _ in os.walk(input_folder)
            for text_file in glob(os.path.join(dirpath, '*.txt'))
        ]
    # return names, sizes
    log.info(f'Found {len(names)} text files at {input_folder}')

    return names


def get_task_args(
    object_names: List[str],
    output_root: str,
    input_folder: str,
    n_groups: int,
    tokenizer_name: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    compression: str,
) -> Iterable:
    """Get download_and_convert arguments split across n_groups.

    Each group handles a portion of object_names.

    Args:
        object_names (List[str]): Names of objects to process
        output_root (str): Folder to write MDS shards to
        input_folder (str): Folder of text files to process
        n_groups (int): Number of groups to split the object names into
        tokenizer_name (str): Name of tokenizer to use
        concat_tokens (int): Concantenate up to this many tokens
        eos_text (str): Textend to append to each example to separate concatenated samples
        bos_text (str): Text to prepend to each example to separate concatenated samples
        no_wrap: (bool): Whether to let text examples wrap across multiple training examples
        compression (str): The compression algorithm to use for MDS writing
    """
    num_objects = len(object_names)
    objs_per_group = math.ceil(num_objects / n_groups)
    for group, i in enumerate(range(0, num_objects, objs_per_group)):
        output_subdir = os.path.join(output_root, str(group))
        yield (
            object_names[i:min(i + objs_per_group, num_objects)],
            output_subdir,
            input_folder,
            tokenizer_name,
            concat_tokens,
            eos_text,
            bos_text,
            no_wrap,
            compression,
        )


def download_and_convert_starargs(args: Tuple):
    """Helper function to call download_and_convert with star args.

    This helps us use download_and_convert with mutiprocessing.
    """
    return download_and_convert(*args)


def download_and_convert(
    file_names: List[str],
    output_folder: str,
    input_folder: str,
    tokenizer_name: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    compression: str,
):
    """Downloads and converts text fies to MDS format.

    Args:
        file_names (List[str]): Files to process
        output_folder (str): Folder to write MDS shards to
        input_folder (str): Folder of text files to process
        tokenizer_name (str): Name of tokenizer to use
        concat_tokens (int): Concantenate up to this many tokens
        eos_text (str): Textend to append to each example to separate concatenated samples
        bos_text (str): Text to prepend to each example to separate concatenated samples
        no_wrap: (bool): Whether to let text examples wrap across multiple training examples
        compression (str): The compression algorithm to use for MDS writing
    """
    object_store = maybe_create_object_store_from_uri(input_folder)

    # Download file_names
    with tempfile.TemporaryDirectory() as tmp_dir:
        downloading_iter = DownloadingIterable(object_names=file_names,
                                               output_folder=tmp_dir,
                                               object_store=object_store)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.model_max_length = 5000000000  # Hack to prevent warnings from HuggingFace

        # Use the ConcatTokensDataset from LLM-foundry to concatenate sequences of tokens up
        # to the maximum sequence length
        dataset = ConcatTokensDataset(
            hf_dataset=downloading_iter,
            max_length=concat_tokens,
            tokenizer=tokenizer,
            eos_text=eos_text,
            bos_text=bos_text,
            no_wrap=no_wrap,
        )

        columns = {'tokens': 'bytes'}

        log.info('Converting to MDS format...')
        with MDSWriter(out=output_folder,
                       columns=columns,
                       compression=compression) as out:
            for sample in tqdm(dataset):
                out.write(sample)


def is_remote_path(path: str) -> bool:
    """Checks whether a path is a remote path.

    Args:
        path (str): path to check
    """
    backend, bucket, _ = parse_uri(path)
    return backend != '' and bucket != ''


def is_already_processed(output_root: str, args_str: str,
                         object_names: List[str]) -> bool:
    """Determines whether a group of text files has already been processed.

    Checks the done fie at output root to determine this.

    Args:
        output_root (str): Output folder where a done file may exist
        args_str (str): String representation of the arguments
        object_names (List[str]): Names of objects to convert to MDS format
    """
    # Retrieve the done file contents
    output_object_store = maybe_create_object_store_from_uri(output_root)
    if output_object_store is not None:
        # Download and read the done file from the remote object store
        _, _, output_folder_prefix = parse_uri(output_root)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                done_file = os.path.join(tmp_dir, DONE_FILENAME)
                output_object_store.download_object(
                    os.path.join(output_folder_prefix, DONE_FILENAME),
                    done_file)
                with open(done_file) as df:
                    done_file_contents = df.read().splitlines()
        except FileNotFoundError:
            return False
    else:
        # Read the local done file
        done_file = os.path.join(output_root, DONE_FILENAME)
        if not os.path.isfile(done_file):
            return False
        with open(done_file) as df:
            done_file_contents = df.read().splitlines()
    # Compare the arguments
    prev_args_str = done_file_contents[0]
    if prev_args_str != args_str:
        return False

    # Compare file names
    prev_names = done_file_contents[1:]
    if len(prev_names) != len(object_names):
        return False
    for idx, prev_name in enumerate(prev_names):
        if object_names[idx] != prev_name:
            return False
    return True


def write_done_file(folder: str, args_str: str, object_names: List[str]):
    """Write a file to signify completion.

    This the done file includes the arguments to processing and
    a list of objects that were processed.

    Args:
        folder (str): Folder to write the done file to
        args_str (str): String representation of arguments
        object_names (List[str]): List of objects to convert to MDS format
    """
    with open(os.path.join(folder, DONE_FILENAME), 'w') as done_file:
        done_file.write('\n'.join([args_str] + object_names) + '\n')


def convert_text_to_mds(
    tokenizer_name: str,
    output_folder: str,
    input_folder: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    compression: str,
    processes: int,
    args_str: str,
    reprocess: bool,
):
    """Convert a folder of text files to MDS format.

    Args:
        tokenizer_name (str): Name of tokenizer to use
        output_folder (str): Folder to write MDS shards to
        input_folder (str): Folder of text files to process
        concat_tokens (int): Concantenate up to this many tokens
        eos_text (str): Textend to append to each example to separate concatenated samples
        bos_text (str): Text to prepend to each example to separate concatenated samples
        no_wrap: (bool): Whether to let text examples wrap across multiple training examples
        compression (str): The compression algorithm to use for MDS writing
        processes (int): The number of processes to use.
        args_str (str): String representation of the arguments
        reprocess (bool): Whether to always reprocess the given folder of text files
    """
    is_remote_output = is_remote_path(output_folder)

    object_names = get_object_names(input_folder)
    if len(object_names) == 0:
        raise ValueError(f'No text files were found at {input_folder}.')

    # Check if the text files in the bucket have already been processed.
    if not reprocess and is_already_processed(output_folder, args_str,
                                              object_names):
        log.info(
            f'Input folder {input_folder} is already processed at {output_folder} and '
            +
            'reprocess is set to False. Set reprocess to True if you would like to force reprocessing.'
        )
        return

    # Use a temporary local directory if the output is remote and there are more than 1 processes
    local_output_folder = tempfile.TemporaryDirectory(
    ).name if is_remote_output else output_folder

    if processes > 1:
        # Download and convert the text files in parallel
        args = get_task_args(object_names, local_output_folder, input_folder,
                             processes, tokenizer_name, concat_tokens, eos_text,
                             bos_text, no_wrap, compression)
        with ProcessPoolExecutor(max_workers=processes) as executor:
            list(executor.map(download_and_convert_starargs, args))

        # Merge the mds shards from each of the processes into a single folder
        merge_shard_groups(local_output_folder)
    else:
        download_and_convert(object_names, local_output_folder, input_folder,
                             tokenizer_name, concat_tokens, eos_text, bos_text,
                             no_wrap, compression)

    # Write a done file with the args and object names
    write_done_file(local_output_folder, args_str, object_names)

    if is_remote_output:
        # Upload the local output to the remote location
        output_object_store = cast(
            ObjectStore, maybe_create_object_store_from_uri(output_folder))
        _, _, output_folder_prefix = parse_uri(output_folder)
        files_to_upload = os.listdir(local_output_folder)

        for file in files_to_upload:
            assert not os.path.isdir(file)
            remote_path = os.path.join(output_folder_prefix, file)
            output_object_store.upload_object(
                remote_path, os.path.join(local_output_folder, file))


def _args_str(original_args: Namespace) -> str:
    """Create a string from the args to determine whether to reprocess.

    Args:
        original_args (Namespace): Arguments to main function.
    """
    # Take the arguments that influence the final result.
    # reprocess and max_mds_writer_workers are not taken.
    args = Namespace(
        tokenizer_name=original_args.tokenizer,
        output_folder=original_args.output_folder,
        input_folder=original_args.input_folder,
        concat_tokens=original_args.concat_tokens,
        eos_text=original_args.eos_text,
        bos_text=original_args.bos_text,
        no_wrap=original_args.no_wrap,
        compression=original_args.compression,
        processes=original_args.processes,
    )

    return str(args)

def widget_with_default(key, default=None):
    try:
        x = dbutils.widgets.get(key)
        return x
    except py4j.protocol.Py4JJavaError as e:
        if default is None:
            raise ValueError(f"{key} must be provided as no default value can be preset") from e
        if default == 'None':
            return None
        return default


#if __name__ == '__main__':
    #args = parse_args()

# COMMAND ----------

args  = argparse.Namespace()
args.tokenizer=widget_with_default('tokenizer')
args.output_folder=widget_with_default('output_folder')
args.input_folder=widget_with_default('input_folder')
args.concat_tokens=int(widget_with_default('concat_tokens')) # dabs cannot handle number param
args.eos_text=widget_with_default('eos_text', '')
args.bos_text=widget_with_default('bos_text', '')
args.no_wrap=widget_with_default('no_wrap', False)
args.compression=widget_with_default('compression', 'zstd')
args.processes=int(widget_with_default('processes', min(max(psutil.cpu_count() - 2, 1), 32))) # dabs cannot handle number param
args.reprocess=widget_with_default('reprocess', False)

args_str=_args_str(args)

print('args_str = ', args_str)

# COMMAND ----------

!rm -rf {args.output_folder}

# COMMAND ----------
convert_text_to_mds(tokenizer_name=args.tokenizer,
                    output_folder=args.output_folder,
                    input_folder=args.input_folder,
                    concat_tokens=args.concat_tokens,
                    eos_text=args.eos_text,
                    bos_text=args.bos_text,
                    no_wrap=args.no_wrap,
                    compression=args.compression,
                    processes=args.processes,
                    reprocess=args.reprocess,
                    args_str=_args_str(args))


# COMMAND ----------

from torch.utils.data import DataLoader
from streaming import StreamingDataset
import numpy as np

total_tokens = 0
dataset=StreamingDataset(local=args.output_folder)
dataloader = DataLoader(dataset)
sample = next(iter(dataloader))
b = np.asarray(sample['tokens']).tobytes()
token_ids = np.frombuffer(b, dtype=np.int64)
n_token_per_sample = len(token_ids)
print('total_tokens = ', n_token_per_sample * dataset.num_samples)
