# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utility for HF datasets Ingestion."""

# Improvements on snapshot_download:
# 1. Enable resume = True. retry when bad network happens
# 2. Disable progress_bar to prevent browser/terminal crash
# 3. Add a monitor to print out file stats every 15 seconds

import os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars
from byod.utils import monitor_directory_changes
from byod.task import IngestionTask
from streaming.base.util import retry
from typing import Union, Dict, Iterator, List, Tuple, Optional, Any

@monitor_directory_changes()
@retry([Exception, RuntimeError, TimeoutError], num_attempts=10, initial_backoff=10)
def hf_snapshot(repo_id: str,
                local_dir: str,
                revision: Optional[str],
                max_workers: int,
                token: str,
                allow_patterns: Optional[List[str]] = None):
    """API call to HF snapshot_download.

    which internally use hf_hub_download
    """
    print(
        f'Now start to download {repo_id} to {local_dir}, with allow_patterns = {allow_patterns}')
    output = snapshot_download(repo_id,
                               repo_type='dataset',
                               local_dir=local_dir,
                               local_dir_use_symlinks=False,
                               max_workers=max_workers,
                               resume_download=True,
                               token=token,
                               revision=revision,
                               allow_patterns=allow_patterns)
    return output

class HFIngestion(IngestionTask):
    def __init__(self,
                 task_key: str,
                 out_root: str,
                 token: str = 'MY_HFTOKEN',
                 prefix: str = '',
                 revision: Optional[str] = '',
                 submixes: Optional[List[str]] = None,
                 allow_patterns: Optional[List[str]] = None,
                 *args: Any,
                 **kwargs: Any,
                 ):
        super().__init__(task_key, out_root, *args, **kwargs)
        """Disable progress bar and call hf_snapshot.

        Args:
            prefix (str): HF namespace, allenai for example.
            submixes (List): a list of repos within HF namespace, c4 for example.
            revision (str): brach or release verion
            token (str): HF access toekn.
            allow_patterns (List): only files matching the pattern will be download. E.g., "en/*" along with allenai/c4 means to download allenai/c4/en folder only.
        """

        disable_progress_bars()
        self.prefix = prefix
        self.token = token
        self.submixes = submixes
        self.revision = revision
        self.allow_patterns = ['*'] if not allow_patterns else allow_patterns

    def run(self) -> None:

        for submix in self.submixes:
            repo_id = os.path.join(self.prefix, submix)
            local_dir = os.path.join(self.task_output, submix)

            _ = hf_snapshot(
                repo_id,
                local_dir,
                self.revision,
                self.n_driver_cores,
                self.token,
                self.allow_patterns,
            )

if __name__ == '__main__':
    #download_hf_dataset(local_cache_directory="/tmp/xiaohan/cifar10_1233", prefix="", submixes=["cifar10"], max_workers=32)
    prefix = '' # 'allenai/'
    submixes =  ['arxiv_dataset'] # ['wikitext'] # 'c4'

    download_hf_dataset(local_cache_directory='/tmp/xiaohan/arxiv_1316',
                        prefix=prefix,
                        submixes=submixes,
                        allow_patterns= ['*'], # ['en/*'],
                        max_workers=4,
                        token='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'
                       )  # 32 seems to be a sweet point, beyond 32 downloading is not smooth
