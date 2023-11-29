# Copyright 2023 MosaicML byod authors
# SPDX-License-Identifier: Apache-2.0

"""Base class for all BYOD tasks"""

import os
import logging
import time
from typing import Any, Optional, Union, Dict, Tuple, List
import datasets as hf_datasets
import pandas as pd
import multiprocessing
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from abc import ABC, abstractmethod
try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service import jobs
    from databricks.sdk.service.jobs import Task
except ImportError as e:
    e.msg = get_import_exception_message(e.name, 'databricks')  # pyright: ignore
    raise e

from byod.utils import (TaskStatus,
                        get_import_exception_message,
                        byod_now,
                        byod_path)

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.dataframe import DataFrame as SparkDataFrame
except ImportError as e:
    e.msg = get_import_exception_message(e.name, 'spark')
    raise e

try:
    from dask.dataframe import DataFrame as DaskDataFrame
except ImportError as e:
    e.msg = get_import_exception_message(e.name, extra_deps='dask') # pyright: ignore
    raise e

logger = logging.getLogger(__name__)

class ByodTask(Task, ABC):
    def __init__(self,
                 task_key: str,
                 *args: Any,
                 **kwargs: Any,
                 ):
        super().__init__(task_key, *args, **kwargs)
        """Initialize a task.

           Args:
               task_key (str): an ID for databricks.sdk.Task
        """
        self.task_key = task_key
        self.start_time = byod_now()
        self.set_defaults()

    def set_defaults(self):

        # Spark Settings

        self.spark = SparkSession.builder.getOrCreate()  # pyright: ignore
        sjc = self.spark._jsc.sc()
        self.n_workers = len([executor.host() for executor in sjc.statusTracker().getExecutorInfos() ]) - 1
        self.n_driver_cores = multiprocessing.cpu_count()

        # Assume driver and executors are of the same instance
        # Need a better way since the assumption is not always true
        self.total_cores = (self.n_workers + 1) * self.n_driver_cores

        # Set spark configuration to prevent OOMs
        self.spark.conf.set('spark.sql.execution.arrow.maxRecordsPerBatch', 2000)

        # Use timestamp to differentiate runs of the same task
        self.timestamp = byod_now(strftime=True)

    def get_task_key(self):
        return self.task_key

    def get_task_output(self):
        topt = None
        try:
            topt = self.task_output
        except NameError:
            return None
        else:
            return topt

    @abstractmethod
    def run(self, *args, **kwargs):
        pass


class IngestionTask(ByodTask):
    def __init__(self,
                 task_key: str,
                 out_root: str,
                 *args: Any,
                 **kwargs: Any,
                 ):
        super().__init__(task_key, *args, **kwargs)
        self.out_root = out_root
        # Set the default output directory
        self.task_output = os.path.join(self.out_root, byod_path(self.task_key, self.timestamp))
        self.setup_task()

    def setup_task(self, *args, **kwargs) -> None:
        os.makedirs(self.task_output, exist_ok=True)
        logger.warning(f"task_output is set to {self.task_output}")


class TransformationTask(ByodTask):
    def __init__(self,
                 df: Union[SparkDataFrame, DaskDataFrame],
                 task_key: str,
                 *args: Any,
                 **kwargs: Any,
                 ):
        super().__init__(task_key, *args, **kwargs)
        self.df = df

    def sanity_checks(self):
        if self.df is None:
            raise ValueError("TransformationTask needs an non-null dataframe")

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

class MaterializeTask(TransformationTask):
    def __init__(self,
                 df: Union[SparkDataFrame, DaskDataFrame],
                 task_key: str,
                 out_root: str,
                 *args: Any,
                 **kwargs: Any,
                 ):
        super().__init__(df, task_key, *args, **kwargs)
        self.df = df
        self.out_root = out_root
        self.task_output = os.path.join(self.out_root, byod_path(self.task_key, self.timestamp))

        self.setup_task()

    def setup_task(self, *args, **kwargs) -> None:
        os.makedirs(self.task_output, exist_ok=True)
        logger.warning(f"task_output is set to {self.task_output}")


class ByodJob:
    def __init__(self,
                 job_id: str,
                 cfg_or_yaml_path: Union[DictConfig, str],
                 task_type = 'notebook_task'):

        self.job_id = job_id

        assert task_type == 'notebook_task', f"{task_type} is not supported yet. Use notebook_task instead"

        if isinstance(cfg_or_yaml_path, str):
            yaml_path = cfg_or_yaml_path
            with open(yaml_path) as f:
                cfg = om.load(f)
            om.resolve(cfg)
        elif isinstance(cfg_or_yaml_path, DictConfig):
            cfg = cfg_or_yaml_path
        else:
            raise ValueError("ByodJob requires DictConfig to continue")

        self.client = self._create_workspace_client()

        self.byod_tasks = []

        self._create_job(cfg)

    def run_sanity_checks(self):
        # Ensure cluster is running. If the cluster is not running, start it. Timeout = 20min

        assert len(self.byod_tasks)>0, "Task list is empty"

    def _create_workspace_client(self):
        try:
            return WorkspaceClient()
        except RuntimeError as e:
            e.msg = "databricks-sdk authentication failed for WorkspaceClient()"
            raise e

    def _create_job(self, cfg):

        for t in cfg.tasks:
            nb_task = t.get('notebook_task')
            cluster_id = t.get('existing_cluster_id')
            self.client.clusters.ensure_cluster_is_running(cluster_id)
            bt = jobs.Task(task_key = t.get("task_key"),
                           description = t.get('task_key'),
                           existing_cluster_id = cluster_id,
                           notebook_task = jobs.NotebookTask(notebook_path = str(nb_task.get('notebook_path')),
                                                             base_parameters = dict(nb_task.get('base_parameters'))))

            print('------------------------------------------------------')

            print(t.get('existing_cluster_id'))
            print(t.get("task_key"))
            print(nb_task.get('notebook_path'))
            print(nb_task.get('base_parameters'))
            print(type(nb_task.get('base_parameters')))
            print(nb_task.get('source'))

            print('------------------------------------------------------')

            self.byod_tasks.append(bt)

        #try:
        # Run sanity checks before submitting the job
        self.run_sanity_checks()

        self.created_job = self.client.jobs.create(name=self.job_id,
                                                   tasks=self.byod_tasks)
        print("Job created successfully!")
        #except Exception as e:
        #    self.handle_error(e)

    def submit_job(self):
        try:
            self.client.jobs.run_now(job_id=self.created_job.job_id)
            print("Job submitted successfully!")
        except Exception as e:
            self.handle_error(e)

    def handle_error(self, error):
        print(error)
        raise NotImplementedError



