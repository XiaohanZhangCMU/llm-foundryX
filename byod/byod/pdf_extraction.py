import nltk
from enum import Enum
from unstructured.partition.pdf import partition_pdf
import os
import glob
import json
import time
import fitz  # PyMuPDF
import multiprocessing
import pandas as pd
from collections import OrderedDict
from typing import Optional, Tuple, Union, Any
from functools import partial
import shutil
import numpy as np
from datetime import datetime
from byod.utils import TaskStatus, get_import_exception_message

try:
    from pyspark.sql.types import StringType, StructType, StructField, IntegerType
except ImportError as e:
    e.msg = get_import_exception_message(e.name, 'spark')
    raise e

from byod.task import IngestionTask

MAX_SINGLE_FILE_PAGES = 18

class Code(Enum):
    CANNOT_OPEN = 1
    CANNOT_EXTRACT = 2
    EXTRACT_TIMEOUT = 3

def get_number_of_pages(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        num_pages = pdf_document.page_count
        pdf_document.close()
        return (pdf_path, num_pages)
    except Exception as e:
        return (pdf_path, -1)

SINGLE_FILE_PROC_TIMEOUT = 120 # seconds

class PdfExtract(IngestionTask):
    def __init__(self,
                 task_key: str,
                 out_root: str,
                 *args: Any,
                 **kwargs: Any,
                 ):
        super().__init__(task_key, out_root, *args, **kwargs)

        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

        pass

    def define_input(self, dataset_folder, *args, **kwargs) -> None:
        self.pdf_files = glob.glob(os.path.join(dataset_folder, "*.pdf"))

    def submit_spark_jobs(self, pdf_with_meta):
        def extract_text_from_pdf(pdf_path):
            try:
                try:
                    elements = partition_pdf(pdf_path, strategy="fast")
                except:
                    elements = partition_pdf(pdf_path, strategy="hi_res")

                d = [  e.to_dict() for e in elements ]
                pdf_filename = os.path.basename(pdf_path)[:-3]+'json'
                with open(os.path.join(global_output_dir, pdf_filename), 'w') as f:
                    json.dump(d, f)
                return 'OK'
            except:
                print('failed to process ' + pdf_path)
                return 'Failed'
        def process_partition(partition):
            processed_pdfs = []
            for pdf in partition:
                records = pdf.to_dict('records')
                for sample in records:
                    file_paths = sample['pdf_path'].split(';')
                    for file_path in file_paths:
                        if len(file_path) > 1:
                            res = extract_text_from_pdf(file_path)
                            processed_pdfs.append(res)
            yield pd.DataFrame({'pdf_path': pd.Series(processed_pdfs)})

        data = [ (0, p) for p in pdf_with_meta]
        schema = StructType([StructField("num_pages", IntegerType(), True),
                            StructField("pdf_path", StringType(), True)])
        df = self.spark.createDataFrame(data=data, schema=schema)

        result_schema = StructType([
            StructField('pdf_path', StringType(), False)
        ])
        global_output_dir = self.task_output
        res = df.mapInPandas(func=process_partition, schema=result_schema).collect()
        return res

    def run(self):

        with multiprocessing.Pool(self.n_driver_cores) as pool:
            results = pool.map(get_number_of_pages, self.pdf_files)

        page_sizes = dict(results)
        n = sum([ v==-1 for k, v in page_sizes.items()])
        for k, v in page_sizes.items():
            if v == -1:
                print(k)
        total_pages = sum([v if v != -1 else 0 for k, v in page_sizes.items()])
        global_average = MAX_SINGLE_FILE_PAGES # np.median(list(page_sizes.values())) # int(total_pages / num_partitions)

        print(f"Get num of pages for {len(page_sizes)} pdf files")
        print(f"{n} entries have number of pages -1")
        print(f"global average = {global_average}, total_pages = {total_pages}")
        pdf_num_pages = sorted([(pdf, page_num) for pdf, page_num in page_sizes.items()], key=lambda x: x[1])

        num_partitions = self.total_cores
        mesh = []
        start = 0
        end = num_partitions
        while end < len(self.pdf_files):
            mesh.append(pdf_num_pages[start:end])
            start = end
            end = start + num_partitions

        last_row = pdf_num_pages[start:] + [None] * (num_partitions - len(pdf_num_pages[start:]))
        mesh.append(last_row)

        meshT = [[''] for _ in range(len(mesh[0])) ]
        for c in range(len(mesh[0])):
            vals = [ mesh[r][c][0]  if mesh[r][c] is not None else '' for r in range(len(mesh))]
            meshT[c] = ';'.join(vals)

        self.submit_spark_jobs(meshT)

        return TaskStatus.SUCCESS


