# Databricks notebook source
# MAGIC %md # Debug Only
# MAGIC <br/>
# MAGIC
# MAGIC - The cell below is for debugging only.
# MAGIC - Comment it out in workflow production.
# MAGIC - Check if the values are set as task inputs.

# COMMAND ----------

# %pip install /dbfs/xiaohan-test/mosaicml_byod-0.0.1-py3-none-any.whl
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md # Ingestion Task Run
# MAGIC <br/>
# MAGIC
# MAGIC - If debug dataset is not touched, skip this run as it is time consuming.
# MAGIC - We could turn it off in regression testing so ingest always run and be tested.

# COMMAND ----------

root = dbutils.widgets.get('root')
print('root = ', root)

# COMMAND ----------

import os
from byod import PdfExtract
from byod.utils import byod_now, byod_path
import shutil

# Mount
mount_name = byod_path('dataset_mount', byod_now(strftime=True))
pdf_dataset_directory = byod_path('pdf_dataset', byod_now(strftime=True))
bucket_name = "arxiv-dataset"
try:
    dbutils.fs.mount("gs://%s" % bucket_name, "/mnt/%s" % mount_name)
except:
    print(f"{mount_name} is already mounted to /mnt/")

# Copy
if not os.path.exists(pdf_dataset_directory):
    os.makedirs(pdf_dataset_directory)

src_directory = f'/dbfs/mnt/{mount_name}/arxiv/arxiv/pdf/0704'
file_list = list(os.listdir(src_directory))
for f in file_list[:5]:
    shutil.copyfile(os.path.join(src_directory, f), os.path.join(pdf_dataset_directory, f))

# Run pdf extraction
pdf_task= PdfExtract(task_key="pdf_exract", out_root=root)
pdf_task.define_input(pdf_dataset_directory)
pdf_task.run()

task_output = pdf_task.get_task_output()

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
