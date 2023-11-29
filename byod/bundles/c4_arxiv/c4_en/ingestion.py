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
from byod import HFIngestion

prefix = 'allenai/'
submixes = ['c4']
allow_patterns=['en/c4-train.0000*']

token='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',
hf_ingest = HFIngestion('ingest', root, token, prefix, None, submixes, allow_patterns)
hf_ingest.run()

task_output = hf_ingest.get_task_output()

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
