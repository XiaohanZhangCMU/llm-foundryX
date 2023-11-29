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

root        = dbutils.jobs.taskValues.get(taskKey = "Ingest_C4_from_HuggingFace", key = "root",        default = '', debugValue = '/Volumes/datasets/default/byod/c4/')
input_path  = dbutils.jobs.taskValues.get(taskKey = "Ingest_C4_from_HuggingFace", key = "task_output", default = '', debugValue = '/Volumes/datasets/default/byod/c4/ingest/2023-11-14-11-14-47/')

print('root = ', root)
print('input_path = ', input_path)

# COMMAND ----------

# MAGIC %md # Auto Spark Read

# COMMAND ----------

text_key = 'text' # The original text field column name in the dataframe
df = spark.sql("SELECT * FROM read_files('%s')" % input_path).select(text_key).withColumnRenamed(text_key, 'text')

# COMMAND ----------

# MAGIC %md # Filter

# COMMAND ----------

from byod import Filter

filteration = Filter(df,
                     task_key="filter",
                     text_col="text",
                     min_len = 1,
                     max_len = 50000,
                     max_char_len = 500_000,
                     alpha_ratio = 0.1,
                     max_freq_ratio = 0.0,
                     )
df = filteration.run()

# COMMAND ----------

# MAGIC %md # Tokenize+MDS

# COMMAND ----------

from byod import Tokenize

print(df.count())

tokenization = Tokenize(df=df, task_key='tokenize', out_root=root, tokenizer='tiktoken', seqlen=1024)

if True:
    tokenization.run()

# COMMAND ----------

# MAGIC %md # Shard Integrity Check

# COMMAND ----------

task_output = tokenization.get_task_output()
print(f'task_output = {task_output}')
dbutils.jobs.taskValues.set(key = "task_output", value = task_output)

# COMMAND ----------

!ls {task_output} | wc -l

# COMMAND ----------

!ls {task_output}/index.json
