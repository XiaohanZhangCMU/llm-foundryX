# Databricks notebook source
# MAGIC %md # Not Implemented Yet
# MAGIC <br/>
# MAGIC
# MAGIC What items to check?
# MAGIC -
# MAGIC -
# MAGIC -

# COMMAND ----------

arxiv_mds_root = dbutils.jobs.taskValues.get(taskKey = "Arxiv_Filter_Tokenization_MDSWrite", key = "task_output", default = '', debugValue = 'SOMETHING IS WRONG - ARXIV')
c4_en_mds_root = dbutils.jobs.taskValues.get(taskKey = "C4_Filter_Tokenization_MDSWrite", key = "task_output", default = '', debugValue = 'SOMETHING IS WRONG - C4_EN')

# COMMAND ----------

print(arxiv_mds_root)
print(c4_en_mds_root)

# COMMAND ----------

!ls {arxiv_mds_root} | wc -l

# COMMAND ----------

!ls {arxiv_mds_root}/index.json

# COMMAND ----------

 !ls {c4_en_mds_root} | wc -l

# COMMAND ----------

!ls {c4_en_mds_root}/index.json
