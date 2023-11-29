from byod import Tokenize
import pandas as pd
from dask.dataframe import from_pandas
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ByodFramework").getOrCreate()
data = [("John", 25),
        ("Alice", 30),
        ("Bob", 22),
        ("Cod", 33)]
columns = ["text", "age"]

#spark_df = spark.createDataFrame(data, columns)
pdf = pd.DataFrame(data, columns=columns)
dask_df = from_pandas(pdf, npartitions=2)
#df = spark.sql("SELECT * FROM read_files('%s')" % input_path).select(text_key).withColumnRenamed(text_key, 'text').repartition(self.n_repartitions) # Non determinsitc loading of df so not deterministic still

#tm = Tokenize(spark_df, 'tokenize_task', '/tmp/tokenize_test/', 'tiktoken', 1024)
#tm.run()

tm = Tokenize(dask_df, 'tokenize_task', '/tmp/tokenize_test/', 'tiktoken', 1024)
tm.run()

#tm = Tokenize(df, 'tokenize_task', 'gpt-neox-20b', 1024, '/tmp/tokenize_test/', )
#tm.run()

