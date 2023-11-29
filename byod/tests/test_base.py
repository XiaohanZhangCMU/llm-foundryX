from byod import ByodTask, IngestionTask, TransformationTask, MaterializeTask
from pyspark.sql import SparkSession

# Sample Spark session
spark = SparkSession.builder.appName("ByodFramework").getOrCreate()

# Sample IngestionTask
class ConcreteIngestionTask(IngestionTask):
    def run(self) -> None:
        print(f"Concrete Ingesting data for task {self.task_key} to {self.task_output}")

ingestion_task = ConcreteIngestionTask(task_key="ingestion", out_root="ingestion_output")
ingestion_task.run()

# Sample DataFrame for TransformTask
data = [("John", 25), ("Alice", 30), ("Bob", 22)]
columns = ["name", "age"]
df = spark.createDataFrame(data, columns)

# Sample TransformTask
class ConcreteTransformationTask(TransformationTask):
    def run(self) -> None:
        print(f"Concrete Transforming data for task {self.task_key}")

transform_task = ConcreteTransformationTask(df, task_key="transformation")
transform_task.run()


# Sample DataFrame for MaterializeTask
df = spark.createDataFrame(data, columns)

# Sample TransformTask
class ConcreteMaterializeTask(MaterializeTask):
    def run(self) -> None:
        print(f"Concrete Transforming data for task {self.task_key} to {self.task_output}")

materialize_task = ConcreteMaterializeTask(df, task_key="materialization", out_root="/tmp/A/B/C/")
materialize_task.run()

assert("/tmp/A/B/C/" in materialize_task.get_task_output())
print('task_output = ', materialize_task.get_task_output())

# Stop Spark session
spark.stop()
