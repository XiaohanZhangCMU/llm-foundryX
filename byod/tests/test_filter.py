from byod import Filter
from pyspark.sql import SparkSession
from byod.task import Filter

# This test does not work on mac
# Because BlingFire does not have support for mac arm64.
# You can download https://github.com/microsoft/BlingFire/blob/master/nuget/lib/runtimes/osx-arm64/native/libblingfiretokdll.dylib
# And replace it with the installed one. But mac osx cannot open it still

def main():
    spark = SparkSession.builder.appName("FilterTest").getOrCreate()
    data = [("Lorem ipsum",), ("123456789",), ("This is a test",), ("Special characters: !@#$%^&*()_+",)]
    columns = ["text"]
    df = self.spark.createDataFrame(data, columns)

    task = Filter(df=self.df, task_key="filter_task", text_col="text")
    result_df = task.run()

    # Ensure the result DataFrame has the expected schema and values
    expected_data = [("Lorem ipsum",), ("123456789",), ("This is a test",)]
    expected_columns = ["text"]
    expected_result_df = self.spark.createDataFrame(expected_data, expected_columns)

    self.assertListEqual(result_df.columns, expected_result_df.columns)
    self.assertListEqual(result_df.collect(), expected_result_df.collect())

if __name__ == '__main__':
    main()

