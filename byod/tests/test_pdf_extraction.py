from byod import ByodTask, IngestionTask, TransformationTask
from byod import PdfExtract
from pyspark.sql import SparkSession

if __name__ == '__main__':

    task = PdfExtract(
        task_key="pdf_exract",
        out_root="/tmp/pdf_extract_output",
    )

    # You can update this list with the actual PDF files in your folder
    pdf_file_dataset = "tests/resources/pdf_dataset/"

    task.define_input(pdf_file_dataset)

    task.run()

