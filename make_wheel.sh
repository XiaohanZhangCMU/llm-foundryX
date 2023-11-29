#!/bin/sh
pip install -e .

python setup.py bdist_wheel

databricks fs cp --overwrite ./dist/llm_foundry-0.4.0-py3-none-any.whl dbfs:/xiaohan-test/

