.PHONY: all install build deploy

all: install build deploy

install:
	pip install -e .

build: install
	python setup.py bdist_wheel

deploy: build
	databricks fs cp --overwrite ./dist/mosaicml_byod-0.0.1-py3-none-any.whl dbfs:/xiaohan-test/

