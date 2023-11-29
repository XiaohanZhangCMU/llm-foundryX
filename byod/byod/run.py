import subprocess
import argparse
import yaml
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from byod import ByodTask, ByodJob

def run_byod_with_jobAPI(cfg: DictConfig):
    byod_job = ByodJob(job_id = 'byod',
                       cfg_or_yaml_path = cfg,
                       task_type = 'notebook_task')
    byod_job.submit_job()

"""
An example of cfg for run_byod:

    cfg = om.create({
        'tasks': [
            {
                'task_key': 'Ingest_C4_from_HuggingFace',
                'notebook_task': {
                    'notebook_path': '/Users/xiaohan.zhang/Codes/byod/bundles/c4_arxiv/c4_en/ingestion.py',
                    'base_parameters': {
                        'root': '/Volumes/datasets/default/byod/c4/',
                    },
                },
                'existing_cluster_id': '1116-234530-6seh113n',
                'libraries': {
                   'whl': 'dbfs:/xiaohan-test/mosaicml_byod-0.0.1-py3-none-any.whl'
                }
            },
            {
                'task_key': 'C4_Filter_Tokenization_MDSWrite',
                'depends_on': {
                    'task_key': 'Ingest_C4_from_HuggingFace'
                },
                'notebook_task': {
                    'notebook_path': '/Users/xiaohan.zhang/Codes/byod/bundles/c4_arxiv/c4_en/ingestion.py',
                    'base_parameters': {
                        'root': '/Volumes/datasets/default/byod/c4/',
                    },
                },
                'existing_cluster_id': '1116-234530-6seh113n',
                'libraries': {
                   'whl': 'dbfs:/xiaohan-test/mosaicml_byod-0.0.1-py3-none-any.whl'
                }
            },
            {
                'task_key': 'Ingest_GCS_Arxiv_Extract_PDF_Text',
                'notebook_task': {
                    'notebook_path': '/Users/xiaohan.zhang/Codes/byod/bundles/c4_arxiv/arxiv/ingestion.py',
                    'base_parameters': {
                        'root': '/Volumes/datasets/default/byod/c4/',
                    },
                },
                'existing_cluster_id': '1116-234530-6seh113n',
                'libraries': {
                   'whl': 'dbfs:/xiaohan-test/mosaicml_byod-0.0.1-py3-none-any.whl'
                }
            },
            {
                'task_key': 'Arxiv_Filter_Tokenization_MDSWrite',
                'depends_on': {
                    'task_key': 'Ingest_GCS_Arxiv_Extract_PDF_Text'
                },
                'notebook_task': {
                    'notebook_path': '/Users/xiaohan.zhang/Codes/byod/bundles/c4_arxiv/arxiv/processing.py',
                    'base_parameters': {
                      'root': '/Volumes/datasets/default/byod/c4/',
                    },
                },
                'existing_cluster_id': '1116-234530-6seh113n',
                'libraries': {
                   'whl': 'dbfs:/xiaohan-test/mosaicml_byod-0.0.1-py3-none-any.whl'
                }
            },
            {
                'task_key': 'mds_shard_integrity_check',
                'depends_on': {
                    'task_key': 'Arxiv_Filter_Tokenization_MDSWrite',
                    'task_key': 'C4_Filter_Tokenization_MDSWrite'
                },
                'notebook_task': {
                    'notebook_path': '/Users/xiaohan.zhang/Codes/byod/bundles/c4_arxiv/c4_en/mds_shard_integrity_checks.py',
                    'base_parameters': {
                        'root': '/Volumes/datasets/default/byod/c4/',
                    },
                },
                'existing_cluster_id': '1116-234530-6seh113n',
                'libraries': {
                   'whl': 'dbfs:/xiaohan-test/mosaicml_byod-0.0.1-py3-none-any.whl'
                }
            }
        ],
    })

"""

def set_param_in_yml(yaml_file_path, job_name, env, tokenizer, concat_tokens, output_folder, input_folder, eos_text):
    with open(yaml_file_path, 'r') as file:
        # Load the YAML file
        yml_data = yaml.safe_load(file)

    # Navigate to the specific job and its base_parameters
    job_data = yml_data.get('resources', {}).get('jobs', {}).get(job_name, {})
    base_params = job_data.get('tasks', [{}])[0].get('notebook_task', {}).get('base_parameters', {})

    # Update the parameters
    base_params['tokenizer'] = tokenizer
    base_params['concat_tokens'] = concat_tokens
    base_params['output_folder'] = output_folder
    base_params['input_folder'] = input_folder
    if eos_text:
        base_params['eos_text'] = eos_text

    with open(yaml_file_path, 'w') as file:
        # Write the updated data back to the YAML file
        yaml.dump(yml_data, file)

# Example usage
yaml_file_path = 'path/to/your/yml/file.yml'
job_name = 'dev_xiaohan_zhang_ft_preprocess'
set_param_in_yml(yaml_file_path, job_name, 'dev', 'my_tokenizer', True, '/path/to/output', '/path/to/input', 'EOS')


def run_byod_with_DABs(job_name, env, tokenizer, concat_tokens, output_folder, input_folder, eos_text ='<|endoftext|>'):
    """ Fork a subprocess to run DABs. Need to be called within a bundle.
    """
    command1 = f"databricks bundle deploy -t {env} {job_name}"
    command2 = f"databricks bundle run -t {env} {job_name}"

    result1 = subprocess.run(command1, shell=True, capture_output=True, text=True)
    result2 = subprocess.run(command2, shell=True, capture_output=True, text=True)

    if result1.returncode != 0:
        raise RuntimeError(f"Command 1 failed with return code {result1.returncode}. "
                           f"Output: {result1.stdout}\nError: {result1.stderr}")

    if result2.returncode != 0:
        raise RuntimeError(f"Command 2 failed with return code {result2.returncode}. "
                           f"Output: {result2.stdout}\nError: {result2.stderr}")

    return result1.stdout, result1.stderr, result2.stdout, result2.stderr

"""
    Example
    python byod/run.py --job dev_xiaohan_zhang_BYOD_POC_2010 --env dev

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute Databricks bundles shell commands")
    parser.add_argument("--job", type=str, help="Name of the job")
    parser.add_argument("--env", type=str, help="Environment name")
    args = parser.parse_args()

    output1, error1, output2, error2 = run_byod_with_DABs(args.job, args.env)
    print("Command 1 StdOut:")
    print(output1)
    print("Command 1 StdErr:")
    print(error1)
    print("Command 2 StdOut:")
    print(output2)
    print("Command 2 StdErr:")
    print(error2)



