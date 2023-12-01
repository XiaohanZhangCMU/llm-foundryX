import subprocess
import argparse
import yaml
import os
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

def run(job_name, env):

    command1 = f"databricks bundle deploy -t {env} {job_name}"
    command2 = f"databricks bundle run -t {env} {job_name}"

    result1 = subprocess.run(command1, shell=True, capture_output=True, text=True)
    result2 = subprocess.run(command2, shell=True, capture_output=True, text=True)

    if result1.stdout:
        print(result1.stdout)
    if result1.stderr:
        print(result1.stderr)

    if result1.returncode != 0:
        raise RuntimeError(f"Command 1 failed with return code {result1.returncode}. "
                           f"Output: {result1.stdout}\nError: {result1.stderr}")

    if result2.returncode != 0:
        raise RuntimeError(f"Command 2 failed with return code {result2.returncode}. "
                           f"Output: {result2.stdout}\nError: {result2.stderr}")

    if result2.stdout:
        print(result2.stdout)
    if result2.stderr:
        print(result2.stderr)

    return result1.stdout, result1.stderr, result2.stdout, result2.stderr

def driver():
    """Entry point as a wrapper of databricks bundle process to deploy and run a byod bundle.
       Make sure you are in the same directory of databricks.yml, otherwise dabs can't proceed.

       driver:
          1. Finds the matched job based on ``job`` within all the included job ymls in databricks.yml
          2. Modifies the parameters in job.yml according to the arguments passed to the script
          3. Submit the run in a forked subprocess and wait till it completes.

       Note:
          The job arguments must match the params defined in the job.yaml file

       Example usage:

          cd llmfoundry/byod
          python scripts/run.py --job dev_xiaohan_zhang_BYOD_POC_2010 --env dev --tokenizer neox --concat_tokens 1024

       Return:
          0: runs and finishes successfully.

       Throws a Runtime Exception if any stage of the bundle run goes wrong.
    """
    parser = argparse.ArgumentParser(description="Execute Databricks bundles shell commands")
    parser.add_argument("--job", required=True, type=str, help="Name of the job")
    parser.add_argument("--env", required=True, type=str, help="Environment name")
    args, extra_args = parser.parse_known_args()
    job_params = {arg.lstrip('--'): value for arg, value in zip(extra_args[::2], extra_args[1::2])}

    databricks_yml = os.path.join(os.getcwd(), 'databricks.yml')

    if not os.path.exists(databricks_yml):
        raise RuntimeError("Can not find databricks.yml for bundle deploy/run locally! Make sure cd to the bundle root folder first!")

    with open(databricks_yml, 'r') as file:
        yml_data = yaml.safe_load(file)

    job_ymls = yml_data.get('include', [{}])
    job_ymls = [ os.path.join(os.getcwd(), jp) if not os.path.isabs(jp) else jp for jp in job_ymls ]

    if job_ymls is None:
        raise ValueError("Need to provide the path to job.yamls")

    for job_yaml_file in job_ymls:
        with open(job_yaml_file, 'r') as file:
            yml_data = yaml.safe_load(file)

        jobs = yml_data.get('resources').get('jobs')

        for job_name in jobs.keys():
            if job_name != args.job:
                continue

            job_data = yml_data.get('resources', {}).get('jobs', {}).get(job_name, {})
            base_params = job_data.get('tasks', [{}])[0].get('notebook_task', {}).get('base_parameters', {})

            # If new arguments are set then update the job yaml file
            for k, v in job_params.items():
                base_params[k] = v

            # Write out the new yaml file and run job with DAB
            with open(job_yaml_file, 'w') as file:
                yaml.dump(yml_data, file)

            output1, error1, output2, error2 = run(job_name, args.env)
            return 0

    raise RuntimeError("Cannot find any job.yaml that contains the specified job {job_name}!")

if __name__ == "__main__":
    driver()




