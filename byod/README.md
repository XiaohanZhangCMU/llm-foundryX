# BYOD
<br/> 
BYOD (Bring Your Own Data) is a Databricks Asset Bundle (DABs) containinig a suite of ETL workflows for MosaicML's LLM pretraining and Finetuning, validated by monthly regression testings. DABs is a wrapper of databricks-sdk that handles the deployment and running of workflow jobs. 
<br/>
<br/>
BYOD serves two primary purposes: 

- CI/CD: The research-data-team consistently integrates new notebooks into BYOD, ensuring routine testing and verification.
- Workflow API: External services leveraging existing Spark workflows can utilize BYOD as a convenient API entry point.
<br/>  
To get started, you will need to configure the access of databricks-as-code.

1. Install [Databricks CLI](https://docs.databricks.com/dev-tools/cli/databricks-cli.html). You can either install it on your laptop or any clusters. 

2. Authenticate to your Databricks workspace:
    ```
    databricks configure
    ```   
   See [PAT documents](https://docs.databricks.com/en/dev-tools/auth.html#pat) for more guidance. Basically you will need to set up two environment variables specific to your workspace:
   ```
   DATABRICKS_HOST and `DATABRICKS_TOKEN`
   ```
3. To test if the authentication is working, Type:
   ```
   databricks fs ls dbfs:/something
   ```

Check existing template in [byod/bundles](https://github.com/databricks/byod/tree/simple_pipeline). For example, one template developed is ```c4_arxiv```, which has two ingest and process c4/en and arxiv dataset 

![alt text](https://github.com/databricks/byod/blob/simple_pipeline/bundles/c4_arxiv/workflow.png)

Let's use bundles/ft_data_prep as an example and walk through the steps to set up byod and illustrate how to add your own data workflow.

1. Set up local virtual environment and byod repo
   ```
   python3.10 -m venv ven_byod_py310
   git clone https://github.com/databricks/byod.git
   git checkout simple_pipeline [to be removed]
   ```
2. Set up a new folder for your source scripts.
   ```
   mkdir bundles/ft_data_pret
   ```
   The new folder will contain all of your notebooks/python scripts and a job.yml that defines the job to be executed. Remember to add the new job.yml file to databricks.cfg alongside with other jobs. 
3. Some tips in preparing job.yml:
    - ``How do I find my cluster id?`` If you don't have a cluster yet, create one. Go to the cluster page and change the view to Json. Copy the cluster id from there (at the bottom). 
    - ``How should I prepare the job yaml file? It looks complicated!``  
        If you are super familiar with DABs and Databricks' eco system, you can skip and write your own yml. Otherwise, it's recommended to modify existing template job.ymls and stay close to them. If you need something completely different like a complicated workflow, you probably need to first try it out in the workflow UI, then click on the three-dots on the right-upper corner to ``VIEW YAML/JSON``. Copy the content and replace the template yaml with it. Don't forget update `notebook_path` sections.

4. After all the files (yaml plus python scripts) are ready, type:
   ```
   make all
   ```
   This builds byod as a wheel file and publish it to dbfs:/xiaohan-test. Clusters are configured to install from the dbfs wheel file at start. Note: this step will not be necessary if byod is pbulic accessible repo such that it can be pip install from github source directly. Or if your data prep scripts do not need to import anything from byod, it's safe to skip this step as well. 
4. To launch the byod job.
* **Option A** Use databricks bundles CLI commands:
    ```
    databricks bundle deploy --target dev (or prod)
    ```
    (Note that "dev" is the default target, so the `--target` parameter is optional here.)

    This deploys everything that has been defined for this project. For example, the default template would deploy a job called `[dev yourname] c4_arxiv_poc_job` to your workspace. You can find that job by opening your workpace and clicking on **Workflows**. Notice that DABs will run some sanity checks of your folder, for example, if a notebook is missing it will report an error and stop. Also, remember that the workflow job name is defined in resources/job.yml. It is allowed to have duplicated job names in workflow, but I found it useful to rename job before deployment. And run the deployed job with
   ```
   databricks bundle run c4_arxiv_poc_job
   ```
   To specify a target name, use
   ```
   databricks bundle run -t dev c4_arxiv_poc_job
   ```
   See [Databricks CI/CD with DABs](https://docs.databricks.com/en/dev-tools/bundles/work-tasks.html) for more options.

* **Option B** Call byod's orchestrate function:
   run_byod_with_DABs internally forks a subprocess to run databricks bundle as shell command. It has some error handling such as raise RuntimeError if either deployment or job run fails. It returns stdout and stderr of both deployment and run stages. 
   ```
   from byod import run_byod_with_DABs
   run_byod_with_DABs(job_name, environment)
   ```
* **Option C** Use jobAPI instead of DABs.
   Note in this way we don't have DABs' sanity check on the job definitions. And the user need to handle the deployment manually. 
   ```
   from byod import run_byod_with_jobAPI
   cfg = path/to/task/yml or cfg = omegaconf.create(job definition)
   run_byod_with_jobAPI(cfg)
   ```

