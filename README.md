# Iguazio MNIST Demo
Converting a MNIST script to run on the platform via MLRun and KubeFlow.

## Code Overview

### Original Code
Original code can be found in the `orig` directory. This includes a script to train the model (`train_digit_recognizer.py`) as well as a Tkinter GUI for model inference (`gui_digit_recognizer.py`).

### Converted Code
The original code has been separated into the following components:
1. `DeployPipeline.ipynb` - Deployment script that creates MLRun project, imports functions from local Python files, defines/runs a KubeFlow pipeline, and queries the model via HTTP REST API.
2. `project/pipeline.py` - KubeFlow pipeline itself. Automatically populated by `DeployPipeline.ipynb`, but is technically a separate file. To completely separate deployment and pipeline code (useful for CI/CD purposes), just remove the pipeline cell from the deployment notebook and edit this file directly.
3. `project/components/(get_prep_data.py,train_eval_model.py,deploy_model.py)` - Pipeline component scripts. Designed to be used in the KubeFlow pipeline and pass elements between components. While `get_prep_data.py` and `train_eval_model.py` are based off of the original code, `deploy_model.py` is our own code.
4. `client/gui_digit_recognizer.py` - Tkinter GUI script designed to be run on local Windows/Mac machine (Tkinter doesn't like Linux). Modified from original script. Uses API Gateway within model inference function for external access to the cluster.

## Getting Started

### Training/Deployment Pipeline
1. Clone git repo to Jupyter service in the platform.
2. Run `~/align_mlrun.sh` in Jupyter service terminal to install correct version of MLRun.
2. Open and run `DeployPipeline.ipynb`. Update project name as you see fit.

### Client GUI Script
1. Create API Gateway for external access to cluster.
    - After running training/deployment notebook, go to Projects tab and select `API gateways (Nuclio)` in bottom left-hand corner.
    - Select `New API Gateway`.
        - No authentication.
        - `Name` whatever you like (such as "inference" or "predict"). Leave `Description` and `Path` blank.
        - For `Primary`, start typing in `deploy-model` and your running Nuclio function should come up.
        - Press Save to create.
2. Clone git repo to local Windows/Mac machine. Install `client/requirements.txt` via pip.
3. Update `API_GATEWAY_URL` in `client/gui_digit_recognizer.py`. Full model URL will be in the form of `http://{API_GATEWAY_URL}/v2/models/model/predict`.
4. Run `client/gui_digit_recognizer.py`.