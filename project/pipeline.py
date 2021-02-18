
import os
from kfp import dsl
from mlrun import mount_v3io, NewTask
import yaml
import nuclio

funcs = {}

# Configure function resources and local settings
def init_functions(functions: dict, project=None, secrets=None):
    for fn in functions.values():
        # Mount V3IO filesystem
        fn.apply(mount_v3io())

    functions["deploy-model"].spec.base_spec['spec']['loggerSinks'] = [{'level': 'info'}]
    functions["deploy-model"].spec.min_replicas = 1
    functions["deploy-model"].spec.max_replicas = 1
    functions["deploy-model"].spec.default_class = 'MNISTModel'

# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(
    name="MNIST Digit Classification Pipeline",
    description="Kubeflow Pipeline Demo to classify Handwritten Digits"
)
def kfpipeline(batch_size:int=128,
               num_classes:int=10,
               epochs:int=25,
               debug_logs:bool=True):    
    
    # Get data from feature store, prep, train/test split
    get_prep_data = funcs['get-prep-data'].as_step(
        handler="handler",
        outputs=["X_train", "X_test", "y_train", "y_test"],
        verbose=debug_logs)
    
    # Train/evaluate model
    train = funcs['train-eval-model'].as_step(
        handler="handler",
        inputs={"X_train" : get_prep_data.outputs['X_train'],
                "X_test" : get_prep_data.outputs['X_test'],
                "y_train" : get_prep_data.outputs['y_train'],
                "y_test" : get_prep_data.outputs['y_test']},
        params={"batch_size" : batch_size,
                "num_classes" : num_classes,
                "epochs" : epochs},
        outputs=["model"],
        verbose=debug_logs)
    
    # Deploy model
    deploy = funcs["deploy-model"].deploy_step(models=[{"key": "model",
                                                        "model_path": train.outputs['model']}])
