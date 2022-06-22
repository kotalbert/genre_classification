import logging

import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    steps_ = config["main"]["execute_steps"]
    if isinstance(steps_, str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = steps_.split(",")
    else:
        assert isinstance(steps_, ListConfig)
        steps_to_execute = steps_

    input_artifact_name = "raw_data.parquet"
    if "download" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "download"),
            "main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": input_artifact_name,
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            },
        )

    preprocessed_artifact_name = "preprocessed_data.csv"
    if "preprocess" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "preprocess"),
            "main",
            parameters={
                "input_artifact": f"{input_artifact_name}:latest",
                "artifact_name": preprocessed_artifact_name,
                "artifact_type": "preprocessed_data",
                "artifact_description": "Data preprocessed"
            }
        )

    if "check_data" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "check_data"),
            "main",
            parameters={
                "reference_artifact": config["data"]["reference_dataset"],
                "sample_artifact": f"{preprocessed_artifact_name}:latest",
                "ks_alpha": config["data"]["ks_alpha"],
            }
        )

    if "segregate" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "segregate"),
            "main",
            parameters={
                "input_artifact": f"{preprocessed_artifact_name}:latest",
                "artifact_root": "sample",
                "artifact_type": "split_data",
                "test_size": config["data"]["test_size"],
                "random_state": config["main"]["random_seed"],
                "stratify": config["data"]["stratify"]
            }
        )

    rf_pipeline = config["random_forest_pipeline"]
    if "random_forest" in steps_to_execute:
        # Serialize decision tree configuration
        model_config = os.path.abspath("random_forest_config.yml")

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(rf_pipeline))

        _ = mlflow.run(
            os.path.join(root_path, "random_forest"),
            "main",
            parameters={
                "train_data": "sample_train.csv:latest",
                "model_config": model_config,
                "export_artifact": rf_pipeline["export_artifact"],
                "random_seed": rf_pipeline["random_forest"]["random_state"],
                "val_size": config["data"]["val_size"],
                "stratify": config["data"]["stratify"]

            }

        )

    if "evaluate" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "evaluate"),
            "main",
            parameters={
                "model_export": f'{rf_pipeline["export_artifact"]}:latest',
                "test_data": "sample_test.csv:latest"
            }
        )


if __name__ == "__main__":
    go()
