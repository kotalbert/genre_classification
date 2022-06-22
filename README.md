# Genre Classification

## Mlflow exercise: example ML pipeline.

To run whole pipeline:

```shell
mlflow run .
```

To run single step:

```shell
mlflow run . -P hydra_options="main.execute_steps='random_forest'"
```

To run list of steps:

```shell
mlflow run . -P hydra_options="main.execute_steps='download,preprocess'"
```