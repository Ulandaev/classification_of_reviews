import os
import tempfile

import mlflow


class MLflowLogger:
    def __init__(self, tracking_uri, experiment_name):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name=None):
        mlflow.start_run(run_name=run_name)

    def end_run(self):
        mlflow.end_run()

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics, step=None):
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path):
        mlflow.log_artifact(local_path)

    def log_figure(self, figure, artifact_file):
        """Log matplotlib figure to MLflow"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = os.path.join(tmp_dir, artifact_file)
            figure.savefig(temp_path)
            mlflow.log_artifact(temp_path)

    def log_model(self, model, artifact_path):
        mlflow.sklearn.log_model(model, artifact_path)


def setup_mlflow(cfg):
    return MLflowLogger(
        tracking_uri=cfg.experiment.tracking_uri, experiment_name=cfg.experiment.name
    )
