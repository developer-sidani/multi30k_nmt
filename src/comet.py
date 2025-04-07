from comet_ml import Experiment, ExistingExperiment
from dotenv import load_dotenv
import os
from typing import Dict, Optional, Union


def setup_comet_experiment(
        args,
        experiment_name: str = None,
        tags: Optional[list] = None,
        log_code: bool = True,
        log_graph: bool = True,
        auto_metric_logging: bool = True,
        existing_experiment: str = None
) -> Union[Experiment, None]:
    """
    Set up a Comet ML experiment for logging.

    Args:
        args: Command-line arguments
        experiment_name: Name of the experiment
        tags: List of tags for the experiment
        log_code: Whether to log code
        log_graph: Whether to log the model graph
        auto_metric_logging: Whether to enable automatic metric logging
        existing_experiment: Existing experiment key to resume

    Returns:
        Comet ML experiment if comet_logging is enabled, None otherwise
    """
    if not args.comet_logging:
        return None

    # Load API key and other info from .env file
    load_dotenv()
    api_key = os.getenv('COMET_API_KEY') or args.comet_key
    workspace = os.getenv('COMET_WORKSPACE') or args.comet_workspace
    project_name = os.getenv('COMET_PROJECT_NAME') or args.comet_project_name

    # Set up experiment
    if existing_experiment:
        experiment = ExistingExperiment(
            api_key=api_key,
            previous_experiment=existing_experiment,
            workspace=workspace,
            project_name=project_name,
            log_code=log_code,
            log_graph=log_graph,
            auto_metric_logging=auto_metric_logging
        )
    else:
        experiment = Experiment(
            api_key=api_key,
            workspace=workspace,
            project_name=project_name,
            log_code=log_code,
            log_graph=log_graph,
            auto_metric_logging=auto_metric_logging
        )

    # Set experiment name if provided
    if experiment_name:
        experiment.set_name(experiment_name)

    # Add tags if provided
    if tags:
        experiment.add_tags(tags)

    # Log command-line arguments
    experiment.log_parameters(vars(args))

    print(f"Comet.ml experiment URL: {experiment.url}")
    return experiment


def log_metrics(
        experiment: Experiment,
        metrics: Dict,
        step: int = None,
        epoch: int = None,
        phase: str = "train"
) -> None:
    """
    Log metrics to Comet ML.

    Args:
        experiment: Comet ML experiment
        metrics: Dictionary of metrics to log
        step: Current step
        epoch: Current epoch
        phase: 'train', 'validation' or 'test'
    """
    if experiment is None:
        return

    context = None
    if phase == "train":
        context = experiment.train
    elif phase == "validation":
        context = experiment.validate
    elif phase == "test":
        context = experiment.test

    with context():
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                experiment.log_metric(
                    metric_name,
                    metric_value,
                    step=step,
                    epoch=epoch
                )


def log_artifacts(
        experiment: Experiment,
        artifacts: Dict[str, str],
        step: int = None,
        epoch: int = None
) -> None:
    """
    Log artifacts to Comet ML.

    Args:
        experiment: Comet ML experiment
        artifacts: Dictionary mapping artifact names to file paths
        step: Current step
        epoch: Current epoch
    """
    if experiment is None:
        return

    for name, path in artifacts.items():
        experiment.log_artifact(path, name=name, step=step, epoch=epoch)


def log_model(
        experiment: Experiment,
        model_path: str,
        step: int = None,
        epoch: int = None
) -> None:
    """
    Log model to Comet ML.

    Args:
        experiment: Comet ML experiment
        model_path: Path to the model directory
        step: Current step
        epoch: Current epoch
    """
    if experiment is None:
        return

    experiment.log_model("seq2seq", model_path, step=step, epoch=epoch)