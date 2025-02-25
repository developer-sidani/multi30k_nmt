# src/comet_logger.py
import os
from dotenv import load_dotenv
from comet_ml import Experiment
import subprocess

def send_message(message):
    """
    Executes a Python script with a message as an argument.

    Args:
        message (str): The message to pass to the script.

    Returns:
        tuple: A tuple containing (stdout, stderr) as decoded strings.

    Raises:
        RuntimeError: If the script exits with a non-zero status.
    """
    try:
        # Expand `~` to the home directory
        script_path = os.path.expanduser("~/message.py")
        
        # Prepare the command
        command = ["python3", script_path, message]
        
        # Run the Python script
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Decode and return stdout and stderr
        return result.stdout.decode(), result.stderr.decode()

    except subprocess.CalledProcessError as e:
        # Raise an exception with detailed error information
        raise RuntimeError(f"Error executing script: {e.stderr.decode()}") from e
    


def init_comet(args, mode):
    load_dotenv()  # load .env file
    comet_api_key = os.getenv('COMET_API_KEY')
    comet_project = os.getenv('COMET_PROJECT_NAME')
    comet_workspace = os.getenv('COMET_WORKSPACE')
    # Set your Comet credentials as environment variables
    # os.environ["COMET_API_KEY"] = comet_api_key
    # os.environ["COMET_WORKSPACE"] = comet_workspace
    # os.environ["COMET_PROJECT_NAME"] = comet_project

    experiment = Experiment(api_key=comet_api_key, project_name=comet_project, workspace=comet_workspace)
    experiment.set_name(f"{mode}_{args.src}_to_{args.tgt}")
    experiment.log_parameters(args)
    send_message(f"Experiment URL: {experiment.url}")
    return experiment
