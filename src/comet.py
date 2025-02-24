# src/comet_logger.py
import os
from dotenv import load_dotenv
from comet_ml import Experiment

def init_comet(args, mode):
    load_dotenv()  # load .env file
    comet_api_key = os.getenv('COMET_API_KEY')
    comet_project = os.getenv('COMET_PROJECT_NAME')
    comet_workspace = os.getenv('COMET_WORKSPACE')
    experiment = Experiment(api_key=comet_api_key, project_name=comet_project, workspace=comet_workspace)
    experiment.set_name(f"{mode}_{args.src}_to_{args.tgt}")
    experiment.log_parameters(args)
    return experiment
