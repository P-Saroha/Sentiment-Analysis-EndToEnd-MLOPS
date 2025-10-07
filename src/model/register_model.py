"""
Model Registration Module
Registers trained models to MLflow Model Registry
"""

import json
import mlflow
import logging
import dagshub
import os

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("DAGSHUB_TOKEN")
# if not dagshub_token:
#     raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "P-Saroha"
# repo_name = "Sentiment-Analysis-EndToEnd-MLOPS"

# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# Below code block is for local use (comment out for production)
# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri('https://dagshub.com/P-Saroha/Sentiment-Analysis-EndToEnd-MLOPS.mlflow')
dagshub.init(repo_owner='P-Saroha', repo_name='Sentiment-Analysis-EndToEnd-MLOPS', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry (DagHub compatible)."""
    try:
        run_id = model_info['run_id']
        
        # Check if this is a fallback run_id (when MLflow logging failed)
        if run_id == "fallback_run_id":
            logging.info("Fallback mode detected - skipping MLflow model registration")
            print(f" Model '{model_name}' evaluation completed successfully!")
            print("  MLflow model registration skipped due to DagHub limitations")
            print(" Model saved locally at: models/model.pkl")
            
            # Load and display metrics from local files
            try:
                import json
                with open('reports/metrics.json', 'r') as f:
                    metrics = json.load(f)
                print(f" Model Metrics:")
                print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                print(f"   Precision: {metrics.get('precision', 'N/A'):.4f}")
                print(f"   Recall: {metrics.get('recall', 'N/A'):.4f}")
                print(f"   AUC: {metrics.get('auc', 'N/A'):.4f}")
            except Exception as e:
                logging.warning(f"Could not load metrics: {e}")
            
            return None
        
        # Use MLflow run URI for real runs
        model_uri = f"runs:/{run_id}/{model_info['model_path']}"
        logging.info(f"Registering model with URI: {model_uri}")
        
        # Register the model (basic registration for DagHub compatibility)
        model_version = mlflow.register_model(model_uri, model_name)
        
        logging.info(f'Model {model_name} version {model_version.version} registered successfully.')
        logging.info(f'Model URI: {model_uri}')
        logging.info(f'Model Metrics - Accuracy: {model_info.get("accuracy", "N/A")}, '
                    f'Precision: {model_info.get("precision", "N/A")}, '
                    f'Recall: {model_info.get("recall", "N/A")}, '
                    f'AUC: {model_info.get("auc", "N/A")}')
        
        print(f" Model '{model_name}' successfully registered!")
        print(f" Version: {model_version.version}")
        print(f" Accuracy: {model_info.get('accuracy', 'N/A')}")
        print(f" Model URI: {model_uri}")
        
        # Note: Stage transitions may not be supported by DagHub
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            logging.info(f'Model transitioned to Staging stage.')
        except Exception as stage_error:
            logging.warning(f'Could not transition model to Staging stage (DagHub limitation): {stage_error}')
            print("  Note: Model stage transition not supported by DagHub")
        
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        # Handle DagHub specific errors more gracefully
        if "unsupported endpoint" in str(e):
            logging.warning("DagHub endpoint limitation encountered")
            print(f"  Warning: {e}")
            print("Basic model registration may have completed successfully despite the warning.")
        else:
            raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()