"""
Simple Model Evaluation - Windows Compatible
This version avoids Unicode encoding issues with Windows terminal
"""

import sys
import os
import pickle
import pandas as pd
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import json
import mlflow
import dagshub

# Add src to path
sys.path.insert(0, os.path.abspath('.'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s')

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

def load_model(file_path: str):
    """Load the trained model from pickle file"""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('Model file not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Error loading model: %s', e)
        raise

def load_data(file_path: str):
    """Load the test data"""
    try:
        data = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return data
    except FileNotFoundError:
        logging.error('Data file not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Error loading data: %s', e)
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics"""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        
        logging.info('Model evaluation metrics calculated')
        return metrics
        
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str):
    """Save metrics to JSON file"""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error saving metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.info('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    """Main evaluation function with MLflow integration"""
    try:
        print("Starting model evaluation...")
        
        # Load model and test data
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_bow.csv')
        
        # Prepare test data
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        
        # Evaluate model
        metrics = evaluate_model(clf, X_test, y_test)
        
        # Save metrics locally first
        save_metrics(metrics, 'reports/metrics.json')
        
        # Try MLflow logging with error handling for Windows Unicode issues
        try:
            mlflow.set_experiment("sentiment-analysis-production")
            
            with mlflow.start_run() as run:
                # Log metrics to MLflow
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model parameters to MLflow
                if hasattr(clf, 'get_params'):
                    params = clf.get_params()
                    for param_name, param_value in params.items():
                        mlflow.log_param(param_name, param_value)
                
                # Log model to MLflow
                mlflow.sklearn.log_model(clf, "model")
                
                # Save model info for registration
                save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
                
                # Log the metrics file to MLflow
                mlflow.log_artifact('reports/metrics.json')
                
                run_id = run.info.run_id
                print(f"MLflow Run ID: {run_id}")
                
        except Exception as mlflow_error:
            logging.warning(f'MLflow logging failed (Windows Unicode issue): {mlflow_error}')
            # Create a fallback experiment info for model registration
            fallback_info = {
                'run_id': 'fallback_run_id',
                'model_path': 'model'
            }
            save_model_info(fallback_info['run_id'], fallback_info['model_path'], 'reports/experiment_info.json')
            print("MLflow logging failed but continuing with local evaluation...")
        
        # Display results
        print("\\n=== Model Evaluation Results ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print("================================\\n")
        
        print("Model evaluation completed successfully!")
        return True
        
    except Exception as e:
        logging.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)