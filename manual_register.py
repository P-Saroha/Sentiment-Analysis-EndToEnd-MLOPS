"""
Manual model registration script
"""
import mlflow
import dagshub
import json
import pickle

# Initialize DagHub and MLflow
mlflow.set_tracking_uri('https://dagshub.com/P-Saroha/Sentiment-Analysis-EndToEnd-MLOPS.mlflow')
dagshub.init(repo_owner='P-Saroha', repo_name='Sentiment-Analysis-EndToEnd-MLOPS', mlflow=True)

def register_model_manually():
    try:
        # Use the run ID from the earlier successful model evaluation
        run_id = "c7003d86abef43beb4e3600813c45819"  # From your earlier output
        model_name = "my_model"
        
        # Create model URI from the run
        model_uri = f"runs:/{run_id}/model"
        print(f"Attempting to register model from URI: {model_uri}")
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        print(f"✅ Model '{model_name}' successfully registered!")
        print(f"   Version: {model_version.version}")
        print(f"   Run ID: {run_id}")
        print(f"   Model URI: {model_uri}")
        
        return model_version
        
    except Exception as e:
        print(f"❌ Error registering model: {e}")
        print("This is likely due to DagHub MLflow limitations.")
        print("The model can still be loaded directly from the run.")
        return None

if __name__ == "__main__":
    register_model_manually()