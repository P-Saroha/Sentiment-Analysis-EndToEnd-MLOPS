import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Suppress MLflow artifact download warnings
# os.environ["MLFLOW_DISABLE_ARTIFACTS_DOWNLOAD"] = "1"

# Set MLflow Tracking URI & DAGsHub integration
def setup_mlflow():
    """Setup MLflow with fallback to local tracking if DagHub is unavailable."""
    MLFLOW_TRACKING_URI = "https://dagshub.com/P-Saroha/Sentiment-Analysis-EndToEnd-MLOPS.mlflow"
    try:
        # Try to connect to DagHub
        dagshub.init(repo_owner="P-Saroha", repo_name="Sentiment-Analysis-EndToEnd-MLOPS", mlflow=True)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("LoR Hyperparameter Tuning")
        print("✓ Connected to DagHub MLflow tracking")
        return True
    except Exception as e:
        print(f"⚠ Failed to connect to DagHub: {e}")
        print("📁 Falling back to local MLflow tracking")
        # Fallback to local tracking
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("LoR Hyperparameter Tuning")
        return False

# Setup MLflow with error handling
dagshub_connected = setup_mlflow()


# ==========================
# Text Preprocessing Functions
# ==========================
def preprocess_text(text):
    """Applies multiple text preprocessing steps."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # Remove punctuation
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatization & stopwords removal
    
    return text.strip()


# ==========================
# Load & Prepare Data
# ==========================
def load_and_prepare_data(filepath):
    """Loads, preprocesses, and vectorizes the dataset."""
    df = pd.read_csv(filepath)
    
    # Apply text preprocessing
    df["review"] = df["review"].astype(str).apply(preprocess_text)
    
    # Filter for binary classification
    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})
    
    # Convert text data to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer


# ==========================
# Train & Log Model
# ==========================
def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer):
    """Trains a Logistic Regression model with GridSearch and logs results to MLflow."""
    
    param_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }
    
    with mlflow.start_run():
        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Log all hyperparameter tuning runs
        for params, mean_score, std_score in zip(grid_search.cv_results_["params"], 
                                                 grid_search.cv_results_["mean_test_score"], 
                                                 grid_search.cv_results_["std_test_score"]):
            with mlflow.start_run(run_name=f"LR with params: {params}", nested=True):
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score
                }
                
                # Log parameters & metrics
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                
                print(f"🔍 Params: {params}")
                print(f"📊 Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | CV Score: {mean_score:.4f} ± {std_score:.4f}")
                print("-" * 80)

        # Log the best model with error handling
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)
        
        # Try to log the model with error handling
        try:
            mlflow.sklearn.log_model(best_model, "model")
            print("✅ Model logged successfully to MLflow")
        except Exception as model_log_error:
            print(f"⚠ Warning: Could not log model to remote tracking: {model_log_error}")
            print("📊 Metrics and parameters were still logged successfully")
        
        print(f"\n🏆 Best Hyperparameters: {best_params}")
        print(f"🎯 Best Cross-Validation F1 Score: {best_f1:.4f}")
        
        # Save results to local file as backup
        with open("hyperparameter_results.txt", "w", encoding="utf-8") as f:
            f.write("Logistic Regression Hyperparameter Tuning Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Best Parameters: {best_params}\n")
            f.write(f"Best F1 Score: {best_f1:.4f}\n")
            f.write(f"MLflow Tracking: {'Remote (DagHub)' if dagshub_connected else 'Local (./mlruns)'}\n")
        print("📄 Results saved to: hyperparameter_results.txt")


# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    print("🚀 Starting Logistic Regression Hyperparameter Tuning")
    print(f"📊 MLflow tracking: {'Remote (DagHub)' if dagshub_connected else 'Local (./mlruns)'}")
    print("🔧 Grid Search Parameters: C=[0.1, 1, 10], penalty=['l1', 'l2'], solver=['liblinear']")
    print("=" * 80)
    
    try:
        (X_train, X_test, y_train, y_test), vectorizer = load_and_prepare_data("notebooks/data.csv")
        print(f"📈 Data loaded - Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
        print("🔄 Starting hyperparameter tuning...")
        train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)
        print("\n✅ Hyperparameter tuning completed successfully!")
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise
