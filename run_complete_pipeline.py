"""
Complete ML Pipeline Runner
This script runs the entire ML pipeline from data ingestion to model registration
"""

import sys
import os
import subprocess
import logging

# Add src to path
sys.path.insert(0, os.path.abspath('.'))

def run_pipeline_step(step_name, module_path):
    """Run a pipeline step and handle errors"""
    print(f"\n{'='*50}")
    print(f"🚀 Running: {step_name}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', module_path
        ], check=True, capture_output=True, text=True)
        
        print(f"✅ {step_name} completed successfully!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} failed!")
        print(f"Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    """Run the complete ML pipeline"""
    print("🎯 Starting Complete ML Pipeline")
    print("=" * 60)
    
    # Pipeline steps
    steps = [
        ("Data Ingestion", "src.data.data_ingestion"),
        ("Data Preprocessing", "src.data.data_preprocessing"),
        ("Feature Engineering", "src.features.feature_engineering"),
        ("Model Building", "src.model.model_building"),
        ("Model Evaluation", "src.model.model_evaluation"),
        ("Model Registration", "src.model.register_model")
    ]
    
    success_count = 0
    
    for step_name, module_path in steps:
        if run_pipeline_step(step_name, module_path):
            success_count += 1
        else:
            print(f"\n  Pipeline stopped at: {step_name}")
            break
    
    print(f"\n{'='*60}")
    print(f"📊 Pipeline Summary: {success_count}/{len(steps)} steps completed")
    
    if success_count == len(steps):
        print("🎉 Complete ML Pipeline executed successfully!")
        print("🔗 Check your MLflow dashboard at: https://dagshub.com/P-Saroha/Sentiment-Analysis-EndToEnd-MLOPS.mlflow")
    else:
        print("⚠️  Pipeline completed with some issues.")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()