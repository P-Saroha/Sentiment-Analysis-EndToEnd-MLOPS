#!/usr/bin/env python3
"""
Simple ML Pipeline Runner - Safe for version control
"""
import subprocess
import sys

def run_step(command, description):
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print("✅ SUCCESS!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    python_exe = "F:/Sentiment-Analysis-EndToEnd-MLOPS/myenv/Scripts/python.exe"
    
    steps = [
        (f"{python_exe} src\\data\\data_ingestion.py", "Step 1: Data Ingestion"),
        (f"{python_exe} src\\data\\data_preprocessing.py", "Step 2: Data Preprocessing"),  
        (f"{python_exe} src\\features\\feature_engineering.py", "Step 3: Feature Engineering"),
    ]
    
    print("🎯 Starting ML Pipeline")
    
    for command, description in steps:
        success = run_step(command, description)
        if not success:
            print(f"⚠️ Pipeline failed at: {description}")
            return
    
    print(f"\n{'='*50}")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("📊 Generated Files:")
    print("  - data/raw/train.csv, test.csv")
    print("  - data/interim/train_processed.csv, test_processed.csv")
    print("  - data/processed/train_bow.csv, test_bow.csv") 
    print("  - models/vectorizer.pkl")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()