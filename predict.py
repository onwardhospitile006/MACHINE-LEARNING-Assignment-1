"""
BT2024165 – Standalone Inference Script.
Loads pre-trained model pipelines from disk and produces test set predictions.
"""

import sys, os
import pandas as pd
import joblib

ROLL = 'BT2024165'

def run_predictions():
    print("=" * 60)
    print("BT2024165 Standalone Model Prediction Pipeline")
    print("=" * 60)
    
    for variant in ['var1', 'var2']:
        model_path = f"{variant}_model.joblib"
        test_path = f"{ROLL}_test_{variant}.csv"
        out_path = f"{ROLL}_pred_{variant}.csv"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found. Please run 'train.py' first.")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data file '{test_path}' not found.")
            
        print(f"\n[+] Loading model: {model_path}")
        pipeline = joblib.load(model_path)
        
        print(f"[+] Loading test features: {test_path}")
        df_test = pd.read_csv(test_path)
        X_test = df_test.values
        
        print(f"[+] Generating predictions for {len(X_test)} samples...")
        preds = pipeline.predict(X_test)
        
        df_pred = pd.DataFrame({'y': preds})
        df_pred.to_csv(out_path, index=False)
        print(f"[OK] Wrote predictions to: {out_path} ({len(df_pred)} rows)")

if __name__ == '__main__':
    run_predictions()
