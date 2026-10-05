"""
Polynomial Regression Model Training & Selection Pipeline for BT2024165.

Key concepts implemented:
  - Polynomial feature expansion (degree sweep)
  - Feature standardisation (preventing feature magnitude dominance under L1/L2 penalties)
  - 5-fold cross-validation with fold-isolated scaling (preventing data leakage)
  - L2 Regularisation (Ridge) and L1 Regularisation (Lasso)
  - Warning accounting (capturing non-convergence instead of silently ignoring)
  - Serialization of fitted pipelines and inference output generation
"""

import sys, os, warnings, json
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso

# Global execution configuration
ROLL = 'BT2024165'
N_FOLDS = 5
RANDOM_STATE = 42

def load_dataset(variant: str):
    """Load train features, train target, and test features for a given problem variant."""
    train_path = f"{ROLL}_train_{variant}.csv"
    test_path = f"{ROLL}_test_{variant}.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing dataset files for {variant}: {train_path}, {test_path}")
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    X = df_train.drop(columns='y').values
    y = df_train['y'].values
    X_test = df_test.values
    return X, y, X_test

def cross_validate_degree(model_type: str, P: np.ndarray, y: np.ndarray, hyperparam: float, kf: KFold):
    """
    Evaluates a model on precomputed polynomial features P using K-fold CV.
    StandardScaler is fitted strictly on training folds to prevent data leakage.
    Tracks non-convergence warnings cleanly.
    """
    val_mse_list = []
    warning_count = 0
    
    for train_idx, val_idx in kf.split(P):
        sc = StandardScaler().fit(P[train_idx])
        Ps_train = sc.transform(P[train_idx])
        Ps_val = sc.transform(P[val_idx])
        
        if model_type == 'Ridge':
            estimator = Ridge(alpha=hyperparam, fit_intercept=True)
        elif model_type == 'Lasso':
            estimator = Lasso(alpha=hyperparam, fit_intercept=True, max_iter=20000, tol=1e-5)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            estimator.fit(Ps_train, y[train_idx])
            warning_count += len(caught_warnings)
            
        preds = estimator.predict(Ps_val)
        val_mse_list.append(np.mean((y[val_idx] - preds) ** 2))
        
    mean_mse = float(np.mean(val_mse_list))
    std_err = float(np.std(val_mse_list, ddof=1) / np.sqrt(len(val_mse_list)))
    return mean_mse, std_err, warning_count

def run_training_pipeline():
    """Executes the systematic CV search, model selection, serialization, and test inference."""
    print("=" * 70, flush=True)
    print("BT2024165 Polynomial Regression Training & Model Selection", flush=True)
    print("=" * 70, flush=True)
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    search_configs = {
        'var1': {
            'degrees': [4, 5, 6, 7],
            'ridge_alphas': np.logspace(-4, 1, 11),
            'lasso_alphas': np.logspace(-4, -1, 7)
        },
        'var2': {
            'degrees': [8, 10, 12, 14],
            'ridge_alphas': np.logspace(-5, 0, 11),
            'lasso_alphas': np.logspace(-4, -1, 7)
        }
    }
    
    cv_records = []
    best_models = {}
    
    for variant, cfg in search_configs.items():
        print(f"\n---> Evaluating Variant: {variant}", flush=True)
        X, y, X_test = load_dataset(variant)
        
        best_variant_mse = np.inf
        best_variant_spec = None
        
        for deg in cfg['degrees']:
            P = PolynomialFeatures(deg, include_bias=False).fit_transform(X)
            nf = P.shape[1]
            print(f"  Degree {deg:2d} ({nf:4d} features)...", end=" ", flush=True)
            
            # Ridge sweep
            for alpha in cfg['ridge_alphas']:
                mse, se, warns = cross_validate_degree('Ridge', P, y, alpha, kf)
                rec = {
                    'problem': variant, 'degree': deg, 'model': 'Ridge',
                    'n_features': nf, 'hyperparam': alpha, 'cv_mse': mse, 'cv_se': se, 'warnings': warns
                }
                cv_records.append(rec)
                if mse < best_variant_mse:
                    best_variant_mse = mse
                    best_variant_spec = rec
            
            # Lasso sweep (manageable degrees)
            if (variant == 'var1' and deg <= 5) or (variant == 'var2' and deg <= 12):
                for alpha in cfg['lasso_alphas']:
                    mse, se, warns = cross_validate_degree('Lasso', P, y, alpha, kf)
                    rec = {
                        'problem': variant, 'degree': deg, 'model': 'Lasso',
                        'n_features': nf, 'hyperparam': alpha, 'cv_mse': mse, 'cv_se': se, 'warnings': warns
                    }
                    cv_records.append(rec)
                    if mse < best_variant_mse:
                        best_variant_mse = mse
                        best_variant_spec = rec
                        
            print(f"best MSE so far = {best_variant_mse:.4f}", flush=True)
            
        print(f"  >> Best for {variant}: {best_variant_spec['model']} (degree {best_variant_spec['degree']}, "
              f"alpha={best_variant_spec['hyperparam']:.2e}) -> CV MSE: {best_variant_spec['cv_mse']:.4f} +/- {best_variant_spec['cv_se']:.4f}", flush=True)
        best_models[variant] = best_variant_spec
        
    # Save CV results
    cv_df = pd.DataFrame(cv_records)
    cv_df.to_csv('cv_results.csv', index=False)
    print("\nSaved CV results to 'cv_results.csv'.", flush=True)
    
    # Fit final models on 100% of training data, save pipelines, and produce predictions
    for variant, spec in best_models.items():
        X, y, X_test = load_dataset(variant)
        deg = spec['degree']
        alpha = spec['hyperparam']
        model_name = spec['model']
        
        if model_name == 'Ridge':
            final_estimator = Ridge(alpha=alpha, fit_intercept=True)
        else:
            final_estimator = Lasso(alpha=alpha, fit_intercept=True, max_iter=20000, tol=1e-5)
            
        final_pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
            ('scaler', StandardScaler()),
            ('reg', final_estimator)
        ])
        
        final_pipeline.fit(X, y)
        
        # Save pipeline object to disk
        model_filename = f"{variant}_model.joblib"
        joblib.dump(final_pipeline, model_filename)
        print(f"Saved fitted model pipeline to '{model_filename}'.", flush=True)
        
        # Generate and save predictions
        test_preds = final_pipeline.predict(X_test)
        pred_filename = f"{ROLL}_pred_{variant}.csv"
        pd.DataFrame({'y': test_preds}).to_csv(pred_filename, index=False)
        print(f"Saved test predictions to '{pred_filename}'.", flush=True)
        
        # Count non-zero coefficients
        coefs = final_pipeline.named_steps['reg'].coef_
        total_feats = len(coefs)
        nonzero_feats = int(np.sum(coefs != 0))
        spec['n_features'] = total_feats
        spec['nonzero_features'] = nonzero_feats
        
    with open('best_models.json', 'w') as f:
        json.dump(best_models, f, indent=2)
    print("Saved best model metadata to 'best_models.json'.", flush=True)

if __name__ == '__main__':
    run_training_pipeline()
