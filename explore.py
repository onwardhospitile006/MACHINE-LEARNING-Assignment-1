"""
BT2024165 – Exploratory OLS Sweep across degrees.
Demonstrates empirical bias-variance trade-off and over-fitting without regularisation.
Saves results to 'explore_ols_results.csv' for complete reproducibility.
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

ROLL, K, SEED = 'BT2024165', 5, 42

def cv_ols(X, y, d, kf):
    pf = PolynomialFeatures(d, include_bias=False)
    P = pf.fit_transform(X)
    nf = P.shape[1]
    
    # Train MSE on full dataset
    lr_full = LinearRegression().fit(P, y)
    train_mse = float(np.mean((y - lr_full.predict(P)) ** 2))
    
    # K-fold CV MSE
    val_errs = []
    for tr, va in kf.split(P):
        lr = LinearRegression().fit(P[tr], y[tr])
        val_errs.append(np.mean((y[va] - lr.predict(P[va])) ** 2))
    
    cv_mse = float(np.mean(val_errs))
    cv_se = float(np.std(val_errs, ddof=1) / np.sqrt(len(val_errs)))
    return nf, train_mse, cv_mse, cv_se

if __name__ == '__main__':
    kf = KFold(K, shuffle=True, random_state=SEED)
    records = []
    
    print(f"{'Problem':<8} {'Degree':<8} {'Features':<10} {'Train MSE':<12} {'CV MSE':<12} {'CV SE':<10}")
    print("-" * 62)
    
    for v, max_deg in [('var1', 6), ('var2', 12)]:
        df = pd.read_csv(f'{ROLL}_train_{v}.csv')
        X = df.drop(columns='y').values
        y = df.y.values
        
        for d in range(1, max_deg + 1):
            nf, tr_mse, cv_mse, cv_se = cv_ols(X, y, d, kf)
            records.append({
                'problem': v,
                'degree': d,
                'n_features': nf,
                'train_mse': tr_mse,
                'cv_mse': cv_mse,
                'cv_se': cv_se
            })
            print(f"{v:<8} {d:<8} {nf:<10} {tr_mse:<12.4f} {cv_mse:<12.4f} {cv_se:<10.4f}")
    
    res_df = pd.DataFrame(records)
    res_df.to_csv('explore_ols_results.csv', index=False)
    print("\nSaved unregularised OLS baseline to 'explore_ols_results.csv'.")
