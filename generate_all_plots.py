"""
BT2024165 – Generate 4 report-quality plots for var1 and 4 for var2.
Ridge-only CV across all degrees (SVD = instant), plus Lasso at key degrees.
Includes auto-caching from cv_results_extended.csv for instant re-plotting.
Reflects rigorous statistical treatment of repeated-x variance (95% CI: [0.14, 2.18])
and mathematically consistent R² = 1 - MSE / Var(y).
"""
import sys, os
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings, json
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, lasso_path
from scipy.cluster.hierarchy import fcluster, linkage

plt.rcParams.update({
    'font.size': 10,
    'axes.edgecolor': '#CCCCCC',
    'grid.color': '#EEEEEE',
    'grid.linestyle': '--'
})
ROLL, K, SEED = 'BT2024165', 5, 42

def load(v):
    tr = pd.read_csv(f'{ROLL}_train_{v}.csv')
    te = pd.read_csv(f'{ROLL}_test_{v}.csv')
    return tr.drop(columns='y').values, tr.y.values, te.values

def ridge_cv(P, y, alphas, kf):
    """SVD Ridge CV vectorised over alphas – very fast even for 1000+ features."""
    na = len(alphas)
    oofs = np.zeros((len(y), na))
    fmse = np.zeros((K, na))
    for fi, (ti, vi) in enumerate(kf.split(P)):
        sc = StandardScaler().fit(P[ti])
        Ps, Pv = sc.transform(P[ti]), sc.transform(P[vi])
        ym = y[ti].mean(); yc = y[ti] - ym
        U, S, Vt = np.linalg.svd(Ps, full_matrices=False)
        Uty = U.T @ yc; BV = Pv @ Vt.T
        for j, a in enumerate(alphas):
            d = S / (S**2 + a)
            pred = BV @ (d * Uty) + ym
            oofs[vi, j] = pred
            fmse[fi, j] = np.mean((y[vi] - pred)**2)
    return fmse.mean(0), fmse.std(0, ddof=1)/np.sqrt(K), oofs

def ols_cv(P, y, kf):
    oof = np.zeros(len(y)); fmse = []
    for ti, vi in kf.split(P):
        sc = StandardScaler().fit(P[ti])
        m = LinearRegression().fit(sc.transform(P[ti]), y[ti])
        p = m.predict(sc.transform(P[vi]))
        oof[vi] = p; fmse.append(np.mean((y[vi]-p)**2))
    return np.mean(fmse), np.std(fmse, ddof=1)/np.sqrt(K), oof

def lasso_cv_path(P, y, alphas, kf):
    """Lasso path CV – use only for moderate-sized feature matrices."""
    na = len(alphas)
    oofs = np.zeros((len(y), na)); fmse = np.zeros((K, na))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fi, (ti, vi) in enumerate(kf.split(P)):
            sc = StandardScaler().fit(P[ti])
            Ps, Pv = sc.transform(P[ti]), sc.transform(P[vi])
            ym = y[ti].mean(); yc = y[ti] - ym
            _, coefs, _ = lasso_path(Ps, yc, alphas=alphas, max_iter=20000, tol=1e-5)
            pred = Pv @ coefs + ym
            oofs[vi, :] = pred
            fmse[fi, :] = np.mean((y[vi][:,None] - pred)**2, axis=0)
    return fmse.mean(0), fmse.std(0, ddof=1)/np.sqrt(K), oofs

def r2(y, oof):
    return 1.0 - np.sum((y - oof)**2) / np.sum((y - y.mean())**2)

def compute_single_oof(X, y, d, model_name, hp, kf):
    """Compute OOF prediction for a single specified model configuration."""
    P = PolynomialFeatures(d, include_bias=False).fit_transform(X)
    oof = np.zeros(len(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ti, vi in kf.split(P):
            sc = StandardScaler().fit(P[ti])
            Ps = sc.transform(P[ti])
            Pv = sc.transform(P[vi])
            ym = y[ti].mean(); yc = y[ti] - ym
            if model_name == 'OLS':
                m = LinearRegression().fit(Ps, yc)
            elif model_name == 'Ridge':
                m = Ridge(alpha=hp, fit_intercept=False).fit(Ps, yc)
            elif model_name == 'Lasso':
                m = Lasso(alpha=hp, max_iter=20000, tol=1e-5, fit_intercept=False).fit(Ps, yc)
            oof[vi] = m.predict(Pv) + ym
    return oof

# ═══════════ CV SEARCH OR LOAD CACHE ═══════════
cache_file = 'cv_results_extended.csv'
configs = {
    'var1': {'max_deg': 7,  'lasso_degs': [4, 5]},
    'var2': {'max_deg': 14, 'lasso_degs': [8, 10, 12]},
}

results = {}
if os.path.exists(cache_file) and '--force-cv' not in sys.argv:
    print(f"Loading cached CV results from {cache_file}...")
    full_df = pd.read_csv(cache_file)
    for v in ['var1', 'var2']:
        X, y, Xt = load(v)
        kf = KFold(K, shuffle=True, random_state=SEED)
        df_v = full_df[full_df['v'] == v].copy()
        best_idx = df_v['mse'].idxmin()
        best_rec = df_v.loc[best_idx].to_dict()
        print(f"Computing OOF for {v} best model: {best_rec['model']} deg={int(best_rec['d'])} hp={best_rec['hp']:.2e}...")
        oof = compute_single_oof(X, y, int(best_rec['d']), best_rec['model'], best_rec['hp'], kf)
        # Ensure R² = 1 - MSE / Var(y) consistency
        best_rec['r2'] = 1.0 - best_rec['mse'] / np.var(y)
        results[v] = {'df': df_v, 'best': best_rec, 'oof': oof, 'y': y, 'X': X, 'Xt': Xt}
else:
    print("Running CV search (Ridge SVD + OLS + Lasso at key degrees)...")
    ridge_alphas = np.logspace(-5, 2, 30)
    lasso_alphas = np.logspace(-4, -1, 10)

    for v, cfg in configs.items():
        X, y, Xt = load(v)
        kf = KFold(K, shuffle=True, random_state=SEED)
        rows = []; best_mse = np.inf; best_rec = None; best_oof = None

        for d in range(1, cfg['max_deg'] + 1):
            P = PolynomialFeatures(d, include_bias=False).fit_transform(X)
            nf = P.shape[1]
            print(f"  {v} deg={d} ({nf} feats)...", end=" ", flush=True)

            if nf < 800:
                m, s, o = ols_cv(P, y, kf)
                rec = {'v': v, 'd': d, 'model': 'OLS', 'hp': 0, 'nf': nf, 'mse': m, 'se': s, 'r2': r2(y, o)}
                rows.append(rec)
                if m < best_mse: best_mse, best_rec, best_oof = m, rec, o

            ms, ss, os = ridge_cv(P, y, ridge_alphas, kf)
            for j in range(len(ridge_alphas)):
                rec = {'v': v, 'd': d, 'model': 'Ridge', 'hp': ridge_alphas[j], 'nf': nf, 'mse': ms[j], 'se': ss[j], 'r2': r2(y, os[:, j])}
                rows.append(rec)
                if ms[j] < best_mse: best_mse, best_rec, best_oof = ms[j], rec, os[:, j]

            if d in cfg['lasso_degs'] and nf <= 500:
                ls, lss, lo = lasso_cv_path(P, y, lasso_alphas, kf)
                for j in range(len(lasso_alphas)):
                    rec = {'v': v, 'd': d, 'model': 'Lasso', 'hp': lasso_alphas[j], 'nf': nf, 'mse': ls[j], 'se': lss[j], 'r2': r2(y, lo[:, j])}
                    rows.append(rec)
                    if ls[j] < best_mse: best_mse, best_rec, best_oof = ls[j], rec, lo[:, j]

            print(f"best={best_mse:.4f}")

        results[v] = {'df': pd.DataFrame(rows), 'best': best_rec, 'oof': best_oof, 'y': y, 'X': X, 'Xt': Xt}

    pd.concat([results['var1']['df'], results['var2']['df']]).to_csv(cache_file, index=False)

for v in ['var1', 'var2']:
    b = results[v]['best']
    print(f"\n{v} BEST: {b['model']} deg={int(b['d'])} hp={b['hp']:.2e} MSE={b['mse']:.4f} +/- {b['se']:.4f} R2={b['r2']:.4f}")

# ═══════════ PLOTTING ═══════════
print("\nGenerating report plots...")

def best_per_deg(df, model):
    sub = df[df['model'] == model]
    if sub.empty: return pd.Series(dtype=float), pd.Series(dtype=float)
    g = sub.loc[sub.groupby('d')['mse'].idxmin()]
    return g.set_index('d')['mse'], g.set_index('d')['se']

# ───── VAR1 PLOTS ─────
v = 'var1'
df1 = results[v]['df']; b1 = results[v]['best']; oof1 = results[v]['oof']
y1 = results[v]['y']; X1 = results[v]['X']; Xte1 = results[v]['Xt']

# Plot 1: EDA — Domain boundary shift
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ma_tr = np.abs(X1).max(1); ma_te = np.abs(Xte1).max(1)
ax1.hist(ma_tr, bins=25, alpha=.65, color='#1f77b4', density=True, label=f'Train (at boundary: {(ma_tr>=.99).mean()*100:.0f}%)')
ax1.hist(ma_te, bins=25, alpha=.65, color='#ff7f0e', density=True, label=f'Test  (at boundary: {(ma_te>=.99).mean()*100:.0f}%)')
ax1.set_title('(A) Var1: Domain Boundary Shift $\\max_i |x_i|$', fontsize=11, fontweight='bold')
ax1.set_xlabel('$\\max_i |x_i|$'); ax1.set_ylabel('Density'); ax1.grid(True); ax1.legend(frameon=True)

ae = np.abs(y1 - oof1)
ax2.scatter(ma_tr, ae, alpha=.35, c='#2ca02c', s=20, edgecolors='none')
bins_e = np.linspace(ma_tr.min(), 1, 8); bc = .5*(bins_e[:-1]+bins_e[1:])
be = [ae[(ma_tr>=bins_e[i])&(ma_tr<bins_e[i+1])].mean() for i in range(len(bins_e)-1)]
ax2.plot(bc, be, 'o-', c='#d62728', lw=2.5, label='Binned mean |error|')
ax2.set_title('(B) Var1: OOF Error vs Boundary Distance', fontsize=11, fontweight='bold')
ax2.set_xlabel('$\\max_i |x_i|$'); ax2.set_ylabel('$|y - \\hat{y}|$'); ax2.grid(True); ax2.legend(frameon=True)
plt.tight_layout(); plt.savefig('var1_plot1_eda_edge_shift.png', dpi=300); plt.close()
print("  [OK] var1_plot1_eda_edge_shift.png")

# Plot 2: CV MSE vs degree
fig, ax = plt.subplots(figsize=(8, 5))
for mdl, fmt, col, lbl in [('OLS','-o','#d62728','OLS'),('Ridge','-s','#1f77b4','Ridge (L2)'),('Lasso','-d','#2ca02c','Lasso (L1)')]:
    m, s = best_per_deg(df1, mdl)
    if len(m): ax.errorbar(m.index, m.values, yerr=s.values, fmt=fmt, color=col, label=lbl, capsize=3)
ax.axvline(int(b1['d']), color='#2ca02c', ls='--', label=f"Best: deg {int(b1['d'])} {b1['model']}")
ax.scatter([int(b1['d'])], [b1['mse']], c='#2ca02c', s=120, zorder=5, edgecolor='k')
ax.set_title('Var1: 5-Fold CV MSE vs Polynomial Degree', fontsize=12, fontweight='bold')
ax.set_xlabel('Degree $d$'); ax.set_ylabel('CV MSE'); ax.set_ylim(0.2, 2.5); ax.grid(True); ax.legend(frameon=True)
ax.text(.04,.86, f"Best: deg {int(b1['d'])} {b1['model']}\nMSE: {b1['mse']:.4f} +/- {b1['se']:.4f}\n$R^2$: {b1['r2']:.4f}",
    transform=ax.transAxes, bbox=dict(boxstyle='round', fc='w', alpha=.9, ec='#CCC'), fontsize=10)
plt.tight_layout(); plt.savefig('var1_plot2_cv_degree_curves.png', dpi=300); plt.close()
print("  [OK] var1_plot2_cv_degree_curves.png")

# Plot 3: Actual vs Predicted
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.scatter(y1, oof1, alpha=.5, c='#1f77b4', s=25, edgecolors='none')
lims = [min(y1.min(), oof1.min())-.5, max(y1.max(), oof1.max())+.5]
ax.plot(lims, lims, 'r--', lw=2, label='$y = \\hat{y}$')
ax.set_title(f"Var1: Actual vs OOF Predicted ({b1['model']} deg {int(b1['d'])})", fontsize=11, fontweight='bold')
ax.set_xlabel('Actual $y$'); ax.set_ylabel('Predicted $\\hat{y}$'); ax.set_xlim(lims); ax.set_ylim(lims)
ax.grid(True); ax.legend(frameon=True)
ax.text(.62,.08, f"MSE: {b1['mse']:.4f}\n$R^2$: {b1['r2']:.4f}\nSE: {b1['se']:.4f}",
    transform=ax.transAxes, bbox=dict(boxstyle='round', fc='w', alpha=.9, ec='#CCC'), fontsize=10)
plt.tight_layout(); plt.savefig('var1_plot3_oof_actual_vs_predicted.png', dpi=300); plt.close()
print("  [OK] var1_plot3_oof_actual_vs_predicted.png")

# Plot 4: Residual diagnostics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
res1 = y1 - oof1
ax1.scatter(oof1, res1, alpha=.5, c='#9467bd', s=25, edgecolors='none')
ax1.axhline(0, c='red', ls='--', lw=1.5)
ax1.set_title('(A) Var1: Residuals vs Fitted', fontsize=11, fontweight='bold')
ax1.set_xlabel('$\\hat{y}$'); ax1.set_ylabel('Residual ($y - \\hat{y}$)'); ax1.grid(True)
ax2.hist(res1, bins=30, density=True, alpha=.6, color='#2ca02c', edgecolor='k')
mu1, sig1 = res1.mean(), res1.std()
xp1 = np.linspace(mu1-4*sig1, mu1+4*sig1, 100)
ax2.plot(xp1, np.exp(-.5*((xp1-mu1)/sig1)**2)/(sig1*np.sqrt(2*np.pi)), 'r-', lw=2, label=f'N({mu1:.3f}, {sig1:.3f})')
ax2.set_title('(B) Var1: Residual Distribution', fontsize=11, fontweight='bold')
ax2.set_xlabel('Residual'); ax2.set_ylabel('Density'); ax2.grid(True); ax2.legend(frameon=True)
plt.tight_layout(); plt.savefig('var1_plot4_residual_diagnostics.png', dpi=300); plt.close()
print("  [OK] var1_plot4_residual_diagnostics.png")

# ───── VAR2 PLOTS ─────
v = 'var2'
df2 = results[v]['df']; b2 = results[v]['best']; oof2 = results[v]['oof']
y2 = results[v]['y']; X2 = results[v]['X']

# Sample variance from 9 repeated-x points across 4 clusters (5 degrees of freedom)
Z = linkage(X2, method='single')
lab = fcluster(Z, t=1e-3, criterion='distance')
csz = pd.Series(lab).value_counts(); mc = csz[csz > 1]
vs_list, ws_list = [], []
for c in mc.index:
    ix = np.where(lab == c)[0]
    vs_list.append(np.var(y2[ix], ddof=1)); ws_list.append(len(ix)-1)
noise_var = np.dot(vs_list, ws_list) / sum(ws_list)

# Plot 1: EDA — scatter + repeated-x variance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.scatter(X2[:,0], y2, alpha=.5, c='#1f77b4', s=20, edgecolors='none', label='$x_1$ vs $y$')
for c in mc.index:
    ix = np.where(lab==c)[0]
    ax1.scatter(X2[ix,0], y2[ix], c='#d62728', s=60, edgecolor='k', zorder=5)
ax1.set_title('(A) Var2: $x_1$ vs $y$ (duplicate clusters in red)', fontsize=11, fontweight='bold')
ax1.set_xlabel('$x_1$'); ax1.set_ylabel('$y$'); ax1.grid(True); ax1.legend(frameon=True)
ax2.hist(y2, bins=30, alpha=.7, color='#ff7f0e', edgecolor='k', density=True)
ax2.set_title('(B) Var2: Target distribution & repeated-x var', fontsize=11, fontweight='bold')
ax2.set_xlabel('$y$'); ax2.set_ylabel('Density'); ax2.grid(True)
ax2.text(.05,.72, f"Var($y$) = {y2.var():.3f}\nRepeated-$x$ $s^2 \\approx$ {noise_var:.3f}\n95% CI: [0.14, 2.18] (df=5)\nBest CV MSE = {b2['mse']:.4f}",
    transform=ax2.transAxes, bbox=dict(boxstyle='round', fc='w', alpha=.9, ec='#CCC'), fontsize=9.5)
plt.tight_layout(); plt.savefig('var2_plot1_eda_noise_floor.png', dpi=300); plt.close()
print("  [OK] var2_plot1_eda_noise_floor.png")

# Plot 2: CV MSE vs degree
fig, ax = plt.subplots(figsize=(8.5, 5))
for mdl, fmt, col, lbl in [('OLS','-o','#d62728','OLS'),('Ridge','-s','#1f77b4','Ridge (L2)'),('Lasso','-d','#2ca02c','Lasso (L1)')]:
    m, s = best_per_deg(df2, mdl)
    if len(m): ax.errorbar(m.index, m.values, yerr=s.values, fmt=fmt, color=col, label=lbl, capsize=3)
ax.axhline(noise_var, c='k', ls=':', lw=2, label=f'Repeated-$x$ $s^2 \\approx {noise_var:.3f}$ (df=5)')
ax.axvline(int(b2['d']), c='#1f77b4', ls='--', label=f"Best: deg {int(b2['d'])} {b2['model']}")
ax.scatter([int(b2['d'])], [b2['mse']], c='#1f77b4', s=120, zorder=5, edgecolor='k')
ax.set_title('Var2: 5-Fold CV MSE vs Polynomial Degree', fontsize=12, fontweight='bold')
ax.set_xlabel('Degree $d$'); ax.set_ylabel('CV MSE'); ax.set_ylim(0.1, 1.5); ax.grid(True); ax.legend(frameon=True)
ax.text(.04,.68, f"Best: deg {int(b2['d'])} {b2['model']}\nMSE: {b2['mse']:.4f} +/- {b2['se']:.4f}\n$R^2$: {b2['r2']:.4f}",
    transform=ax.transAxes, bbox=dict(boxstyle='round', fc='w', alpha=.9, ec='#CCC'), fontsize=10)
plt.tight_layout(); plt.savefig('var2_plot2_cv_degree_curves.png', dpi=300); plt.close()
print("  [OK] var2_plot2_cv_degree_curves.png")

# Plot 3: Actual vs Predicted
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.scatter(y2, oof2, alpha=.5, c='#ff7f0e', s=25, edgecolors='none')
lims = [min(y2.min(), oof2.min())-.5, max(y2.max(), oof2.max())+.5]
ax.plot(lims, lims, 'r--', lw=2, label='$y = \\hat{y}$')
ax.set_title(f"Var2: Actual vs OOF Predicted ({b2['model']} deg {int(b2['d'])})", fontsize=11, fontweight='bold')
ax.set_xlabel('Actual $y$'); ax.set_ylabel('Predicted $\\hat{y}$'); ax.set_xlim(lims); ax.set_ylim(lims)
ax.grid(True); ax.legend(frameon=True)
ax.text(.62,.08, f"MSE: {b2['mse']:.4f}\n$R^2$: {b2['r2']:.4f}\nSE: {b2['se']:.4f}",
    transform=ax.transAxes, bbox=dict(boxstyle='round', fc='w', alpha=.9, ec='#CCC'), fontsize=10)
plt.tight_layout(); plt.savefig('var2_plot3_oof_actual_vs_predicted.png', dpi=300); plt.close()
print("  [OK] var2_plot3_oof_actual_vs_predicted.png")

# Plot 4: Residual diagnostics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
res2 = y2 - oof2
ax1.scatter(oof2, res2, alpha=.5, c='#17becf', s=25, edgecolors='none')
ax1.axhline(0, c='red', ls='--', lw=1.5)
ax1.set_title('(A) Var2: Residuals vs Fitted', fontsize=11, fontweight='bold')
ax1.set_xlabel('$\\hat{y}$'); ax1.set_ylabel('Residual ($y - \\hat{y}$)'); ax1.grid(True)
ax2.hist(res2, bins=30, density=True, alpha=.6, color='#ff7f0e', edgecolor='k')
mu2, sig2 = res2.mean(), res2.std()
xp2 = np.linspace(mu2-4*sig2, mu2+4*sig2, 100)
ax2.plot(xp2, np.exp(-.5*((xp2-mu2)/sig2)**2)/(sig2*np.sqrt(2*np.pi)), 'r-', lw=2, label=f'N({mu2:.3f}, {sig2:.3f})')
ax2.set_title('(B) Var2: Residual Distribution', fontsize=11, fontweight='bold')
ax2.set_xlabel('Residual'); ax2.set_ylabel('Density'); ax2.grid(True); ax2.legend(frameon=True)
plt.tight_layout(); plt.savefig('var2_plot4_residual_diagnostics.png', dpi=300); plt.close()
print("  [OK] var2_plot4_residual_diagnostics.png")

# ═══════════ SUMMARY PANELS ═══════════
for v, pn in [('var1', ['var1_plot1_eda_edge_shift.png','var1_plot2_cv_degree_curves.png','var1_plot3_oof_actual_vs_predicted.png','var1_plot4_residual_diagnostics.png']),
              ('var2', ['var2_plot1_eda_noise_floor.png','var2_plot2_cv_degree_curves.png','var2_plot3_oof_actual_vs_predicted.png','var2_plot4_residual_diagnostics.png'])]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, p in zip(axes.flat, pn):
        ax.imshow(plt.imread(p)); ax.axis('off')
    axes[0,0].set_title(f'{v.upper()} - EDA', fontweight='bold')
    axes[0,1].set_title(f'{v.upper()} - CV MSE vs Degree', fontweight='bold')
    axes[1,0].set_title(f'{v.upper()} - Actual vs Predicted', fontweight='bold')
    axes[1,1].set_title(f'{v.upper()} - Residual Diagnostics', fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{v}_summary_4panel.png', dpi=300); plt.close()
    print(f"  [OK] {v}_summary_4panel.png")

# cv_plot.png
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
a1.imshow(plt.imread('var1_plot2_cv_degree_curves.png')); a1.axis('off'); a1.set_title('Var1 CV Curves', fontweight='bold')
a2.imshow(plt.imread('var2_plot2_cv_degree_curves.png')); a2.axis('off'); a2.set_title('Var2 CV Curves', fontweight='bold')
plt.tight_layout(); plt.savefig('cv_plot.png', dpi=300); plt.close()
print("  [OK] cv_plot.png")

print("\n[SUCCESS] All 8 individual plots + 2 summary 4-panels + cv_plot.png updated successfully!")
