import numpy as np, warnings
from scipy import sparse
import warnings
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from cca_zoo.linear import ElasticCCA
from joblib import Parallel, delayed
import logging
log = logging.getLogger("scca.auto_tuning")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


# --- utils ---
def fisher_z(r):
    """Fisher z on correlations to allow meaningful averaging across folds."""
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

def inv_fisher_z(z):
    return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)

def standardize_infold(Xtr, Xte):
    sc = StandardScaler(with_mean=True, with_std=True)
    return sc.fit_transform(Xtr), sc.transform(Xte)

def make_scaled_folds(X, Y, n_splits, random_state=0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for tr, te in kf.split(X):
        Xtr, Xte = X[tr], X[te]
        Ytr, Yte = Y[tr], Y[te]
        Xtrz, Xtez = standardize_infold(Xtr, Xte)
        Ytrz, Ytez = standardize_infold(Ytr, Yte)
        folds.append((Xtrz, Xtez, Ytrz, Ytez))
    return folds

def eval_combo_cached(folds, params, epochs=120, tol=1e-3, random_state=0):
    z_vals = []
    for (Xtrz, Xtez, Ytrz, Ytez) in folds:
        m = ElasticCCA(
            latent_dimensions=1,
            alpha=[params["alpha_x"], params["alpha_y"]],
            l1_ratio=[params["l1_x"], params["l1_y"]],
            initialization="pls",
            epochs=epochs, tol=tol,
            random_state=random_state, verbose=False
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", r".*Ill-conditioned matrix.*")
            try:
                m.fit((Xtrz, Ytrz))
                Zx, Zy = m.transform((Xtez, Ytez))
            except ValueError:
                # just skip this fold
                continue

        sx, sy = np.std(Zx[:, 0]), np.std(Zy[:, 0])
        if not np.isfinite(sx) or not np.isfinite(sy) or sx == 0 or sy == 0:
            # skip degenerate fold
            continue

        r = float(np.corrcoef(Zx[:, 0], Zy[:, 0])[0, 1])
        if np.isfinite(r):
            z_vals.append(fisher_z(abs(r)))

    if not z_vals:
        return None

    z_mean = float(np.mean(z_vals))
    return {
        "mean_r": float(inv_fisher_z(z_mean)),
        "mean_z": z_mean,
        "params": params,
        "folds": len(z_vals),
    }

def top_variance(A, k=None):
    if k is None or A.shape[1] <= k: 
        return A, np.arange(A.shape[1])
    v = np.var(A, axis=0)
    keep = np.argsort(v)[::-1][:k]
    return A[:, keep], keep

def maybe_to_dense(A, mem_limit_mb=8000):
    """Densify safely if sparse. Estimate memory before converting."""
    if hasattr(A, "values"): A = A.values
    if sparse.issparse(A):
        est_mb = (A.shape[0] * A.shape[1] * 8) / (1024**2)
        if est_mb > mem_limit_mb:
            raise MemoryError(f"Densifying would use ~{est_mb:.0f} MB > limit={mem_limit_mb} MB")
        return A.toarray()
    return A

# --- single-combo CV evaluator (uses Fisher-z mean) ---
def eval_combo_ElasticCCA(X, Y, n_splits, params, epochs=400, tol=1e-4, random_state=0):  
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    z_vals = []
    failed = 0
    Fold = 1
    max_failed_folds = max(1, n_splits // 2)  # if > half the folds fail, discard params
    
    for tr, te in kf.split(X):
        Fold += 1
        log.info("Fold %s | alpha_x=%.4f, alpha_y=%.4f, l1_x=%.2f, l1_y=%.2f",
                 Fold, params["alpha_x"], params["alpha_y"], params["l1_x"], params["l1_y"])

        Xtr, Xte = X[tr], X[te]
        Ytr, Yte = Y[tr], Y[te]
        Xtrz, Xtez = standardize_infold(Xtr, Xte)
        Ytrz, Ytez = standardize_infold(Ytr, Yte)
            
        m = ElasticCCA(
            latent_dimensions=1,
            alpha=[params["alpha_x"], params["alpha_y"]],
            l1_ratio=[params["l1_x"], params["l1_y"]],
            initialization="pls", epochs=epochs, tol=tol,
            random_state=random_state, verbose=False
        )
    
        try:
            # Suppress only convergence warnings during fit/transform
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                m.fit((Xtrz, Ytrz))
                Zx, Zy = m.transform((Xtez, Ytez))
        except ValueError as e:
            # Typical when Y target becomes NaN (over-sparsified view)
            failed += 1
            if failed > max_failed_folds:
                log.info(f"Parameters failed across {failed} folds. Rejecting parameter set.")
                return None
            continue
            
        # skip if a score is constant/degenerate
        sx, sy = np.std(Zx[:, 0]), np.std(Zy[:, 0])
        if not np.isfinite(sx) or not np.isfinite(sy) or sx == 0 or sy == 0:
            log.info(f"Score is constant/degenerate. Skipping...")
            continue

        r = float(np.corrcoef(Zx[:, 0], Zy[:, 0])[0, 1])
        if np.isfinite(r):
            z_vals.append(fisher_z(abs(r)))
            
    if not z_vals:
        return None  # failed
    z_mean = float(np.mean(z_vals))
    r_mean = float(inv_fisher_z(z_mean))
    return {"mean_r": r_mean, "mean_z": z_mean, "params": params, "folds": len(z_vals)}


# --- main adaptive interface ---
def auto_scca(
    X, Y, mem_limit_mb=8000, random_state=0
):
    
    # Extract X and Y sample / feature count
    n, pX, pY = X.shape[0], X.shape[1], Y.shape[1]

    # Densify safely (estimate first)
    X = maybe_to_dense(X, mem_limit_mb=mem_limit_mb)
    Y = maybe_to_dense(Y, mem_limit_mb=mem_limit_mb)
    
    # Provide a broad l1_x grid including 0.01 (very light sparsity) through 0.8 (quite sparse).
    l1x_grid   = [0.01, 0.2, 0.4, 0.6, 0.8]
    
    # Keep up to 3000 highest-variance features in X to stabilize and speed up.
    X_use, _ = top_variance(X, min(3000, pX))
    
    # Do 5 splits unless n < 120, then do 3
    
    n_splits = 5 if n < 120 else 3
    
    # Regimes guided by size of Y
    if pY <= 50:
        regime = "A_smallY" 
        log.info(f"{pY} <= 50: Choosing A_smallY regime (No L1 restraint or variance filtering for Y)")
        
        # grids
        l1y_grid   = [0.0]   # all Ridge on Y
        Y_use = Y

    elif pY <= 300:
        regime = "B_mediumY"
        log.info(f"{pY} > 50 but <= 300: Choosing B_mediumY regime (Light L1 restraint but no variance filtering for Y)")
        l1y_grid   = [0.0, 0.1, 0.2]
        Y_use = Y
    
    else:
        regime = "C_largeY"
        log.info(f"{pY} >= 300: Choosing C_largeY regime (Coarse L1 restraint and variance filtering with up to 2000 features for Y)")
        l1y_grid   = [0.0, 0.2, 0.4, 0.6, 0.8]  # coarse on Y
        Y_use, _ = top_variance(Y, min(2000, pY))
    
    # Cast to float32 before CV to cut memory roughly in half
    
    X_use = X_use.astype(np.float32, copy=False)
    Y_use = Y_use.astype(np.float32, copy=False)
    
    # Build folds and scale within each fold to prevent leakage 
    folds = make_scaled_folds(X_use, Y_use, n_splits, random_state)

    # Stage A: random search (parallel)
    log.info("Starting Stage A: random search")
    rng = np.random.default_rng(random_state)
    
    # Define number of hyperparameter combinations 
    R = 32 if pY <= 300 else 64  # fewer for small/medium, more for large

    # Build a list of random hyperparam combos
    param_list_A = [{
        # Selects ridge-like penalty between 0.03 and 1.0 continuously and log-uniformly to sample smaller alpha values more often
        "alpha_x": float(10**rng.uniform(np.log10(0.03), 0.0)), 
        "alpha_y": float(10**rng.uniform(np.log10(0.03), 0.0)), 
        # Selects lasso penalty at random 
        "l1_x":    float(rng.choice([0.01, 0.2, 0.4, 0.6, 0.8])), 
        "l1_y":    float(rng.choice(l1y_grid)),
    } for _ in range(R)]

    # Define and run function for parallel computing
    def eval_params_list(folds, param_list, epochs, tol, random_state, n_jobs=-1):
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(eval_combo_cached)(folds, p, epochs=epochs, tol=tol, random_state=random_state)
            for p in param_list
        )
        return [r for r in results if r is not None]

    cand_A = eval_params_list(
        folds, param_list_A, epochs=120, tol=1e-3,
        random_state=random_state, n_jobs=-1
    )

    # If all combinations failed, then rerun again with Ridge penality only
    if not cand_A:
        log.info("No good candidates found, trying Ridge only...")
        ridge_params = [{
            "alpha_x": a, "alpha_y": a, "l1_x": 0.0, "l1_y": 0.0
        } for a in np.logspace(-2, 0, 5)]
        cand_A = eval_params_list(folds, ridge_params, epochs=120, tol=1e-3,
                                  random_state=random_state, n_jobs=-1)
        if not cand_A:
            return {
                "regime": regime, "n_splits": n_splits,
                "best": None, "chosen": None,
                "stageA_count": 0, "stageB_count": 0,
                "note": "No viable models after sparsity/ridge retries. Try reducing features or raising alpha bounds."
            }
    
    topA = sorted(cand_A, key=lambda d: d['mean_r'], reverse=True)[:min(8, len(cand_A))]

    # Stage B: refine around top with small neighborhood and 1-SE rule
    log.info("Starting Stage B: refine around top with small neighborhood and 1-SE rule")
    alphas = np.logspace(-3, 0, 7)
    def neigh(val, grid):
        i = np.argmin(np.abs(grid - val))
        lo, hi = max(i-1,0), min(i+1, len(grid)-1)
        return np.unique(grid[lo:hi+1])

    cand_B = []
    for t in topA:
        axN = neigh(t["params"]["alpha_x"], alphas)
        ayN = neigh(t["params"]["alpha_y"], alphas)
        lxN = neigh(t["params"]["l1_x"], np.array(l1x_grid))
        lyN = neigh(t["params"]["l1_y"], np.array(l1y_grid))
        for ax in axN:
            for ay in ayN:
                for lx in lxN:
                    for ly in lyN:
                        params = {"alpha_x": float(ax), "alpha_y": float(ay), "l1_x": float(lx), "l1_y": float(ly)}
                        out = eval_combo_ElasticCCA(X_use, Y_use, n_splits=n_splits, params=params, epochs=220, tol=8e-4, random_state=random_state)
                        if out is not None:
                            cand_B.append(out)

    cand_B.sort(key=lambda d: d["mean_r"], reverse=True)
    
    if not cand_B:
        # fall back to best from Stage A
        best = sorted(cand_A, key=lambda d: d["mean_r"], reverse=True)[0]
        chosen = best
        return {
            "regime": regime, "n_splits": n_splits,
            "best": best, "chosen": chosen,
            "stageA_count": len(cand_A), "stageB_count": 0,
            "note": "Refine stage produced no candidates; returning Stage A best."
        }

    best = cand_B[0]
    
    # 1-SE threshold
    scores = np.array([d["mean_r"] for d in cand_B])
    se = np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0
    thresh = best["mean_r"] - se
    eligible = [d for d in cand_B if d["mean_r"] >= thresh]

    # Sparsest preference: higher l1_x, then higher (alpha_x+alpha_y)
    eligible.sort(key=lambda d: (d["params"]["l1_x"], d["params"]["alpha_x"]+d["params"]["alpha_y"]), reverse=True)
    chosen = eligible[0]

    return {
        "regime": regime,
        "n_splits": n_splits,
        "best": best,
        "chosen": chosen,
        "stageA_count": len(cand_A),
        "stageB_count": len(cand_B)
    }
