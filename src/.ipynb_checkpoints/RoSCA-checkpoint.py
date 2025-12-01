#!/usr/bin/env python
# sCCA.py
import json
import argparse
import numpy as np
import pandas as pd
from biom import load_table
from preprocessing import preprocess
from auto_tuning import auto_scca  
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from cca_zoo.linear import ElasticCCA
import logging; logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scca")

def load_input(path):
    log.info("Loading data from %s", path)
    if path.endswith((".biom", ".h5", ".hdf5")):
        return load_table(path)  # BIOM Table
    # else assume TSV/CSV samples x features
    df = pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",", index_col=0)
    # wrap to a BIOM-like interface
    from biom import Table
    return Table(df.T.values, df.columns.astype(str), df.index.astype(str))

def cross_validated_scores_and_loadings(X, Y, params, n_splits=5, random_state=0):
    n = X.shape[0]
    Uo = np.zeros(n)
    Vo = np.zeros(n)
    
    Lx_folds = []
    Ly_folds = [] 

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, te in kf.split(X):
        # scale within each fold
        sx = StandardScaler().fit(X[tr])
        sy = StandardScaler().fit(Y[tr])
        Xtrz = sx.transform(X[tr]); Xtez = sx.transform(X[te])
        Ytrz = sy.transform(Y[tr]); Ytez = sy.transform(Y[te])

        model = ElasticCCA(
            latent_dimensions=1,
            alpha=[params["alpha_x"], params["alpha_y"]],
            l1_ratio=[params["l1_x"], params["l1_y"]],
            initialization="pls",
            epochs=800, tol=1e-4,
            random_state=random_state, verbose=False
        ).fit((Xtrz, Ytrz))
        
        # Out-of-fold scores for scatter / CV correlation
        U_te, V_te = model.transform((Xtez, Ytez))
        Uo[te] = U_te[:, 0]
        Vo[te] = V_te[:, 0]
        
        # Fold-wise loadings on training data
        Lx_fold, Ly_fold = model.canonical_loadings_((Xtrz, Ytrz), normalize=True)
        Lx_folds.append(Lx_fold[:, 0])
        Ly_folds.append(Ly_fold[:, 0])
    
    # Stack into (n_folds, p) arrays
    Lx_folds = np.vstack(Lx_folds)
    Ly_folds = np.vstack(Ly_folds)
    
    # --- Sign alignment across folds ---
    Lx_ref = Lx_folds[0]
    for i in range(1, Lx_folds.shape[0]):
        if np.corrcoef(Lx_ref, Lx_folds[i])[0,1] < 0:
            Lx_folds[i] *= -1

    Ly_ref = Ly_folds[0]
    for i in range(1, Ly_folds.shape[0]):
        if np.corrcoef(Ly_ref, Ly_folds[i])[0,1] < 0:
            Ly_folds[i] *= -1

    # Median loadings across folds
    Lx_median = np.median(Lx_folds, axis=0)
    Ly_median = np.median(Ly_folds, axis=0)

    return Uo, Vo, Lx_median, Ly_median

def main():
    ap = argparse.ArgumentParser(description="Robust Sparse CCA pipeline for microbiome data")
    ap.add_argument("--X", required=True, help="Path to X (BIOM/TSV/CSV)")
    ap.add_argument("--Y", required=True, help="Path to Y (BIOM/TSV/CSV)")
    ap.add_argument("--y-compositional", action="store_true", default=False,
                    help="Applies compositional transform (rCLR + matrix completion) to Y")
    ap.add_argument("--min-sample-count", type=int, default=0)
    ap.add_argument("--min-feature-count", type=int, default=0)
    ap.add_argument("--min-feature-frequency", type=float, default=0)
    ap.add_argument("--mem-limit-mb", type=int, default=16000)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--out-directory", required=True)
    args = ap.parse_args()

    X_tbl = load_input(args.X)
    Y_tbl = load_input(args.Y)
    
    log.info("Running preprocess...")
    Y_compositional=args.y_compositional
    
    # Preprocess data using Gemelli's RPCA function 
    Xn, Yn, sample_ids, x_feat_ids, y_feat_ids = preprocess(
        X_tbl, Y_tbl,
        min_sample_count=args.min_sample_count,
        min_feature_count=args.min_feature_count,
        min_feature_frequency=args.min_feature_frequency,
        Y_compositional=Y_compositional
    )
    
    if not np.isfinite(Xn).all() or not np.isfinite(Yn).all():
        raise ValueError("X_use/Y_use contain NaN/inf. Clean data before auto_scca.")
        
    log.info("Running adaptive sCCA tuner...")

    # Run adaptive sCCA tuner 
    out = auto_scca(
        X=Xn, Y=Yn,
        mem_limit_mb=args.mem_limit_mb,
        random_state=args.random_state
    )

    # Save JSON summary
    with open(f"{args.out_directory}/summary.json", "w") as fh:
        json.dump(out, fh, indent=2)
    
    log.info("Refitting chosen model on full data...")
    # Refit the chosen model on full data and save weights/scores
    try:
        params = out["chosen"]["params"]
        
        # Running cross-validated model and saving scores/loadings...
        U_cv, V_cv, Lx_median, Ly_median = cross_validated_scores_and_loadings(Xn, Yn, out["chosen"]["params"], n_splits=out["n_splits"])
        r_cv = np.corrcoef(U_cv, V_cv)[0,1]
        print("Cross-validated r:", r_cv)

        # Save weights with feature names
        pd.Series(Lx_median, index=x_feat_ids, name="Lx").to_csv(f"{args.out_directory}/Lx.tsv", sep="\t")
        pd.Series(Ly_median, index=y_feat_ids, name="Ly").to_csv(f"{args.out_directory}/Ly.tsv", sep="\t")
        
        # Save scores with sample ids
        pd.DataFrame({"U1": U_cv, "V1": V_cv}, index=sample_ids).to_csv(f"{args.out_directory}/scores.tsv", sep="\t")
    except Exception as e:
        with open(f"{args.out_directory}/fit_warning.txt", "w") as fh:
            fh.write(f"Final fit failed or was skipped: {e}\n")

if __name__ == "__main__":
    main()
