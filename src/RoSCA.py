#!/usr/bin/env python
# sCCA.py
import json
import argparse
import numpy as np
import pandas as pd
from biom import load_table
from preprocessing import preprocess
from auto_tuning import auto_scca   
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
        from cca_zoo.linear import ElasticCCA
        params = out["chosen"]["params"]
        
        # Standardize on full data before final fit (train-only scaling is for CV; final model fits on all)
        from sklearn.preprocessing import StandardScaler
        sx = StandardScaler().fit(Xn); sy = StandardScaler().fit(Yn)
        Xz = sx.transform(Xn); Yz = sy.transform(Yn)

        model = ElasticCCA(
            latent_dimensions=1,
            alpha=[params["alpha_x"], params["alpha_y"]],
            l1_ratio=[params["l1_x"], params["l1_y"]],
            initialization="pls", epochs=800, tol=1e-4,
            random_state=args.random_state, verbose=False
        ).fit((Xz, Yz))

        U, V = model.transform((Xz, Yz))     # canonical variates (scores)
        Lx, Ly = model.canonical_loadings_((Xz, Yz), normalize=True)  # canonical weight vectors

        # Save weights with feature names
        pd.Series(Lx[:, 0], index=x_feat_ids, name="Lx").to_csv(f"{args.out_directory}/Lx.tsv", sep="\t")
        pd.Series(Ly[:, 0], index=y_feat_ids, name="Ly").to_csv(f"{args.out_directory}/Ly.tsv", sep="\t")
        
        # Save scores with sample ids
        pd.DataFrame({"U1": U[:, 0], "V1": V[:, 0]}, index=sample_ids).to_csv(f"{args.out_directory}/scores.tsv", sep="\t")
    except Exception as e:
        with open(f"{args.out_directory}/fit_warning.txt", "w") as fh:
            fh.write(f"Final fit failed or was skipped: {e}\n")

if __name__ == "__main__":
    main()
