#!/usr/bin/env python
# preprocessing.py
import sys
import numpy as np
import pandas as pd
from biom import Table
from gemelli.preprocessing import matrix_rclr
from gemelli.matrix_completion import MatrixCompletion
import logging
log = logging.getLogger("scca.preprocess")


def matrix_completion(ft: Table) -> Table:
    """rCLR + low-rank matrix completion (Gemelli) on a BIOM Table."""
    sample_ids = ft.ids(axis="sample")
    feature_ids = ft.ids(axis="observation")
    with np.errstate(divide='ignore'):
        # rCLR expects samples x features (rows=samples)
        rclr_tbl = matrix_rclr(ft.matrix_data.toarray().T)  # -> dense (n_samples, n_features)

    opt = MatrixCompletion(n_components=10, max_iterations=10).fit(rclr_tbl)
    # Wrap back to BIOM with original ids; transpose solution back to features x samples
    completed = Table(opt.solution.T, feature_ids, sample_ids)
    return completed


def _filter_with_rpca(table: Table,
                      min_sample_count: int,
                      min_feature_count: int,
                      min_feature_frequency: float) -> Table:
    """Use gemelli's rpca preprocessing filters on a BIOM Table."""
    # get shape of table
    n_features, n_samples = table.shape
    log.info(f"Initial table shape: {n_features} features × {n_samples} samples")

    # filter sample to min seq. depth
    def sample_filter(val, id_, md):
        return sum(val) > min_sample_count

    # filter features to min total counts
    def observation_filter(val, id_, md):
        return sum(val) > min_feature_count

    # filter features by N samples presence
    def frequency_filter(val, id_, md):
        return (np.sum(val > 0) / n_samples) > (min_feature_frequency / 100)

    # filter and import table for each filter above
    if min_feature_count is not None:
        pre_f = table.shape[0]
        table = table.filter(observation_filter,
                             axis='observation',
                             inplace=False)
        log.info(f"Filtered by min_feature_count>{min_feature_count}: "
         f"{pre_f} → {table.shape[0]} features")
    if min_feature_frequency is not None:
        pre_f = table.shape[0]
        table = table.filter(frequency_filter,
                             axis='observation',
                             inplace=False)
        log.info(f"Filtered by min_feature_frequency>{min_feature_frequency}%: "
         f"{pre_f} → {table.shape[0]} features")
    if min_sample_count is not None:
        pre_s = table.shape[1]
        table = table.filter(sample_filter,
                             axis='sample',
                             inplace=False)
        log.info(f"Filtered by min_sample_count>{min_sample_count}: "
         f"{pre_s} → {table.shape[1]} samples")

    # check the table after filtering
    if len(table.ids()) != len(set(table.ids())):
        raise ValueError('Data-table contains duplicate sample IDs')
    if len(table.ids('observation')) != len(set(table.ids('observation'))):
        raise ValueError('Data-table contains duplicate feature IDs')
    if min_sample_count is not None:
        # ensure empty samples / features are removed
        table = table.remove_empty(inplace=False)
        
    n_features_final, n_samples_final = table.shape
    log.info(f"Final table shape: {n_features_final} features × {n_samples_final} samples")

    return table


def preprocess(X_tbl: Table,
               Y_tbl: Table,
               min_sample_count: int = 1000,
               min_feature_count: int = 10,
               min_feature_frequency: float = 0.1,
               Y_compositional: bool = False):
    """
    Returns:
      Xn, Yn: np.ndarray (unstandardized), samples x features
      sample_ids, x_feature_ids, y_feature_ids
    """
    # 1) Filter each table (if compositional for Y)
    Xf = _filter_with_rpca(X_tbl, min_sample_count, min_feature_count, min_feature_frequency)
    Yf = _filter_with_rpca(Y_tbl, min_sample_count, min_feature_count, min_feature_frequency) if Y_compositional else Y_tbl.copy()

    # 2) Align shared samples
    shared = set(Xf.ids(axis="sample")).intersection(Yf.ids(axis="sample"))
    log.info("Number of shared samples: %d", len(shared))
    if len(shared) == 0:
        log.info("ValueError: No shared samples between X and Y", file=sys.stderr)
        sys.exit(2)

    # BIOM filter 
    Xf = Xf.filter(shared, axis='sample', inplace=False)
    Yf = Yf.filter(shared, axis='sample', inplace=False)

    # 3) Compositional transforms (rCLR + Matrix completion)
    Xr = matrix_completion(Xf)
    Yr = matrix_completion(Yf) if Y_compositional else Yf.copy()

    # 4) To pandas (dense), orient samples as rows
    X_df = Xr.to_dataframe(dense=True).T.copy()
    Y_df = Yr.to_dataframe(dense=True).T.copy()

    # 5) Drop zero-variance columns (prevents degenerate features)
    X_df = X_df.loc[:, X_df.var(axis=0) > 0.0]
    Y_df = Y_df.loc[:, Y_df.var(axis=0) > 0.0]

    # 6) Return arrays
    Xn = X_df.values.astype(float, copy=False)
    Yn = Y_df.values.astype(float, copy=False)

    return Xn, Yn, list(X_df.index), list(X_df.columns), list(Y_df.columns)
