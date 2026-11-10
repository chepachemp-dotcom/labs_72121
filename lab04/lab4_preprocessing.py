"""Stateful cleaner. Everything learned (modes, medians, clip bounds) comes from fit() only,
so fitting on train and calling transform() on test cannot leak test information."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DERIVED_COLS = ["capital_net", "hours_per_age"]


class Cleaner(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols, cat_cols, clip_q=(0.01, 0.99), add_derived=False, missing_token="?"):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.clip_q = clip_q
        self.add_derived = add_derived
        self.missing_token = missing_token

    def _mark_missing(self, X):
        X = X.copy()
        for c in self.cat_cols:
            X[c] = X[c].astype(object).replace(self.missing_token, np.nan)
        return X

    def fit(self, X, y=None):
        X = self._mark_missing(X)
        self.medians_ = X[self.num_cols].astype(float).median()
        self.modes_ = {c: X[c].mode(dropna=True).iloc[0] for c in self.cat_cols}
        filled = X[self.num_cols].astype(float).fillna(self.medians_)
        self.lower_ = filled.quantile(self.clip_q[0])
        self.upper_ = filled.quantile(self.clip_q[1])
        return self

    def transform(self, X):
        X = self._mark_missing(X)
        for c in self.cat_cols:
            X[c] = X[c].fillna(self.modes_[c])
        num = X[self.num_cols].astype(float).fillna(self.medians_)
        X[self.num_cols] = num.clip(self.lower_, self.upper_, axis=1)
        if self.add_derived:  # stateless, row-wise -> no leakage
            X["capital_net"] = X["capital-gain"] - X["capital-loss"]
            X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16).clip(lower=1)
        return X