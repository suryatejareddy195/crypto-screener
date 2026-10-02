#!/usr/bin/env python3
"""mlutils.py - importable helper classes used by ML artifacts.

Classes that get joblib-pickled into model artifacts MUST live in an
importable module (not under ``__main__``), otherwise ``joblib.load`` from
another process (app.py) fails. Keep calibration helpers here.

``IsotonicMulticlass`` is retained for backward compatibility: schema-1
(multiclass) artifacts pickled before the binary rework reference it. New
schema-2 artifacts use :class:`IsotonicBinary`.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import IsotonicRegression


class StandardScalerImp:
    """Minimal standard scaler used inside logistic pipelines in artifacts.

    Lives here (not in train_model.py) because joblib-pickled classes MUST
    live in an importable module - a class defined in a script is pickled as
    ``__main__.ClassName`` and fails to load from another process (app.py).
    Avoids depending on sklearn's internal attribute layout across versions.
    """

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.nan_to_num((X - self.mean_) / self.scale_, nan=0.0)

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X).transform(X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "mean_")


class IsotonicBinary:
    """Isotonic calibration for a single binary probability.

    Fitted on walk-forward out-of-fold probabilities (never on the held-out
    test set) so the deployed "70% confidence" claim is anchored to a real
    measured curve. Monotone, clips out-of-bounds, and floors values at
    1e-6 so a zero probability never fully suppresses downstream use.
    """

    def __init__(self) -> None:
        self.regressor_: IsotonicRegression | None = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicBinary":
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=float)
        iso = IsotonicRegression(out_of_bounds="clip", increasing="auto")
        iso.fit(p, y)
        self.regressor_ = iso
        return self

    def predict(self, p: np.ndarray | float) -> np.ndarray:
        if self.regressor_ is None:
            raise RuntimeError("IsotonicBinary.fit must be called first")
        p = np.asarray(p, dtype=float)
        out = self.regressor_.predict(p)
        return np.clip(out, 1e-6, 1.0)


class IsotonicMulticlass:
    """Per-class isotonic regression wrapper (monotone, clips out-of-bounds).

    Fitted on walk-forward out-of-fold probabilities (never on the held-out
    test set). ``predict`` applies each class curve then renormalises.

    Kept for schema-1 artifact compatibility; new training uses
    :class:`IsotonicBinary` per direction.
    """

    def __init__(self) -> None:
        self.regressors_: list[IsotonicRegression] = []

    def fit(self, P: np.ndarray, y: np.ndarray) -> "IsotonicMulticlass":
        self.regressors_ = []
        for c in range(P.shape[1]):
            iso = IsotonicRegression(out_of_bounds="clip", increasing="auto")
            iso.fit(P[:, c], (y == c).astype(float))
            self.regressors_.append(iso)
        return self

    def predict(self, P: np.ndarray) -> np.ndarray:
        out = np.column_stack([r.predict(P[:, c])
                               for c, r in enumerate(self.regressors_)])
        out = np.clip(out, 1e-6, 1.0)
        return out / out.sum(axis=1, keepdims=True)
