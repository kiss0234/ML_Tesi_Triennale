from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted


class LogMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, log_transform: str = "log1p", offset: float = 1e-9) -> None:
        self.log_transform: str = log_transform
        self.offset: float = offset
        self.scaler: MinMaxScaler = MinMaxScaler()

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "LogMinMaxScaler":
        X = np.asarray(X)
        if self.log_transform == "log1p":
            X_log = np.log1p(X)
        else:
            X_log = np.log(X + self.offset)
        self.scaler.fit(X_log)

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        check_is_fitted(self.scaler)
        X = np.asarray(X)
        if self.log_transform == "log1p":
            X_log = np.log1p(X)
        else:
            X_log = np.log(X + self.offset)
        return self.scaler.transform(X_log)

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = []
        return np.array(input_features)