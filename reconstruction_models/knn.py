"""
K-Nearest Neighbors Imputation
Uses KNN algorithm to impute missing values based on neighboring values.
"""

import pandas as pd
from sklearn.impute import KNNImputer


def knn_impute(data: pd.Series, n_neighbors: int = 3) -> pd.Series:
    """
    Impute missing values using K-Nearest Neighbors.
    
    Args:
        data: Pandas Series with missing values (NaN)
        n_neighbors: Number of neighbors to use for imputation
        
    Returns:
        Pandas Series with missing values imputed
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    values = imputer.fit_transform(data.values.reshape(-1, 1))
    return pd.Series(values.flatten(), index=data.index)

