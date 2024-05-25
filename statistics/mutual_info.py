import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def mutual_info(x: pd.DataFrame, y: pd.Series, n_neighbors: int = 3) -> pd.Series:
    mi = mutual_info_regression(
        x, y, n_neighbors=n_neighbors, discrete_features=False, random_state=42
    )
    return pd.Series(mi, index=x.columns, name=y.name)
