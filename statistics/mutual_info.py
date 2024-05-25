import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def mutual_info(
    x: pd.DataFrame,
    y: pd.Series,
    n_neighbors: int = 3,
    random_state: int | None = None,
) -> pd.Series:
    mi = mutual_info_regression(
        x,
        y,
        n_neighbors=n_neighbors,
        discrete_features=False,
        random_state=random_state,
    )
    return pd.Series(mi, index=x.columns, name=y.name)


def mutual_info_df(
    x: pd.DataFrame,
    y: pd.DataFrame,
    n_neighbors: int = 3,
    random_state: int | None = None,
) -> pd.DataFrame:
    mi_series = [
        mutual_info(x, y[target], n_neighbors, random_state) for target in y.columns
    ]
    return pd.concat(mi_series, axis=1)
