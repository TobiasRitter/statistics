import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular import TabularExplainer


def main() -> None:
    housing = fetch_california_housing()
    df = pd.DataFrame(
        np.concatenate([housing.data, housing.target.reshape((-1, 1))], axis=1),
        columns=list(housing.feature_names) + ["target"],
    )
    x, y = df.values[:, :-1], df.values[:, -1]

    rf = RandomForestRegressor()
    rf.fit(x, y)

    explainers = TabularExplainer(
        explainers=["lime", "shap", "sensitivity", "pdp", "ale"],
        mode="regression",
        data=Tabular(df, target_column="target"),
        model=rf,
        preprocess=lambda tabular: tabular.data.values,
    )
    explanations = explainers.explain_global()

    explanations["sensitivity"].ipython_plot()
    explanations["pdp"].ipython_plot()
    explanations["ale"].ipython_plot()


if __name__ == "__main__":
    main()
