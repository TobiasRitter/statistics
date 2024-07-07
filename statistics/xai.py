import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from omnixai.data.tabular import Tabular
from omnixai.preprocessing.base import Identity
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.explainers.tabular import TabularExplainer


def main() -> None:
    housing = fetch_california_housing()
    df = pd.DataFrame(
        np.concatenate([housing.data, housing.target.reshape((-1, 1))], axis=1),
        columns=list(housing.feature_names) + ["target"],
    )
    tabular_data = Tabular(df, target_column="target")
    transformer = TabularTransform(target_transform=Identity()).fit(tabular_data)
    x, y = df.values[:, :-1], df.values[:, -1]

    rf = RandomForestRegressor(n_estimators=200)
    rf.fit(x, y)

    # Initialize a TabularExplainer
    explainers = TabularExplainer(
        explainers=["lime", "shap", "sensitivity", "pdp", "ale"],
        mode="regression",
        data=tabular_data,
        model=rf,
        preprocess=lambda z: transformer.transform(z),
    )
    explanations = explainers.explain_global()

    print("Sensitivity results:")
    explanations["sensitivity"].ipython_plot()
    print("PDP results:")
    explanations["pdp"].ipython_plot()
    print("ALE results:")
    explanations["ale"].ipython_plot()


if __name__ == "__main__":
    main()
