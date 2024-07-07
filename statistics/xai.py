import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from omnixai.data.tabular import Tabular
from omnixai.preprocessing.base import Identity
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.explainers.tabular import TabularExplainer
from sklearn.model_selection import train_test_split


def main() -> None:
    housing = fetch_california_housing()
    df = pd.DataFrame(
        np.concatenate([housing.data, housing.target.reshape((-1, 1))], axis=1),
        columns=list(housing.feature_names) + ["target"],
    )
    tabular_data = Tabular(df, target_column="target")
    transformer = TabularTransform(target_transform=Identity()).fit(tabular_data)
    x = transformer.transform(tabular_data)
    x_train, x_test, y_train, y_test = train_test_split(
        x[:, :-1], x[:, -1], train_size=0.80
    )

    rf = RandomForestRegressor(n_estimators=200)
    rf.fit(x_train, y_train)

    # Convert the transformed data back to Tabular instances
    train_data = transformer.invert(x_train)
    test_data = transformer.invert(x_test)

    # Initialize a TabularExplainer
    explainers = TabularExplainer(
        explainers=["lime", "shap", "sensitivity", "pdp", "ale"],
        mode="regression",
        data=train_data,
        model=rf,
        preprocess=lambda z: transformer.transform(z),
        params={"lime": {"kernel_width": 3}, "shap": {"nsamples": 100}},
    )
    # Generate explanations
    test_instances = test_data[0:5]
    local_explanations = explainers.explain(X=test_instances)
    global_explanations = explainers.explain_global(
        params={
            "pdp": {
                "features": [
                    "MedInc",
                    "HouseAge",
                    "AveRooms",
                    "AveBedrms",
                    "Population",
                    "AveOccup",
                    "Latitude",
                    "Longitude",
                ]
            }
        }
    )

    index = 0
    print("LIME results:")
    local_explanations["lime"].ipython_plot(index)
    print("SHAP results:")
    local_explanations["shap"].ipython_plot(index)
    print("Sensitivity results:")
    global_explanations["sensitivity"].ipython_plot()
    print("PDP results:")
    global_explanations["pdp"].ipython_plot()
    print("ALE results:")
    global_explanations["ale"].ipython_plot()


if __name__ == "__main__":
    main()
