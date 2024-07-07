import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular import TabularExplainer


def explain(model, x: pd.DataFrame, y: pd.Series) -> None:
    explainers = TabularExplainer(
        explainers=["lime", "shap", "sensitivity", "pdp", "ale"],
        mode="regression",
        data=Tabular(pd.concat([x, y], axis=1), target_column=y.name),
        model=model,
        preprocess=lambda tabular: tabular.data.values,
    )
    explanations = explainers.explain_global()

    explanations["sensitivity"].ipython_plot()
    explanations["pdp"].ipython_plot()
    explanations["ale"].ipython_plot()


def main() -> None:
    x, y = load_diabetes(return_X_y=True, as_frame=True)
    rf = RandomForestRegressor()
    rf.fit(x, y.values.ravel())
    explain(rf, x, y)


if __name__ == "__main__":
    main()
