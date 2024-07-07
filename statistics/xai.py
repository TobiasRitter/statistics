from pathlib import Path
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular import TabularExplainer
from omnixai.explanations.base import ExplanationBase
from plotly.graph_objects import Figure
from sklearn.model_selection import train_test_split


def plot_explanation(explanation: ExplanationBase, path: Path) -> None:
    fig: Figure = explanation.plotly_plot().component
    fig.write_html(path)


def explain(
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
) -> None:
    explainer = TabularExplainer(
        explainers=["lime", "shap", "sensitivity", "pdp", "ale"],
        mode="regression",
        data=Tabular(pd.concat([x_train, y_train], axis=1), target_column=y_train.name),
        model=model,
        preprocess=lambda tabular: tabular.data.values,
    )
    local_explains = explainer.explain(x_val)
    global_explains = explainer.explain_global()

    plot_explanation(local_explains["lime"], Path("lime.html"))
    plot_explanation(local_explains["shap"], Path("shap.html"))
    plot_explanation(global_explains["sensitivity"], Path("sensitivity.html"))
    plot_explanation(global_explains["pdp"], Path("pdp.html"))
    plot_explanation(global_explains["ale"], Path("ale.html"))


def main() -> None:
    x, y = load_diabetes(return_X_y=True, as_frame=True)
    x_train, x_val, y_train, _ = train_test_split(x, y, test_size=0.2)
    rf = RandomForestRegressor()
    rf.fit(x, y.values.ravel())
    explain(rf, x_train, y_train, x_val)


if __name__ == "__main__":
    main()
