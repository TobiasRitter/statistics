from pathlib import Path
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular import TabularExplainer
from omnixai.explanations.base import ExplanationBase
from plotly.graph_objects import Figure


def plot_explanation(explanation: ExplanationBase, path: Path) -> None:
    fig: Figure = explanation.plotly_plot().component
    fig.write_html(path)
    pass


def explain(model, x: pd.DataFrame, y: pd.Series) -> None:
    explainers = TabularExplainer(
        explainers=["lime", "shap", "sensitivity", "pdp", "ale"],
        mode="regression",
        data=Tabular(pd.concat([x, y], axis=1), target_column=y.name),
        model=model,
        preprocess=lambda tabular: tabular.data.values,
    )
    explanations = explainers.explain_global()

    plot_explanation(explanations["sensitivity"], Path("sensitivity.html"))
    plot_explanation(explanations["pdp"], Path("pdp.html"))
    plot_explanation(explanations["ale"], Path("ale.html"))


def main() -> None:
    x, y = load_diabetes(return_X_y=True, as_frame=True)
    rf = RandomForestRegressor()
    rf.fit(x, y.values.ravel())
    explain(rf, x, y)


if __name__ == "__main__":
    main()
