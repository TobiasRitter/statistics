from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


@dataclass
class Predictions:
    y_pred_train: pd.Series
    y_pred_test: pd.Series
    y_true_train: pd.Series
    y_true_test: pd.Series


@dataclass
class ModelEval:
    r2: float
    mse: float
    mae: float
    mape: float

    @classmethod
    def from_predictions(cls, predictions: Predictions) -> "ModelEval":
        return cls(
            r2=r2_score(predictions.y_true_test, predictions.y_pred_test),
            mse=mean_squared_error(predictions.y_true_test, predictions.y_pred_test),
            mae=mean_absolute_error(predictions.y_true_test, predictions.y_pred_test),
            mape=mean_absolute_percentage_error(
                predictions.y_true_test, predictions.y_pred_test
            ),
        )
