import pandas as pd
from statistics.mutual_info import mutual_info, mutual_info_df


def test_mutual_info() -> None:
    x = pd.DataFrame(
        {
            "a": [21, 22, 23, 24, 25, 26, 27, 28, 29],
            "b": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "c": [-10, 5, 30, -12, -50, 100, 20, 13, -90],
        }
    )
    y = pd.Series([21, 22, 23, 24, 25, 26, 27, 28, 29], name="target")

    result = mutual_info(x, y)
    assert len(result) == 3
    assert result.name == "target"


def test_mutual_info_df() -> None:
    x = pd.DataFrame(
        {
            "a": [21, 22, 23, 24, 25, 26, 27, 28, 29],
            "b": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "c": [-10, 5, 30, -12, -50, 100, 20, 13, -90],
        }
    )
    y = pd.DataFrame(
        {
            "x": [21, 22, 23, 24, 25, 26, 27, 28, 29],
            "y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "z": [-10, 5, 30, -12, -50, 100, 20, 13, -90],
        }
    )

    result = mutual_info_df(x, y)
    assert result.shape == (3, 3)
