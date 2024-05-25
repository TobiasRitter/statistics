import pandas as pd
from statistics.mutual_info import mutual_info


def test_mutual_info() -> None:
    x = pd.DataFrame(
        {
            "a": [21, 22, 23, 24, 25, 26, 27, 28, 29],
            "b": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "c": [-10, 5, 30, -12, -50, 100, 20, 13, -90],
        }
    )
    y = pd.Series([21, 22, 23, 24, 25, 26, 27, 28, 29], name="target")
    mi = mutual_info(x, y)
    assert len(mi) == 3
    assert mi.name == "target"
