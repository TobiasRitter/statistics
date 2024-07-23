import pandas as pd
from ucimlrepo import fetch_ucirepo


def main() -> None:
    metro = fetch_ucirepo(id=492)
    x: pd.DataFrame = metro.data.features
    y: pd.DataFrame = metro.data.targets

    print(x.head())
    print(y.head())


if __name__ == "__main__":
    main()
