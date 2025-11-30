import pandas as pd
import torch


def load_tabular_data():
    wine_dataframe = pd.read_csv(
        "data/tabular_data/winequality-white.csv", delimiter=";")
    print(wine_dataframe.head)
    print(wine_dataframe.columns)
    return wine_dataframe


def extract_targets(wine_dataframe: pd.DataFrame):
    quality_target = wine_dataframe.loc[:, 'quality']
    return quality_target.values


def create_one_hot_tensor(target_tensor):
    one_hot = torch.nn.functional.one_hot(target_tensor)
    return one_hot


if __name__ == "__main__":
    df = load_tabular_data()
    targets = extract_targets(df)
    target_tensor = torch.tensor(targets)
    one_hot_tensor = create_one_hot_tensor(target_tensor)
    print(one_hot_tensor)
