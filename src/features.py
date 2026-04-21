import pandas as pd


def aggregate_sales(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    freq_map = {
        "M": "ME",
        "Y": "YE",
    }
    freq = freq_map.get(freq, freq)

    group_cols = [
        "business_id",
        "store_id",
        "business_type",
        "store_type",
        "store_size_sqft",
        "region",
        "product_category",
    ]

    out = (
        df.groupby(group_cols + [pd.Grouper(key="date", freq=freq)], dropna=False)["sales"]
        .sum()
        .reset_index()
        .sort_values(["store_id", "product_category", "date"])
    )
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    return df


def add_group_lags(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(group_cols + ["date"])

    grouped = df.groupby(group_cols)

    df["lag_1"] = grouped["sales"].shift(1)
    df["lag_2"] = grouped["sales"].shift(2)

    df["lag_4_mean"] = grouped["sales"].transform(
        lambda x: x.shift(1).rolling(4).mean()
    )

    df["lag_8_mean"] = grouped["sales"].transform(
        lambda x: x.shift(1).rolling(8).mean()
    )

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target"] = df["sales"] / df["store_size_sqft"].clip(lower=1)
    return df