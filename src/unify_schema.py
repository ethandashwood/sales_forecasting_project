import pandas as pd


def load_rossmann(folder):
    train_df = pd.read_csv(
        folder / "train.csv",
        dtype={"StateHoliday": "str"},
        low_memory=False
    )
    store_df = pd.read_csv(folder / "store.csv")

    df = train_df.merge(store_df, on="Store", how="left")

    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df["store_id"] = df["Store"].astype(str)
    df["business_id"] = "rossmann"
    df["is_holiday"] = df["StateHoliday"].apply(lambda x: 0 if x == "0" else 1)

    df["business_type"] = "pharmacy"
    df["store_type"] = df["StoreType"].fillna("unknown")
    df["region"] = "germany"
    df["product_category"] = df["Assortment"].fillna("general")

    df["store_size_sqft"] = 2000
    df.loc[df["StoreType"] == "a", "store_size_sqft"] = 1500
    df.loc[df["StoreType"] == "b", "store_size_sqft"] = 2500
    df.loc[df["StoreType"] == "c", "store_size_sqft"] = 4000
    df.loc[df["StoreType"] == "d", "store_size_sqft"] = 6000

    df = df[
        [
            "date",
            "business_id",
            "sales",
            "store_id",
            "business_type",
            "store_type",
            "store_size_sqft",
            "region",
            "product_category",
            "is_holiday",
        ]
    ].copy()

    df = df.dropna(subset=["date", "sales"])
    return df