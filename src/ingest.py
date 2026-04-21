from pathlib import Path
import pandas as pd
from src.unify_schema import load_rossmann
from src.database import fetch_user_sales_df

INTERIM_DIR = Path("data/interim")


def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    rossmann_path = Path("data/raw/rossmann")
    if not rossmann_path.exists():
        print("Rossmann folder not found: data/raw/rossmann")
        return

    # Load existing CSV datasets
    df_rossmann = load_rossmann(rossmann_path)

    # Load new data from MySQL database
    print("Fetching data from MySQL...")
    df_user = fetch_user_sales_df()

    # Combine both data sources
    df_combined = pd.concat([df_rossmann, df_user], ignore_index=True)

    df_combined.to_csv(INTERIM_DIR / "combined.csv", index=False)

    print("Saved data/interim/combined.csv")
    print(df_combined.tail())


if __name__ == "__main__":
    main()