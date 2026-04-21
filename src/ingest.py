from pathlib import Path
import pandas as pd
from src.unify_schema import load_rossmann

INTERIM_DIR = Path("data/interim")


def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    rossmann_path = Path("data/raw/rossmann")
    if not rossmann_path.exists():
        print("Rossmann folder not found: data/raw/rossmann")
        return

    df = load_rossmann(rossmann_path)
    df.to_csv(INTERIM_DIR / "combined.csv", index=False)

    print("Saved data/interim/combined.csv")
    print(df.head())


if __name__ == "__main__":
    main()