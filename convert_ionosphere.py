from pathlib import Path

import pandas as pd

INPUT_FILE = Path("ionosphere.data")
OUTPUT_DIR = Path("datasets")
OUTPUT_FILE = OUTPUT_DIR / "ionosphere.csv"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Ionosphere ima 34 featurea + zadnji stupac je klasa.
    column_names = [f"f{i}" for i in range(1, 35)] + ["target"]

    df = pd.read_csv(INPUT_FILE, header=None, names=column_names)

    # Target je u originalu:
    # g = good
    # b = bad
    # Pretvaramo u 1/0.
    df["target"] = df["target"].map({"b": 0, "g": 1})

    if df["target"].isna().any():
        raise ValueError("Postoje target vrijednosti koje nisu 'g' ili 'b'.")

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"[OK] Spremljeno: {OUTPUT_FILE}")
    print(f"Broj redaka: {df.shape[0]}")
    print(f"Broj featurea: {df.shape[1] - 1}")
    print(f"Klase: {sorted(df['target'].unique().tolist())}")


if __name__ == "__main__":
    main()