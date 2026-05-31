import pandas as pd
from pathlib import Path

def converter_csv_para_parquet(caminho_csv, caminho_parquet):
    df = pd.read_csv(caminho_csv)
    df.to_parquet(caminho_parquet, engine='pyarrow', index=False)

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    CAMINHO_CSV = BASE_DIR / "dataset" / "data.csv"
    CAMINHO_PARQUET = BASE_DIR / "dataset" / "data.parquet"
    converter_csv_para_parquet(CAMINHO_CSV, CAMINHO_PARQUET)