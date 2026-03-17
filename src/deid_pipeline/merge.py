from pathlib import Path
import glob
import pandas as pd


def merge_shards(input_dir: str, output_csv: str, pattern: str = "out_*.csv") -> None:
    paths = sorted(glob.glob(str(Path(input_dir) / pattern)))
    if not paths:
        raise FileNotFoundError(f"No shard CSVs found in {input_dir} with pattern={pattern}")

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

    if "row_id" in df.columns:
        df = df.sort_values("row_id").drop(columns=["row_id"])

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved merged CSV: {output_csv}")