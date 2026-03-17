#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from pathlib import Path
import pandas as pd

from deid_pipeline.config import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Prepare merged PHI output for de-identification")
    p.add_argument("--config", required=True)
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    prep_cfg = cfg["prepare_stage"]
    runtime_cfg = cfg["runtime"]

    random.seed(runtime_cfg.get("seed", 42))

    df_output = pd.read_csv(args.input_csv)

    input_phi_col = prep_cfg["input_phi_col"]
    output_phi_col = prep_cfg["output_phi_col"]

    if input_phi_col not in df_output.columns:
        raise ValueError(f"Column not found: {input_phi_col}")

    # 行は落とさず、そのまま全件保持
    df_output_phi = df_output.copy()

    # phi を含む行だけ候補を付与するためのマスク
    phi_mask = df_output_phi[input_phi_col].fillna("").str.contains(r"<phi_\w+>", regex=True)

    df_location = pd.read_csv(prep_cfg["location_csv"])
    location_list = df_location[prep_cfg["location_col"]].dropna().tolist()

    name1 = pd.read_json(prep_cfg["name1_ndjson"], lines=True)
    name2 = pd.read_json(prep_cfg["name2_ndjson"], lines=True)
    name_list = (
        name1[prep_cfg["name_col"]].dropna().tolist()
        + name2[prep_cfg["name_col"]].dropna().tolist()
    )

    n_name = prep_cfg.get("num_name_candidates", 3)
    n_loc = prep_cfg.get("num_location_candidates", 3)

    if len(name_list) < n_name:
        raise ValueError("Not enough names for random sampling")
    if len(location_list) < n_loc:
        raise ValueError("Not enough locations for random sampling")

    # phi がある行だけ候補を付与、ない行は空リスト
    df_output_phi["name_list"] = [
        random.sample(name_list, n_name) if is_phi else []
        for is_phi in phi_mask
    ]
    df_output_phi["location_list"] = [
        random.sample(location_list, n_loc) if is_phi else []
        for is_phi in phi_mask
    ]

    # 列名変更
    df_output_phi = df_output_phi.rename(columns={input_phi_col: output_phi_col})

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_output_phi.to_csv(args.output_csv, index=False)

    print(f"Saved prepared CSV: {args.output_csv}")
    print(f"Total rows: {len(df_output_phi)}")
    print(f"Rows with phi tags: {int(phi_mask.sum())}")


if __name__ == "__main__":
    main()