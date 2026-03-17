import os
from pathlib import Path
import pandas as pd


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def insert_row_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(len(df)))
    return df


def validate_shard_args(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= shard_index < num_shards):
        raise ValueError("--shard_index out of range")


def split_dataframe_for_shard(
    df: pd.DataFrame,
    num_shards: int,
    shard_index: int,
    keep_only_shard_rows: bool,
):
    df = insert_row_id(df)

    shard_mask = (df["row_id"] % num_shards) == shard_index
    assigned_df = df.loc[shard_mask].copy()

    if keep_only_shard_rows:
        work_df = assigned_df.reset_index(drop=True)
        eval_mask = [True] * len(work_df)
    else:
        work_df = df.reset_index(drop=True)
        eval_mask = (work_df["row_id"] % num_shards == shard_index).tolist()

    return work_df, eval_mask, assigned_df


def save_csv(df: pd.DataFrame, output_csv: str) -> None:
    ensure_parent_dir(output_csv)
    df.to_csv(output_csv, index=False)


def assign_gpu(shard_index: int, gpus: list[int]) -> int:
    return gpus[shard_index % len(gpus)]


def build_gen_kwargs(
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    eos_token_id,
    pad_token_id,
) -> dict:
    do_sample = temperature > 0.0
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    return kwargs