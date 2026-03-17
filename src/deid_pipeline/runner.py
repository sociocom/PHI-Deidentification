#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
from pathlib import Path

from deid_pipeline.config import load_config
from deid_pipeline.common import assign_gpu
from deid_pipeline.merge import merge_shards


def _run_subprocess(cmd: list[str], env: dict) -> None:
    print("Launch:", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")


def _launch_sharded_stage(
    module_name: str,
    config_path: str,
    input_csv: str,
    output_dir: str,
    num_shards: int,
    gpus: list[int],
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    procs = []
    for shard_index in range(num_shards):
        gpu = assign_gpu(shard_index, gpus)
        output_csv = os.path.join(output_dir, f"out_{shard_index:02d}.csv")

        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            module_name,
            "--config",
            config_path,
            "--input_csv",
            input_csv,
            "--output_csv",
            output_csv,
            "--num_shards",
            str(num_shards),
            "--shard_index",
            str(shard_index),
            "--keep_only_shard_rows",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        print(f"Launch shard {shard_index} on GPU {gpu}")
        procs.append(subprocess.Popen(cmd, env=env))

    for p in procs:
        p.wait()
        if p.returncode != 0:
            raise RuntimeError(f"Sharded stage failed with return code {p.returncode}")


def run_phi(config_path: str) -> str:
    cfg = load_config(config_path)

    project_name = cfg["project_name"]
    gpus = cfg["hardware"]["gpus"]
    num_shards = cfg["hardware"]["phi_num_shards"]

    raw_input_csv = cfg["paths"]["raw_input_csv"]
    output_root = cfg["paths"]["output_root"]

    shard_dir = os.path.join(output_root, "phi", project_name)
    merged_csv = os.path.join(output_root, "phi", f"{project_name}_merged.csv")

    _launch_sharded_stage(
        module_name="deid_pipeline.phi_infer",
        config_path=config_path,
        input_csv=raw_input_csv,
        output_dir=shard_dir,
        num_shards=num_shards,
        gpus=gpus,
    )

    merge_shards(shard_dir, merged_csv)
    return merged_csv


def run_prepare(config_path: str, phi_merged_csv: str) -> str:
    cfg = load_config(config_path)
    project_name = cfg["project_name"]
    output_root = cfg["paths"]["output_root"]

    prepared_csv = os.path.join(output_root, "prepare", f"{project_name}_prepared.csv")
    Path(prepared_csv).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "deid_pipeline.prepare_for_deid",
        "--config",
        config_path,
        "--input_csv",
        phi_merged_csv,
        "--output_csv",
        prepared_csv,
    ]
    _run_subprocess(cmd, os.environ.copy())
    return prepared_csv


def run_deid(config_path: str, prepared_csv: str) -> str:
    cfg = load_config(config_path)

    project_name = cfg["project_name"]
    gpus = cfg["hardware"]["gpus"]
    num_shards = cfg["hardware"]["deid_num_shards"]
    output_root = cfg["paths"]["output_root"]

    shard_dir = os.path.join(output_root, "deid", project_name)
    merged_csv = os.path.join(output_root, "deid", f"{project_name}_merged.csv")

    _launch_sharded_stage(
        module_name="deid_pipeline.deid_infer",
        config_path=config_path,
        input_csv=prepared_csv,
        output_dir=shard_dir,
        num_shards=num_shards,
        gpus=gpus,
    )

    merge_shards(shard_dir, merged_csv)
    return merged_csv


def run_all(config_path: str) -> str:
    phi_merged_csv = run_phi(config_path)
    prepared_csv = run_prepare(config_path, phi_merged_csv)
    deid_merged_csv = run_deid(config_path, prepared_csv)
    return deid_merged_csv