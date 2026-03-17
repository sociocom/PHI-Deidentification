#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from deid_pipeline.runner import run_phi, run_prepare, run_deid, run_all
from deid_pipeline.config import load_config


def main():
    parser = argparse.ArgumentParser(prog="deid-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    phi_p = subparsers.add_parser("phi", help="Run PHI tagging and merge shards")
    phi_p.add_argument("--config", required=True)

    deid_p = subparsers.add_parser("deid", help="Prepare CSV, run de-identification, and merge shards")
    deid_p.add_argument("--config", required=True)

    all_p = subparsers.add_parser("all", help="Run phi + prepare + deid")
    all_p.add_argument("--config", required=True)

    prep_p = subparsers.add_parser("prepare", help="Run only prepare step after phi merge")
    prep_p.add_argument("--config", required=True)

    args = parser.parse_args()

    if args.command == "phi":
        phi_merged = run_phi(args.config)
        print(f"PHI stage done: {phi_merged}")

    elif args.command == "prepare":
        cfg = load_config(args.config)
        phi_merged = f'{cfg["paths"]["output_root"]}/phi/{cfg["project_name"]}_merged.csv'
        prepared = run_prepare(args.config, phi_merged)
        print(f"Prepare stage done: {prepared}")

    elif args.command == "deid":
        cfg = load_config(args.config)
        phi_merged = f'{cfg["paths"]["output_root"]}/phi/{cfg["project_name"]}_merged.csv'
        prepared = run_prepare(args.config, phi_merged)
        deid_merged = run_deid(args.config, prepared)
        print(f"De-identification stage done: {deid_merged}")

    elif args.command == "all":
        final_csv = run_all(args.config)
        print(f"All stages done: {final_csv}")


if __name__ == "__main__":
    main()