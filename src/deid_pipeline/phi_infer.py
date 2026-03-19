#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from deid_pipeline.config import load_config
from deid_pipeline.common import (
    validate_shard_args,
    split_dataframe_for_shard,
    save_csv,
)

STRUCTURED_TEMPLATE = """
<guideline>
### Input:
<text>
### Output:
"""


def parse_args():
    p = argparse.ArgumentParser(description="PHI tagging inference for one shard")
    p.add_argument("--config", required=True)
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--num_shards", type=int, required=True)
    p.add_argument("--shard_index", type=int, required=True)
    p.add_argument("--keep_only_shard_rows", action="store_true")
    return p.parse_args()


def load_guideline(path: str) -> str:
    if path is None or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_prompt(guideline: str, record_text: str) -> str:
    template = STRUCTURED_TEMPLATE.replace("<guideline>", guideline)
    return template.replace("<text>", record_text.strip())


def load_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=quant_cfg,
        use_cache=True,
    )    
    model.eval()
    torch.cuda.empty_cache()

    return model, tokenizer


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    out = model.generate(**enc, **gen_kwargs)

    input_len = enc["input_ids"].shape[-1]
    gen_ids = out[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    stage_cfg = cfg["phi_stage"]
    runtime_cfg = cfg["runtime"]

    torch.manual_seed(runtime_cfg.get("seed", 42))
    validate_shard_args(args.num_shards, args.shard_index)

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"input_csv not found: {args.input_csv}")

    guideline = load_guideline(stage_cfg["guideline_path"])
    model, tokenizer = load_model_and_tokenizer(
        stage_cfg["model_id"],
    )

    df = pd.read_csv(args.input_csv)

    input_col = stage_cfg["input_col"]
    output_col = stage_cfg["output_col"]

    if input_col not in df.columns:
        raise ValueError(f"input_col not found: {input_col}")

    work_df, eval_mask, assigned_df = split_dataframe_for_shard(
        df=df,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        keep_only_shard_rows=args.keep_only_shard_rows,
    )

    print(f"Total rows: {len(df)}")
    print(
        f"Shard: num_shards={args.num_shards}, "
        f"shard_index={args.shard_index}, assigned_rows={len(assigned_df)}"
    )

    preds = [""] * len(work_df)
    processed = 0
    save_every = runtime_cfg.get("save_every", 50)

    for i in range(len(work_df)):
        if not eval_mask[i]:
            continue

        x = work_df.at[i, input_col]
        if pd.isna(x):
            y_pred = ""
        else:
            x = str(x).strip()
            if not x:
                y_pred = ""
            else:
                prompt = build_prompt(guideline, x)
                y_pred = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_input_length=stage_cfg["max_input_length"],
                    max_new_tokens=stage_cfg["max_new_tokens"],
                    temperature=stage_cfg["temperature"],
                    top_p=stage_cfg["top_p"],
                    repetition_penalty=stage_cfg["repetition_penalty"],
                )

        preds[i] = y_pred
        processed += 1

        if processed % save_every == 0:
            tmp_df = work_df.copy()
            tmp_df[output_col] = preds
            save_csv(tmp_df, args.output_csv)
            print(f"Saved interim: processed={processed}")

    out_df = work_df.copy()
    out_df[output_col] = preds
    save_csv(out_df, args.output_csv)
    print(f"Saved final: {args.output_csv}")


if __name__ == "__main__":
    main()