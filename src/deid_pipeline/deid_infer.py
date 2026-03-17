#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from deid_pipeline.config import load_config
from deid_pipeline.common import (
    validate_shard_args,
    split_dataframe_for_shard,
    save_csv,
    build_gen_kwargs,
)

NAMES_PLACEHOLDER = "（ここにNamesリストを貼る）"
LOCATIONS_PLACEHOLDER = "（ここにLocationsリストを貼る）"
PHONES_PLACEHOLDER = "（ここにPhonesリストを貼る）"

TAG_TO_GUIDELINE = {
    "phi_age": "guideline_age.txt",
    "phi_id": "guideline_id.txt",
    "phi_tel": "guideline_tel.txt",
    "phi_job": "guideline_job.txt",
    "phi_location": "guideline_location.txt",
    "phi_person": "guideline_person.txt",
    "phi_hospital": "guideline_hospital.txt",
}


def parse_args():
    p = argparse.ArgumentParser(description="De-identification inference for one shard")
    p.add_argument("--config", required=True)
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--num_shards", type=int, required=True)
    p.add_argument("--shard_index", type=int, required=True)
    p.add_argument("--keep_only_shard_rows", action="store_true")
    return p.parse_args()


def detect_tags(text: str) -> list:
    pattern = r"<(phi_\w+)>"
    return list(set(re.findall(pattern, text)))


def has_phi_tags(text: str) -> bool:
    return bool(re.search(r"<phi_\w+>", text))


def load_guideline_file(path: str) -> str:
    if path is None or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_combined_guideline(
    guideline_dir: str,
    detected_tags: list,
    name_list="",
    location_list="",
    phone_list="",
) -> str:
    base_path = os.path.join(guideline_dir, "guideline_base.txt")
    combined = load_guideline_file(base_path)

    for tag in detected_tags:
        if tag in TAG_TO_GUIDELINE:
            tag_guideline_path = os.path.join(guideline_dir, TAG_TO_GUIDELINE[tag])
            tag_guideline = load_guideline_file(tag_guideline_path)
            if tag_guideline:
                combined += "\n\n" + tag_guideline

    if pd.notna(name_list) and name_list:
        combined = combined.replace(NAMES_PLACEHOLDER, str(name_list))
    if pd.notna(location_list) and location_list:
        combined = combined.replace(LOCATIONS_PLACEHOLDER, str(location_list))
    if pd.notna(phone_list) and phone_list:
        combined = combined.replace(PHONES_PLACEHOLDER, str(phone_list))

    return combined


def build_prompt(
    guideline_dir: str,
    record_text: str,
    name_list="",
    location_list="",
    phone_list="",
) -> str:
    detected_tags = detect_tags(record_text)

    guideline = build_combined_guideline(
        guideline_dir=guideline_dir,
        detected_tags=detected_tags,
        name_list=name_list,
        location_list=location_list,
        phone_list=phone_list,
    )

    prompt = f"""{guideline}

### Input:
{record_text.strip()}

### Output:
"""
    return prompt


def load_model_and_tokenizer(model_id: str, use_4bit: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Load base model (without LoRA)...")

    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quant_cfg,
            use_cache=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_cache=True,
        )

    model.eval()
    return model, tokenizer


def truncate_output(text: str) -> str:
    stop_patterns = [
        "\n##",
        "\n**",
        "\n---",
        "\n問題",
        "\n解説",
        "\n説明",
        "\n注:",
        "\n注：",
        "\n※",
        "\n\n\n",
        "\n```",
    ]

    result = text
    for pattern in stop_patterns:
        if pattern in result:
            result = result.split(pattern)[0]

    return result.strip()


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

    stop_strings = ["##", "**", "解説", "---", "\n\n\n", "```"]
    stop_token_ids = []
    for s in stop_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            stop_token_ids.append(ids[0])

    eos_ids = [tokenizer.eos_token_id] + stop_token_ids

    gen_kwargs = build_gen_kwargs(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
    )

    out = model.generate(**enc, **gen_kwargs)

    input_len = enc["input_ids"].shape[-1]
    gen_ids = out[0][input_len:]
    raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return truncate_output(raw_output)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    stage_cfg = cfg["deid_stage"]
    runtime_cfg = cfg["runtime"]

    torch.manual_seed(runtime_cfg.get("seed", 42))
    validate_shard_args(args.num_shards, args.shard_index)

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"input_csv not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    input_col = stage_cfg["input_col"]
    output_col = stage_cfg["output_col"]
    name_list_col = stage_cfg["name_list_col"]
    location_list_col = stage_cfg["location_list_col"]
    phone_list_col = stage_cfg["phone_list_col"]

    if input_col not in df.columns:
        raise ValueError(f"input_col not found: {input_col}")

    has_name_list = name_list_col in df.columns
    has_location_list = location_list_col in df.columns
    has_phone_list = phone_list_col in df.columns

    if not has_name_list:
        print(f"Warning: {name_list_col} column not found")
    if not has_location_list:
        print(f"Warning: {location_list_col} column not found")
    if not has_phone_list:
        print(f"Warning: {phone_list_col} column not found")

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

    model = None
    tokenizer = None

    preds = [""] * len(work_df)
    processed = 0
    skipped = 0
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
            elif not has_phi_tags(x):
                y_pred = x
                skipped += 1
            else:
                if model is None:
                    print("Loading model (first phi tag encountered)...")
                    model, tokenizer = load_model_and_tokenizer(
                        stage_cfg["model_id"],
                        stage_cfg.get("use_4bit", False),
                    )

                name_list = work_df.at[i, name_list_col] if has_name_list else ""
                location_list = work_df.at[i, location_list_col] if has_location_list else ""
                phone_list = work_df.at[i, phone_list_col] if has_phone_list else ""

                prompt = build_prompt(
                    guideline_dir=stage_cfg["guideline_dir"],
                    record_text=x,
                    name_list=name_list,
                    location_list=location_list,
                    phone_list=phone_list,
                )

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
            print(f"Saved interim: processed={processed}, skipped={skipped}")

    out_df = work_df.copy()
    out_df[output_col] = preds
    save_csv(out_df, args.output_csv)
    print(f"Saved final: {args.output_csv}")
    print(f"Total processed={processed}, skipped(no phi tags)={skipped}")


if __name__ == "__main__":
    main()