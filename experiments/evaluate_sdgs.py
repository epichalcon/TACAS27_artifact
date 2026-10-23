#!/usr/bin/env python3
"""Evaluate a model on every SDG JSON file in a directory."""

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import time

from optimize_block import evaluate_block, load_model
from utils.sfs_wrapper import SFSBridge


def classify_result(result: dict) -> str:
    if result.get("equiv"):
        return "equivalent"
    if result.get("valid"):
        if "PUSH values are different" in result.get("reason", ""):
            return "different_push_values"
        return "not_equivalent"
    return "invalid"


def error_detail(index: int, block_id: str, status: str, reason: str) -> dict:
    return {
        "block_idx": index,
        "block_id": block_id,
        "original_seq": "",
        "predicted_seq": "",
        "status": status,
        "reason": reason,
        "valid": False,
        "equivalent": False,
        "original_len": 0,
        "predicted_len": 0,
        "original_gas": 0,
        "gas_saved": 0,
        "length_saved": 0,
        "time": 0,
    }
def write_results_csv(output_csv: Path | None, details: list[dict]):
    if output_csv is None:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "block_idx", "block_id", "original_seq", "predicted_seq", "status", "reason", "valid", "equivalent",
            "original_len", "predicted_len", "original_gas", "gas_saved", "length_saved", "time"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for detail in details:
            writer.writerow(detail)
    print(f"Saved detailed results to {output_csv}")


def evaluate_sdgs(
    input_dir: Path,
    vocab_path: Path,
    g2s_model: Path,
    output_csv: Path,
    limit: int | None = None,
) -> None:
    input_dir = Path(input_dir)
    sdg_paths = sorted(input_dir.glob("*.json"))
    if not sdg_paths:
        raise ValueError(f"No .json SDG files found in {input_dir}")

    vocab_payload = torch.load(vocab_path, weights_only=False)
    vocab = vocab_payload.get("vocab")

    if vocab is None:
        raise KeyError(f"No vocabulary found in {vocab_path}")

    checkpoint = g2s_model
    encoder, decoder = load_model("G2S", checkpoint, len(vocab))
    sfs = SFSBridge()
    device = torch.device("cpu")
    details = []

    for index, sdg_path in enumerate(sdg_paths):
        if limit is not None and index >= limit:
            break

        block_id = sdg_path.stem
        try:
            with sdg_path.open(encoding="utf-8") as source:
                block = json.load(source)
            start_time = time.perf_counter()
            result = evaluate_block(
                encoder,
                decoder,
                sfs,
                block,
                vocab,
                device,
                constants_map=block.get("constants", {}),
            )
            end_time = time.perf_counter()
            details.append({
                "block_idx": index,
                "block_id": block_id,
                "original_seq": block.get("original_seq", ""),
                "predicted_seq": result.get("pred_str", ""),
                "status": classify_result(result),
                "reason": result.get("reason", ""),
                "valid": bool(result.get("valid", False)),
                "equivalent": bool(result.get("equiv", False)),
                "original_len": result.get("orig_len", 0),
                "predicted_len": result.get("pred_len", 0),
                "original_gas": result.get("original_gas", 0),
                "gas_saved": result.get("gas_saved", 0),
                "length_saved": result.get("length_saved", 0),
                "time": end_time - start_time,
            })
        except Exception as exc:
            print(f"  Warning: failed to evaluate {sdg_path}: {exc}")
            details.append(error_detail(index, block_id, "evaluation_error", str(exc)))

    write_results_csv(output_csv, details)
    print(f"Evaluated {len(details)} SDG files from {input_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on SDG JSON files in a directory"
    )
    parser.add_argument("--input", required=True, type=Path, help="SDG directory")
    parser.add_argument("--output-csv", default=None, type=Path, help="Output comparison CSV")
    parser.add_argument("--limit", type=int, default=None, help="Maximum files to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.input = Path("sdgs") / args.input / "sdgs"

    
    if args.output_csv is None:
        args.output_csv = Path("statsdir") / f"{args.input.name}_comparison.csv"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        evaluate_sdgs(
            args.input,
            Path("/app/model/vocab.pt"),
            Path("/app/model/model.pt"),
            args.output_csv,
            args.limit,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        raise


if __name__ == "__main__":
    main()

