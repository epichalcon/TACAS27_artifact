#!/usr/bin/env python3
"""Plot contract counts by compiler version and optimization status from CSV files."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from constants import COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_ORANGE


# Compiler versions with a lower combined count are grouped into "Others".
OTHER_VERSION_THRESHOLD = 50


def main():
    parser = argparse.ArgumentParser(
        description="Plot contract counts by compiler version with optimization breakdown."
    )

    parser.add_argument("--input", required=True, help="Path to first compiler options CSV")
    parser.add_argument(
        "--input2", help="Optional second compiler options CSV to stack for comparison"
    )
    parser.add_argument("-o", "--output", required=True, help="Save the plot to this image path")
    args = parser.parse_args()

    def read_counts_df(path):
        """Returns a DataFrame with MultiIndex columns: (file_label, status)"""
        df = pd.read_csv(path)
        
        # Determine origin and optimization status
        if "origin" in df.columns:
            origin = df["origin"].astype("string").str.strip().str.lower()
        else:
            origin = pd.Series("sol", index=df.index)

        if "OptimizationUsed" in df.columns:
            opt_val = df["OptimizationUsed"]
            if pd.api.types.is_string_dtype(opt_val.dtype) or opt_val.dtype == object:
                opt_used = opt_val.astype("string").str.strip().str.lower().isin({"1", "true", "yes", "y"})
            else:
                opt_used = (opt_val == 1)
        else:
            opt_used = pd.Series(False, index=df.index)

        optimized = (origin == "sol") | ((origin == "json") & opt_used)
        df["_is_optimized"] = optimized.map({True: "Optimized", False: "Unoptimized"})

        versions = df["CompilerVersion"].astype("string").str.split(
            "+commit.", n=1, regex=False
        ).str[0].fillna("Unknown")
        
        df["_version"] = versions
        
        # Group by version and optimization status
        counts = df.groupby(["_version", "_is_optimized"]).size().unstack(fill_value=0)
        # Ensure both columns always exist
        for col in ["Unoptimized", "Optimized"]:
            if col not in counts.columns:
                counts[col] = 0
        return counts[["Unoptimized", "Optimized"]]

    file1_stem = Path(args.input).stem
    counts1 = read_counts_df(args.input)
    
    if args.input2:
        file2_stem = Path(args.input2).stem
        counts2 = read_counts_df(args.input2)
        # Combine both datasets into a MultiIndex DataFrame columns: (File, Status)
        all_counts = pd.concat({file1_stem: counts1, file2_stem: counts2}, axis=1).fillna(0).astype(int)
    else:
        all_counts = counts1.copy()
        # Add file prefix level for uniform indexing structure
        all_counts.columns = pd.MultiIndex.from_product([[file1_stem], all_counts.columns])

    # Group rare versions into "Others" based on total totals across all columns
    version_totals = all_counts.sum(axis=1)
    other_versions = version_totals[version_totals < OTHER_VERSION_THRESHOLD].index
    if len(other_versions):
        others_row = all_counts.loc[other_versions].sum(axis=0)
        all_counts = all_counts.drop(index=other_versions)
        all_counts.loc["Others"] = others_row

    def version_sort_key(version):
        if version in {"Unknown", "Others"}:
            return ()
        return tuple(int(part) for part in re.findall(r"\d+", version))

    compiler_versions = [v for v in all_counts.index if v not in {"Others", "Unknown"}]
    compiler_versions.sort(key=version_sort_key, reverse=True)
    if "Unknown" in all_counts.index:
        compiler_versions.append("Unknown")
    if "Others" in all_counts.index:
        compiler_versions.append("Others")
    all_counts = all_counts.reindex(compiler_versions).fillna(0)

    fig, ax = plt.subplots(figsize=(max(8, len(all_counts) * 0.6), 5))

    if args.input2:
        # Colors: File 1 -> Unoptimized , Optimized 
        #         File 2 -> Unoptimized , Optimized 
        colors = [COLOR_YELLOW, COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, ]
        all_counts.plot(
            kind="bar", stacked=True, ax=ax, color=colors, legend=False
        )
    else:
        colors = [COLOR_YELLOW, COLOR_BLUE, ]
        all_counts.plot(
            kind="bar", stacked=True, ax=ax, color=colors, legend=False
        )

    ax.set_xlabel("Compiler versions")
    ax.set_ylabel("Number of addresses")
    ax.tick_params(axis="x", labelrotation=45, labelsize=12)
    fig.tight_layout()

    fig.savefig(args.output, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()