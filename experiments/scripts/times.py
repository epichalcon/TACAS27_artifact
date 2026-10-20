#!/usr/bin/env python3
"""Plot solver runtime distributions by initial instruction-count range."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from constants import INITIAL_LENGTH_RANGES, TIME_COLORS, TIME_RANGES

# Change this column name if the input CSV uses a different initial-size field.
INITIAL_LENGTH_COLUMN = "original_length"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="times.png", help="Path to output PNG")
    parser.add_argument("--time-column", default="sat_time", help="Name of time column")
    parser.add_argument("--legend", action="store_true", help="Show legend on the plot")
    parser.add_argument("--bigger-letters", action="store_true", help="Show legend on the plot")
    parser.add_argument("--letter", default="", help="Name of time column")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    required_columns = {INITIAL_LENGTH_COLUMN, args.time_column}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        parser.error(
            "input CSV is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )
    if data.empty:
        parser.error("input CSV contains no blocks")

    initial_length = pd.to_numeric(data[INITIAL_LENGTH_COLUMN], errors="coerce")
    solver_time = pd.to_numeric(data[args.time_column], errors="coerce")
    if (
        initial_length.isna().any()
        or (initial_length < 0).any()
        or (initial_length % 1 != 0).any()
    ):
        parser.error(f"'{INITIAL_LENGTH_COLUMN}' must contain non-negative whole numbers")
    if solver_time.isna().any() or (solver_time < 0).any():
        parser.error(f"'{args.time_column}' must contain non-negative numeric values")

    time_masks = []
    for lower, lower_inclusive, upper, upper_inclusive, label in TIME_RANGES:
        in_range = solver_time.notna()
        if lower is not None:
            in_range &= (
                solver_time >= lower if lower_inclusive else solver_time > lower
            )
        if upper is not None:
            in_range &= (
                solver_time <= upper if upper_inclusive else solver_time < upper
            )
        time_masks.append((label, in_range))

    fig, ax = plt.subplots(figsize=(max(8, len(INITIAL_LENGTH_RANGES) * 1.5), 5))
    group_labels = []
    
    # Scaling factor to bring bars closer together
    spacing_factor = 0.75
    
    for group_index, (lower, upper, label) in enumerate(INITIAL_LENGTH_RANGES):
        in_group = initial_length.notna()
        if lower is not None:
            in_group &= initial_length >= lower
        if upper is not None:
            in_group &= initial_length < upper

        group_size = int(in_group.sum())
        group_labels.append(label)
        bottom = 0.0
        
        x_pos = group_index * spacing_factor
        
        for time_label, time_mask in time_masks:
            count = int((in_group & time_mask).sum())
            percentage = count / group_size * 100 if group_size else 0
            color = TIME_COLORS[time_label]
            ax.bar(
                x_pos,
                percentage,
                bottom=bottom,
                label=time_label if group_index == 0 else None,
                color=color,
                width=0.50
            )
            if percentage >= 5:

                if args.bigger_letters:
                    ax.text(
                        x_pos,
                        bottom + percentage / 2,
                        f"{percentage:.1f}%",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=13,
                    )
                else:

                    ax.text(
                        x_pos,
                        bottom + percentage / 2,
                        f"{percentage:.1f}%",
                        ha="center",
                        va="center",
                        color="black",
                    )
            bottom += percentage

        group_mean = solver_time[in_group].mean()
        if args.bigger_letters:
            ax.text(x_pos, 102, f"{group_mean:.2f}s", ha="center", va="bottom", fontsize=13)
        else:
            ax.text(x_pos, 102, f"{group_mean:.2f}s", ha="center", va="bottom")

    ax.set_xticks([i * spacing_factor for i in range(len(group_labels))], group_labels)
    ax.set_xlim(-0.5 * spacing_factor, (len(INITIAL_LENGTH_RANGES) - 0.5) * spacing_factor)
    ax.set_ylim(0, 110)

    if args.bigger_letters:
        ax.set_xlabel("Original sequence length", fontsize=20)
        ax.set_title(f"{args.letter}", fontsize=20)
    else:
        ax.set_xlabel("Original sequence length")
        ax.set_title(f"{args.letter} ")
    
    if args.legend:
        ax.legend(
            title="Runtime", 
            bbox_to_anchor=(1.02, 1), 
            loc="upper left", 
            fontsize=11, 
            title_fontsize=12
        )

    if args.bigger_letters:
        ax.tick_params(labelsize=16)
        
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()