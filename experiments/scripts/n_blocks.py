#!/usr/bin/env python3
"""Plot counts of blocks grouped by their original length."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from constants import COLOR_BLUE, COLOR_GREEN, INITIAL_LENGTH_RANGES as RANGES


def main():
	parser = argparse.ArgumentParser(
		description="Plot the number of blocks in original_length ranges."
	)
	parser.add_argument("--input", required=True, help="Path to input CSV")
	parser.add_argument(
		"--input2", help="Optional second CSV to stack for comparison"
	)
	parser.add_argument("--separator", default=",", help="CSV seaprator")
	parser.add_argument("--original-column", default="original_length", help="CSV column")
	parser.add_argument("--output", required=True, help="Path to output PNG")
	parser.add_argument("--legend", action="store_true", help="Path to output PNG")
	args = parser.parse_args()

	def count_by_range(path):
		data = pd.read_csv(path, sep=args.separator)
		if args.original_column not in data.columns:
			parser.error(
				f"{path} must contain an '{args.original_column}' column"
			)
		lengths = pd.to_numeric(data[args.original_column], errors="coerce")
		counts = []
		for lower, upper, _ in RANGES:
			in_range = lengths.notna()
			if lower is not None:
				in_range &= lengths >= lower
			if upper is not None:
				in_range &= lengths < upper
			counts.append(int(in_range.sum()))
		return counts

	counts_by_input = [count_by_range(args.input)]
	if args.input2:
		counts_by_input.append(count_by_range(args.input2))

	labels = []
	for lower, upper, label in RANGES:
		labels.append(label)

	counts = pd.DataFrame(
		{
			Path(args.input).stem: counts_by_input[0],
			**(
				{Path(args.input2).stem: counts_by_input[1]}
				if args.input2
				else {}
			),
		},
		index=labels,
	)
	ax = counts.plot(
		kind="bar",
		stacked=bool(args.input2),
		color=[COLOR_BLUE, COLOR_GREEN] if args.input2 else COLOR_BLUE,
	)
	ax.set_xlabel("Original sequence length")
	ax.set_ylabel("Number of blocks")
	ax.set_title("Blocks by original length")
	if args.input2:
		ax.legend(title="Input CSV")
	plt.tight_layout()
	plt.savefig(args.output, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
	main()
