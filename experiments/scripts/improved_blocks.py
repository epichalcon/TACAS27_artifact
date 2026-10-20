#!/usr/bin/env python3
"""Plot block counts by saved instruction length."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", required=True, help="Path to input CSV")
	parser.add_argument(
		"--output", default="improved_blocks.png", help="Path to output PNG"
	)
	args = parser.parse_args()

	data = pd.read_csv(args.input)
	if "saved_length" not in data.columns:
		parser.error("input CSV must contain a 'saved_length' column")

	saved_lengths = pd.to_numeric(data["saved_length"], errors="coerce")
	invalid = data["saved_length"].notna() & saved_lengths.isna()
	if invalid.any():
		parser.error("'saved_length' must contain non-negative whole numbers")

	saved_lengths = saved_lengths.dropna()
	if saved_lengths.empty:
		parser.error("input CSV contains no numeric saved_length values")
	if (saved_lengths < 0).any() or (saved_lengths % 1 != 0).any():
		parser.error("'saved_length' must contain non-negative whole numbers")

	saved_lengths = saved_lengths.astype(int)
	improvement_counts = saved_lengths[saved_lengths > 0].value_counts()

	labels = [str(value) for value in range(1, 5)]
	counts = [int(improvement_counts.get(value, 0)) for value in range(1, 5)]
	labels.append("More than 5")
	counts.append(int((saved_lengths >= 5).sum()))

	fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.45), 5))
	ax.bar(labels, counts)
	ax.set_xlabel("Saved length (instructions)")
	ax.set_ylabel("Number of blocks")
	ax.set_title("Blocks by saved instruction length")
	fig.tight_layout()
	fig.savefig(args.output, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
	main()
