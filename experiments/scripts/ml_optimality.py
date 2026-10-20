#!/usr/bin/env python3
"""Create a stacked bar plot of model outcomes from a CSV."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from constants import INITIAL_LENGTH_RANGES, OUTCOME_COLORS


def as_bool(value):
	"""Parse booleans from common CSV formats."""
	if pd.isna(value):
		return False
	if isinstance(value, str):
		return value.strip().lower() in {"true", "1", "yes", "y"}
	return bool(value)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", required=True, help="Path to input CSV")
	parser.add_argument("--output", help="Save plot to this path instead of displaying it")
	parser.add_argument("--letter", default="", help="Name of time column")
	args = parser.parse_args()

	columns = [
		"original_length", "model_len", "model_validity",
		"model_equivalence", "model_optimal",
	]
	df = pd.read_csv(args.input)
	missing = [column for column in columns if column not in df.columns]
	if missing:
		parser.error("Missing required CSV columns: " + ", ".join(missing))

	original = pd.to_numeric(df["original_length"], errors="coerce")
	modeled = pd.to_numeric(df["model_len"], errors="coerce")
	valid = df["model_validity"].map(as_bool)
	equivalent = df["model_equivalence"].map(as_bool)
	optimal = df["model_optimal"].map(as_bool)
	if (
		original.isna().any()
		or (original < 0).any()
		or (original % 1 != 0).any()
	):
		parser.error("'original_length' must contain non-negative whole numbers")

	not_equivalent = ~valid | ~equivalent
	improved = modeled < original

	outcomes = [
		("Already-optimal",  ~improved & optimal & ~not_equivalent),
		("Improved-optimal", improved & optimal & ~not_equivalent),
		("Improved-not-optimal",  improved & ~optimal & ~not_equivalent),
		("Not-equivalent", not_equivalent | ~improved & ~optimal & ~not_equivalent),
	]

	fig, ax = plt.subplots(figsize=(max(13, len(INITIAL_LENGTH_RANGES) * 3), max(5, len(INITIAL_LENGTH_RANGES) * 1.1)))
	group_labels = []
	group_sizes = []
	for group_index, (lower, upper, label) in enumerate(INITIAL_LENGTH_RANGES):
		in_group = pd.Series(True, index=df.index)
		if lower is not None:
			in_group &= original >= lower
		if upper is not None:
			in_group &= original < upper

		group_size = int(in_group.sum())
		group_labels.append(label)
		group_sizes.append(group_size)
		left = 0.0
		for outcome_label, mask in outcomes:
			count = int((in_group & mask).sum())
			percentage = count / group_size * 100 if group_size else 0
			ax.barh(
				group_index,
				count,
				left=left,
				label=outcome_label if group_index == 0 else None,
				color=OUTCOME_COLORS[outcome_label],
			)
			if percentage >= 12.2:
				ax.text(
					left + count / 2,
					group_index,
					f"{percentage:.1f}%",
					ha="center",
					va="center",
					fontsize=13,
					color="black",
				)
			left += count
		ax.text(group_size, group_index, f" {group_size}", ha="left", va="center", fontsize=13)

	ax.set_yticks(range(len(group_labels)), group_labels)
	max_group_size = max(group_sizes, default=0)
	ax.set_xlim(0, max_group_size * 1.05 if max_group_size else 1)
	ax.set_ylabel("Original sequence length", fontsize=20)
	ax.set_xlabel("Number of blocks", fontsize=20)
	ax.set_title(f"{args.letter}", fontsize=20)
	ax.tick_params(labelsize=16)
	ax.legend(
		title="Runtime", 
		loc="upper right", 
		fontsize=18, 
		title_fontsize=20
	)
	fig.tight_layout()
	if args.output:
		fig.savefig(args.output, dpi=200, bbox_inches="tight")
	else:
		plt.show()


if __name__ == "__main__":
	main()
