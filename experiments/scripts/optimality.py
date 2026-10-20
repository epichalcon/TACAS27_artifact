#!/usr/bin/env python3
"""Plot SAT block outcomes by initial length as horizontal stacked bars."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from constants import INITIAL_LENGTH_RANGES, OUTCOME_COLORS as CATEGORY_COLORS, TIMEOUT_SECONDS


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", required=True, help="Path to input CSV")
	parser.add_argument(
		"--output", default="optimality.png", help="Path to output PNG"
	)
	parser.add_argument("--letter", default="", help="Name of time column")
	args = parser.parse_args()

	data = pd.read_csv(args.input)
	required_columns = {
		"sat_length",
		"sat_optimal",
		"sat_time",
		"original_length",
	}
	missing_columns = required_columns - set(data.columns)
	if missing_columns:
		parser.error(
			"input CSV is missing required column(s): "
			+ ", ".join(sorted(missing_columns))
		)
	if data.empty:
		parser.error("input CSV contains no blocks")

	sat_length = pd.to_numeric(data["sat_length"], errors="coerce")
	solver_time = pd.to_numeric(data["sat_time"], errors="coerce")
	initial_n_instrs = pd.to_numeric(data["original_length"], errors="coerce")
	optimal_text = data["sat_optimal"].astype("string").str.strip().str.lower()
	optimal = optimal_text.map({"true": True, "false": False, "1": True, "0": False})

	saved_length = initial_n_instrs - sat_length

	if solver_time.isna().any() or (solver_time < 0).any():
		parser.error("'solver_time_in_sec' must contain non-negative numeric values")
	if (
		initial_n_instrs.isna().any()
		or (initial_n_instrs < 0).any()
		or (initial_n_instrs % 1 != 0).any()
	):
		parser.error("'initial_n_instrs' must contain non-negative whole numbers")
	if optimal.isna().any():
		parser.error("'partially_optimal' values must be true or false")

	timed_out = solver_time > TIMEOUT_SECONDS
	optimal = optimal.astype(bool)

	category_labels = [
	"Already-optimal",
	"Improved-optimal",
	"Improved-Timeout",
    "Nothing-Timeout",
	]
	category_masks = [
		optimal & (saved_length == 0) & ~timed_out,
		optimal & (saved_length > 0) & ~timed_out,
		~optimal & (saved_length > 0) & ~timed_out,
		~optimal & (saved_length == 0) & ~timed_out | timed_out | (saved_length < 0),
	]

	fig, ax = plt.subplots(
		figsize=(
			max(13, len(INITIAL_LENGTH_RANGES) * 3),
			max(5, len(INITIAL_LENGTH_RANGES) * 1.1),
		)
	)
	group_labels = []
	group_sizes = []
	for group_index, (lower, upper, label) in enumerate(INITIAL_LENGTH_RANGES):
		in_group = initial_n_instrs.notna()
		if lower is not None:
			in_group &= initial_n_instrs >= lower
		if upper is not None:
			in_group &= initial_n_instrs < upper

		group_size = int(in_group.sum())
		group_labels.append(label)
		group_sizes.append(group_size)
		left = 0.0
		for category_label, mask in zip(category_labels, category_masks):
			color = CATEGORY_COLORS[category_label]
			count = int((in_group & mask).sum())
			percentage = count / group_size * 100 if group_size else 0
			ax.barh(
				group_index,
				count,
				left=left,
				label=category_label if group_index == 0 else None,
				color=color,
			)
			if percentage >= 13:
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
	ax.legend(title="Outcome",  loc="upper right")

	ax.legend(
		title="Runtime", 
		loc="upper right", 
		fontsize=18, 
		title_fontsize=20
	)

	fig.tight_layout()
	fig.savefig(args.output, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
	main()
