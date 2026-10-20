#!/usr/bin/env python3
"""Show SAT outcomes for blocks with invalid or non-equivalent model results."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from constants import INITIAL_LENGTH_RANGES, OUTCOME_COLORS, TIMEOUT_SECONDS

SAT_OUTCOMES = [
	"Already-optimal",
	"Improved-optimal",
	"Improved-not-optimal",
	"Not-improved-not-optimal",
	"Nothing-Timeout",
	"SAT result unavailable",
]


def parse_bool(value):
	"""Parse recognized CSV booleans, leaving missing/unrecognized values unknown."""
	if pd.isna(value):
		return pd.NA
	if isinstance(value, str):
		value = value.strip().lower()
	if value in {True, 1, "true", "1", "yes", "y"}:
		return True
	if value in {False, 0, "false", "0", "no", "n"}:
		return False
	return pd.NA


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", required=True, help="Path to input CSV")
	parser.add_argument("--output", help="Save plot to this path instead of displaying it")
	args = parser.parse_args()

	data = pd.read_csv(args.input)
	required_columns = {
		"original_length", "model_validity", "model_equivalence",
		"sat_length", "sat_optimal", "sat_time",
	}
	missing_columns = required_columns - set(data.columns)
	if missing_columns:
		parser.error("Missing required CSV columns: " + ", ".join(sorted(missing_columns)))
	if data.empty:
		parser.error("input CSV contains no blocks")

	initial_length = pd.to_numeric(data["original_length"], errors="coerce")
	if (
		initial_length.isna().any()
		or (initial_length < 0).any()
		or (initial_length % 1 != 0).any()
	):
		parser.error("'original_length' must contain non-negative whole numbers")

	model_valid = data["model_validity"].map(parse_bool).fillna(False).astype(bool)
	model_equivalent = data["model_equivalence"].map(parse_bool).fillna(False).astype(bool)
	selected = ~model_valid | ~model_equivalent
	failures = data.loc[selected].copy()
	lengths = initial_length.loc[selected]
	if failures.empty:
		parser.error("No invalid or non-equivalent model blocks found")

	sat_length = pd.to_numeric(failures["sat_length"], errors="coerce")
	sat_time = pd.to_numeric(failures["sat_time"], errors="coerce")
	sat_optimal = failures["sat_optimal"].map(parse_bool)
	sat_known = sat_length.notna() & sat_time.notna() & sat_optimal.notna()
	saved_length = lengths - sat_length
	timed_out = sat_time > TIMEOUT_SECONDS

	category_masks = {
		"Already-optimal": sat_known & sat_optimal.eq(True) & saved_length.eq(0) & ~timed_out,
		"Improved-optimal": sat_known & sat_optimal.eq(True) & saved_length.gt(0) & ~timed_out,
		"Improved-not-optimal": sat_known & sat_optimal.eq(False) & saved_length.gt(0) & ~timed_out,
		"Not-improved-not-optimal": sat_known & sat_optimal.eq(False) & saved_length.eq(0) & ~timed_out,
		"Nothing-Timeout": sat_known & (timed_out | saved_length.lt(0)),
		"SAT result unavailable": ~sat_known,
	}

	print(
		f"Invalid or non-equivalent model blocks: {len(failures)}/{len(data)} "
		f"({len(failures) / len(data) * 100:.1f}%)"
	)
	print("SAT outcomes for selected blocks:")
	for outcome in SAT_OUTCOMES:
		count = int(category_masks[outcome].sum())
		print(f"  {outcome}: {count} ({count / len(failures) * 100:.1f}%)")

	fig, ax = plt.subplots(figsize=(max(9, len(INITIAL_LENGTH_RANGES) * 1.6), 5))
	group_labels = []
	for group_index, (lower, upper, label) in enumerate(INITIAL_LENGTH_RANGES):
		in_group = pd.Series(True, index=failures.index)
		if lower is not None:
			in_group &= lengths >= lower
		if upper is not None:
			in_group &= lengths < upper

		group_size = int(in_group.sum())
		group_labels.append(label)
		bottom = 0.0
		for outcome in SAT_OUTCOMES:
			count = int((in_group & category_masks[outcome]).sum())
			percentage = count / group_size * 100 if group_size else 0
			ax.bar(
				group_index,
				percentage,
				bottom=bottom,
				label=outcome if group_index == 0 else None,
				color=OUTCOME_COLORS[outcome],
			)
			if percentage >= 5:
				ax.text(
					group_index,
					bottom + percentage / 2,
					f"{percentage:.1f}%",
					ha="center",
					va="center",
					color="black",
				)
			bottom += percentage
		ax.text(group_index, 102, f"n={group_size}", ha="center", va="bottom")

	ax.set_xticks(range(len(group_labels)), group_labels)
	ax.set_ylim(0, 110)
	ax.set_xlabel("Original sequence length")
	ax.set_ylabel("SAT outcomes within selected blocks (%)")
	ax.set_title("SAT outcomes when the model is invalid or non-equivalent")
	ax.legend(title="SAT outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
	fig.tight_layout()
	if args.output:
		fig.savefig(args.output, dpi=200, bbox_inches="tight")
	else:
		plt.show()


if __name__ == "__main__":
	main()
