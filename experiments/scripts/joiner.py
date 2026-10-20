#!/usr/bin/env python3
"""Combine two CSV files with matching headers into one CSV file."""

import argparse
import csv
import sys


def combine_csv(first_path, second_path, output_path):
	with open(first_path, "r", newline="", encoding="utf-8") as first_file:
		first_reader = csv.DictReader(first_file)
		if first_reader.fieldnames is None:
			raise ValueError(f"{first_path} is empty or has no header")
		headers = first_reader.fieldnames

		with open(second_path, "r", newline="", encoding="utf-8") as second_file:
			second_reader = csv.DictReader(second_file)
			if second_reader.fieldnames != headers:
				raise ValueError("The two CSV files must have the same headers")

			with open(output_path, "w", newline="", encoding="utf-8") as output_file:
				writer = csv.DictWriter(output_file, fieldnames=headers)
				writer.writeheader()
				writer.writerows(first_reader)
				writer.writerows(second_reader)


def main():
	parser = argparse.ArgumentParser(
		description="Combine two CSV files that have identical headers."
	)
	parser.add_argument("--first_csv", help="Path to the first input CSV")
	parser.add_argument("--second_csv", help="Path to the second input CSV")
	parser.add_argument("--output_csv", help="Path for the combined output CSV")
	args = parser.parse_args()

	try:
		combine_csv(args.first_csv, args.second_csv, args.output_csv)
	except (OSError, csv.Error, ValueError) as error:
		print(f"Error: {error}", file=sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
