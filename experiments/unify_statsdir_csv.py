#!/usr/bin/env python3
"""Build or extend the unified statsdir/global.csv.

Modes (combine freely):
    --cleaned [CSV]   Build the base global CSV from blocks_cleaned.csv.
    --sat [CSV]       Inner-join SAT results.
    --model [CSV]     Inner-join model predictions.
    --optimal [CSV]   Inner-join optimality-check results (needs model columns).

Without --cleaned, extensions are applied to an existing --global-input.

Adding a new extension: define an `Extension(...)` in the EXTENSIONS section.
The CLI flag, column validation, join, and output columns all follow from it.

File layout:
    1. Configuration      paths, column mappings, constants
    2. Generic helpers    CSV reading, block_id handling
    3. Extension model    the `Extension` class and its join logic
    4. Extension configs  SAT / model / optimality definitions
    5. Base global rows   build from cleaned CSV, or load an existing global CSV
    6. Output             atomic write + verification
    7. CLI                argument parsing and main()
"""

import argparse
import csv
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Optional


# ============================================================================
# 1. Configuration
# ============================================================================

# --- Default paths ---
DEFAULT_CLEANED_INPUT = "statsdir/blocks_cleaned.csv"
DEFAULT_GLOBAL_CSV = "statsdir/global.csv"

# --- Base CSV format ---
CLEANED_DELIMITER = ";"

# Cleaned-CSV column -> global-CSV column
BASE_FIELDS = {
    "block_id": "block_id",
    "previous_solution": "original_block",
    "initial_n_instrs": "original_length",
    "initial_estimated_gas": "original_gas",
    "solution_found": "greedy_solution",
    "optimized_n_instrs": "greedy_length",
    "optimized_estimated_gas": "greedy_gas",
}

# --- Block ID handling ---
# Duplicate IDs become "<id>__dupN". Joins use the original ID via JOIN_KEY.
JOIN_KEY = "_join_block_id"
DUPLICATE_SUFFIX = re.compile(r"__dup\d+$")

# --- Extension column mappings: extension-CSV column -> global-CSV column ---
SAT_FIELDS = {
    "solution_found": "sat_solution",
    "optimized_n_instrs": "sat_length",
    "optimized_estimated_gas": "sat_gas",
    "partially_optimal": "sat_optimal",
    "solver_time_in_sec": "sat_time",
}
MODEL_FIELDS = {
    "predicted_seq": "model_solution",
    "predicted_len": "model_len",
    "time": "model_time",
    "valid": "model_validity",
}
OPTIMALITY_FIELDS = {
    "solution_found": "optimality_check_solution",
    "optimized_n_instrs": "optimality_check_length",
    "partially_optimal": "optimality_check_optimal",
    "timeout": "optimality_check_tout",
    "solver_time_in_sec": "optimality_check_time",
}


# ============================================================================
# 2. Generic helpers
# ============================================================================

def read_csv(path, required_fields, label, delimiter=","):
    """Read a CSV; return (fieldnames, rows). Raise if required columns are missing."""
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        fieldnames = list(reader.fieldnames or [])
        missing = set(required_fields) - set(fieldnames)
        if missing:
            raise ValueError(
                f"{label} CSV is missing required columns: {', '.join(sorted(missing))}"
            )
        return fieldnames, list(reader)


def get_block_id(row, label):
    """Return the row's stripped block_id, raising if it is blank."""
    block_id = (row.get("block_id") or "").strip()
    if not block_id:
        raise ValueError(f"{label} CSV contains a blank block_id.")
    return block_id


def group_rows(rows, key_func):
    """Group rows into {key: [rows]}, preserving first-seen key order."""
    groups = {}
    for row in rows:
        groups.setdefault(key_func(row), []).append(row)
    return groups


def parse_number(value):
    """Parse a numeric CSV value so that 5 and 5.0 compare equal (None if invalid)."""
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None


def make_unique_block_ids(block_ids):
    """Keep the first occurrence as-is; suffix later duplicates with __dupN."""
    original_ids = set(block_ids)
    seen_counts = Counter()
    used_ids = set()
    unique_ids = []

    for block_id in block_ids:
        if not block_id:
            raise ValueError("Input contains a blank block_id; cannot ensure unique IDs.")

        seen_counts[block_id] += 1
        if seen_counts[block_id] == 1:
            candidate = block_id
        else:
            suffix = seen_counts[block_id]
            candidate = f"{block_id}__dup{suffix}"
            while candidate in used_ids or candidate in original_ids:
                suffix += 1
                candidate = f"{block_id}__dup{suffix}"

        if candidate in used_ids:
            raise ValueError(f"Could not create a unique block_id for {block_id!r}.")
        used_ids.add(candidate)
        unique_ids.append(candidate)

    return unique_ids


# ============================================================================
# 3. Extension model
# ============================================================================

@dataclass(frozen=True)
class Extension:
    """A CSV that is inner-joined onto the global rows.

    A global row and an extension row are joined when they share a block_id AND
    `global_key_field` (global row) equals `input_key_field` (extension row).
    """

    # --- Identity / CLI ---
    name: str                       # label used in messages
    flag: str                       # CLI flag (without "--"), also the args attribute
    default_input: str              # input path used when the flag has no value
    help: str                       # --help text
    check_summary: str              # what the join verifies, shown after it runs

    # --- Join rule ---
    global_key_field: str           # global column that must match ...
    input_key_field: str            # ... this extension column
    field_mapping: dict             # extension column -> global column
    strict: bool = False            # raise on key mismatch instead of dropping rows

    # --- Optional features ---
    # global column -> candidate extension columns; the first one present is used
    aliased_fields: dict = field(default_factory=dict)
    # global columns that must already exist (produced by earlier extensions)
    required_global_fields: tuple = ()
    # computes additional global columns from (global_row, extension_row)
    extra_fields: Optional[Callable[[dict, dict], dict]] = None
    extra_output_fields: tuple = ()

    @property
    def required_input_fields(self):
        return {"block_id", self.input_key_field, *self.field_mapping}

    @property
    def output_fields(self):
        """Global columns this extension adds, in output order."""
        return (
            *self.field_mapping.values(),
            *self.aliased_fields,
            *self.extra_output_fields,
        )

    def apply(self, global_rows, input_path):
        """Return the inner join of `global_rows` with the CSV at `input_path`."""
        self._check_global_fields(global_rows)
        fieldnames, input_rows = read_csv(
            input_path, self.required_input_fields, self.name
        )
        mapping = self._resolve_mapping(fieldnames)

        input_by_id = group_rows(input_rows, lambda row: get_block_id(row, self.name))
        global_by_id = group_rows(global_rows, lambda row: row[JOIN_KEY])

        joined_rows = []
        mismatches = []
        for block_id, group in input_by_id.items():
            candidates = global_by_id.get(block_id, [])
            for input_row in group:
                matches = [
                    row for row in candidates
                    if row[self.global_key_field] == input_row[self.input_key_field]
                ]
                if candidates and not matches:
                    mismatches.append(block_id)
                for global_row in matches:
                    joined_rows.append(self._join(global_row, input_row, mapping))

        self._report_mismatches(mismatches)
        return joined_rows

    # --- internals ---

    def _check_global_fields(self, global_rows):
        missing = [
            column for column in self.required_global_fields
            if any(column not in row for row in global_rows)
        ]
        if missing:
            raise ValueError(
                f"The --{self.flag} extension requires {', '.join(missing)}; run the "
                "extensions that produce them first or provide a --global-input "
                "that already contains them."
            )

    def _resolve_mapping(self, fieldnames):
        """Extend field_mapping with whichever alias column exists in the CSV."""
        mapping = dict(self.field_mapping)
        for output_field, candidates in self.aliased_fields.items():
            source = next((name for name in candidates if name in fieldnames), None)
            if source is None:
                options = " or ".join(f"'{name}'" for name in candidates)
                raise ValueError(
                    f"{self.name} CSV must include {options} for {output_field}."
                )
            mapping[source] = output_field
        return mapping

    def _join(self, global_row, input_row, mapping):
        joined = dict(global_row)
        for source_field, output_field in mapping.items():
            joined[output_field] = input_row[source_field]
        if self.extra_fields is not None:
            joined.update(self.extra_fields(global_row, input_row))
        return joined

    def _report_mismatches(self, mismatches):
        if not mismatches:
            return
        if self.strict:
            raise ValueError(
                f"{self.name} {self.input_key_field} did not match "
                f"global {self.global_key_field} for {len(mismatches)} block_id row(s), "
                f"e.g. {', '.join(mismatches[:10])}."
            )
        print(
            f"{self.name} sequence check: {len(mismatches)} block_id row(s) excluded "
            f"because {self.input_key_field} did not match {self.global_key_field}"
        )


# ============================================================================
# 4. Extension configs (edit / add extensions here)
# ============================================================================

def model_optimal_field(global_row, optimality_row):
    """model_optimal = (model length equals optimality-check length)."""
    model_length = parse_number(global_row["model_len"])
    check_length = parse_number(optimality_row["optimized_n_instrs"])
    is_equal = (
        model_length is not None
        and check_length is not None
        and model_length == check_length
    )
    return {"model_optimal": str(is_equal).lower()}


SAT_EXTENSION = Extension(
    name="SAT",
    flag="sat",
    default_input="statsdir/splitted_blocks/sat.csv",
    help="Join SAT data only",
    check_summary="SAT previous_solution/global greedy_solution equality verified",
    global_key_field="greedy_solution",
    input_key_field="previous_solution",
    field_mapping=SAT_FIELDS,
    strict=True,
)

MODEL_EXTENSION = Extension(
    name="Model",
    flag="model",
    default_input="statsdir/september_contracts.csv",
    help="Join model predictions",
    check_summary="Model original_seq/global original_block equality verified",
    global_key_field="original_block",
    input_key_field="original_seq",
    field_mapping=MODEL_FIELDS,
    aliased_fields={"model_equivalence": ("equivalence", "equivalent")},
)

OPTIMALITY_EXTENSION = Extension(
    name="Optimality",
    flag="optimal",
    default_input="statsdir/splitted_blocks_check_for_optimality/sat.csv",
    help="Join optimality-check results",
    check_summary=(
        "Optimality previous_solution/model_solution equality and model length "
        "comparison verified"
    ),
    global_key_field="model_solution",
    input_key_field="previous_solution",
    field_mapping=OPTIMALITY_FIELDS,
    required_global_fields=("model_solution", "model_len"),
    extra_fields=model_optimal_field,
    extra_output_fields=("model_optimal",),
)

# Extensions are always applied in this order, regardless of CLI flag order.
EXTENSIONS = (SAT_EXTENSION, MODEL_EXTENSION, OPTIMALITY_EXTENSION)


# ============================================================================
# 5. Base global rows
# ============================================================================

def build_base_rows(cleaned_path):
    """Build global rows from the cleaned CSV, making block_ids unique."""
    _, source_rows = read_csv(
        cleaned_path, BASE_FIELDS, "Input", delimiter=CLEANED_DELIMITER
    )
    original_ids = [(row.get("block_id") or "").strip() for row in source_rows]
    unique_ids = make_unique_block_ids(original_ids)

    rows = []
    for source_row, unique_id, original_id in zip(source_rows, unique_ids, original_ids):
        row = {out: source_row[src] for src, out in BASE_FIELDS.items()}
        row["block_id"] = unique_id
        row[JOIN_KEY] = original_id
        rows.append(row)
    return rows


def load_global_rows(global_path):
    """Load an existing global CSV; return (rows, fieldnames)."""
    fieldnames, rows = read_csv(global_path, BASE_FIELDS.values(), "Global")
    for row in rows:
        row["block_id"] = get_block_id(row, "Global")
        row[JOIN_KEY] = DUPLICATE_SUFFIX.sub("", row["block_id"])
    return rows, fieldnames


def apply_extensions(rows, output_fields, extensions, args):
    """Apply each requested extension in order; return (rows, output_fields)."""
    output_fields = list(output_fields)
    for extension in extensions:
        input_path = Path(getattr(args, extension.flag))
        rows = extension.apply(rows, input_path)
        output_fields.extend(
            column for column in extension.output_fields if column not in output_fields
        )
        print(f"Applied {extension.name} inner join from {input_path}")
        print(f"  {extension.check_summary} for the resulting {len(rows)} rows")
    return rows, output_fields


# ============================================================================
# 6. Output
# ============================================================================

def write_output(rows, output_fields, output_path):
    """Atomically write the CSV, verify it by reading it back, return the rows read."""
    output_rows = [{col: row.get(col, "") for col in output_fields} for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output_path.parent,
            delete=False,
        ) as output_file:
            temp_path = Path(output_file.name)
            writer = csv.DictWriter(output_file, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(output_rows)

        _, written_rows = read_csv(temp_path, (), "Output")
        _verify_output(written_rows, output_rows)
        temp_path.replace(output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return written_rows


def _verify_output(written_rows, expected_rows):
    block_ids = [row["block_id"] for row in written_rows]
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("Output verification failed: block_id values are not unique.")
    if written_rows != expected_rows:
        raise ValueError(
            "Output verification failed: written rows differ from mapped rows."
        )


# ============================================================================
# 7. CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Build or extend {DEFAULT_GLOBAL_CSV} in a selected mode."
    )
    parser.add_argument(
        "--cleaned",
        nargs="?",
        const=DEFAULT_CLEANED_INPUT,
        metavar="CSV",
        help=f"Build the base global CSV only (default input: {DEFAULT_CLEANED_INPUT})",
    )
    for extension in EXTENSIONS:
        parser.add_argument(
            f"--{extension.flag}",
            nargs="?",
            const=extension.default_input,
            metavar="CSV",
            help=f"{extension.help} (default input: {extension.default_input})",
        )
    parser.add_argument(
        "--output", default=DEFAULT_GLOBAL_CSV, help="Output comma-delimited CSV"
    )
    parser.add_argument(
        "--global-input", default=DEFAULT_GLOBAL_CSV, help="Existing global CSV"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    requested = [ext for ext in EXTENSIONS if getattr(args, ext.flag) is not None]

    if args.cleaned is None and not requested:
        flags = ", ".join(["--cleaned", *(f"--{ext.flag}" for ext in EXTENSIONS)])
        raise SystemExit(f"Choose {flags}, or a combination.")

    output_path = Path(args.output)

    # Step 1: base rows (fresh from cleaned CSV, or an existing global CSV)
    if args.cleaned is not None:
        rows = build_base_rows(Path(args.cleaned))
        output_fields = list(BASE_FIELDS.values())
        duplicate_count = len(rows) - len({row[JOIN_KEY] for row in rows})
        print(f"Built base rows from {args.cleaned}")
        print(f"Duplicate block_id occurrences suffixed: {duplicate_count}")
        source_description = f"cleaned input {args.cleaned}"
    else:
        rows, output_fields = load_global_rows(Path(args.global_input))
        source_description = f"existing global CSV {args.global_input}"
    base_count = len(rows)

    # Step 2: extensions
    rows, output_fields = apply_extensions(rows, output_fields, requested, args)

    # Step 3: write + verify
    written_rows = write_output(rows, output_fields, output_path)

    # Summary
    if requested:
        print(f"Blocks: {base_count} base rows, {len(written_rows)} inner-joined rows")
        print(f"Applied requested extensions to {source_description}")
    else:
        print(f"Blocks: {base_count} input, {len(written_rows)} output (verified equal)")

    block_ids = [row["block_id"] for row in written_rows]
    print(f"Wrote {output_path}")
    print(f"Output block_id values unique: {len(block_ids) == len(set(block_ids))}")


if __name__ == "__main__":
    main()