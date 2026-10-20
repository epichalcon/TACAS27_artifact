"""Shared plotting ranges, timeout settings, and colors for graph scripts."""

# Edit this vector to change original-length bucket boundaries. Each cut is
# exclusive for the bucket before it and inclusive for the bucket after it.


def ranges_from_cuts(jump, columns):
	"""Build (lower-inclusive, upper-exclusive, label) ranges from cut points."""
	if jump <= 0:
		raise ValueError("jump must be a positive number")
	cuts = [i * jump for i in range(1, columns)]
	if any(lower >= upper for lower, upper in zip(cuts, cuts[1:])):
		raise ValueError("jump must generate strictly increasing cuts")
	if not cuts:
		return [(None, None, "All lengths")]

	ranges = [(None, cuts[0], f"<{cuts[0]}")]
	ranges.extend(
		(lower, upper, f"[{lower},{upper})")
		for lower, upper in zip(cuts, cuts[1:])
	)
	ranges.append((cuts[-1], None, f">={cuts[-1]}"))
	return ranges






JUMP = 6
COLUMNS = 6

INITIAL_LENGTH_RANGES = ranges_from_cuts(JUMP, COLUMNS)

# Keep SAT timeout classification consistent across optimality plots.
TIMEOUT_SECONDS = 10

# Entries are (lower bound, lower inclusive, upper bound, upper inclusive, label).
TIME_RANGES = [
	(None, False, 0.1, False, "Very fast (<0.1s)"),
	(0.1, True, 0.5, False, "Fast (0.1-0.5s)"),
	(0.5, True, 1, False, "Normal (0.5-1s)"),
	(1, True, 5, False, "Slow (1-5s)"),
	(5, True, 10, True, "Very slow (5-10s)"),
	(10, False, None, False, "Timeout (>10s)"),
]

# Named palette values; change these to update colors across all plots.
COLOR_ALREADY_OPTIMAL = "#2c7da0"
COLOR_IMPROVED_OPTIMAL = "#4c956c"
COLOR_IMPROVED_NOT_OPTIMAL = "#DAD14E"
COLOR_NOT_IMPROVED_NOT_OPTIMAL = "#DA8B4E"
COLOR_NOT_EQUIVALENT = "#8B0000"
COLOR_SAT_UNAVAILABLE = "#9c9c9c"

COLOR_VERY_FAST = "#0E7B7B"
COLOR_FAST = "#AE7A7E"
COLOR_NORMAL = "#B36D23"
COLOR_SLOW = "#90CCD5"
COLOR_VERY_SLOW = "#625E97"
COLOR_TIMEOUT = "#8B0000"

COLOR_BLUE = COLOR_ALREADY_OPTIMAL
COLOR_GREEN = COLOR_IMPROVED_OPTIMAL
COLOR_YELLOW = COLOR_IMPROVED_NOT_OPTIMAL 
COLOR_ORANGE = COLOR_NOT_IMPROVED_NOT_OPTIMAL


# Shared colors for SAT and model outcome charts.
OUTCOME_COLORS = {
	"Already-optimal": COLOR_ALREADY_OPTIMAL,
	"Improved-optimal": COLOR_IMPROVED_OPTIMAL,
	"Improved-Timeout": COLOR_IMPROVED_NOT_OPTIMAL,
	"Improved-not-optimal": COLOR_IMPROVED_NOT_OPTIMAL,
	"Nothing-Timeout": COLOR_NOT_EQUIVALENT,
	"Not-equivalent": COLOR_NOT_EQUIVALENT,
	"SAT result unavailable": COLOR_SAT_UNAVAILABLE,
}

# Colors for runtime bands in times.py.
TIME_COLORS = {
	"Very fast (<0.1s)": COLOR_VERY_FAST,
	"Fast (0.1-0.5s)": COLOR_FAST,
	"Normal (0.5-1s)": COLOR_NORMAL,
	"Slow (1-5s)": COLOR_SLOW,
	"Very slow (5-10s)": COLOR_VERY_SLOW,
	"Timeout (>10s)": COLOR_TIMEOUT,
}
