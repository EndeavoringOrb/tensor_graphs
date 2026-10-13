"""Primal-dual gap integral for timed planner evaluations (docs/core/search_gap.md).

The native engine logs `[SearchEngine gap] epoch=... event=... iterations=...
incumbent=... bound=...` samples. Between samples the gap is held at the last
sampled value. Incumbents only fall and certified bounds only rise, so this
step function never under-estimates the true gap. Before the first sample (process
start, model loading, preprocessing) the gap is 1.
"""
import math
import re

GAP_LINE = re.compile(
    r"\[SearchEngine gap\] epoch=([0-9.eE+\-]+) event=(\w+) iterations=(\d+) incumbent=(\S+) bound=(\S+)")
# Tolerance for values printed with six decimals.
PRINT_TOLERANCE = 1e-5


def primalDualGap(incumbent, bound):
    """Berthold's primal-dual gap: 1 without an incumbent, 0 when closed."""
    if not math.isfinite(incumbent):
        return 1.0
    if not math.isfinite(bound):
        raise ValueError(f"Infinite certified bound {bound} with incumbent {incumbent}")
    if bound >= incumbent:
        return 0.0
    if incumbent == 0.0 and bound == 0.0:
        return 0.0
    if incumbent * bound < 0.0:
        return 1.0
    return abs(incumbent - bound) / max(abs(incumbent), abs(bound))


def parseGapSamples(output):
    samples = []
    for match in GAP_LINE.finditer(output):
        samples.append({
            "epoch": float(match.group(1)),
            "event": match.group(2),
            "iterations": int(match.group(3)),
            "incumbent": float(match.group(4)),
            "bound": float(match.group(5)),
        })
    return samples


def validateGapSamples(samples):
    if not samples:
        raise ValueError("No [SearchEngine gap] samples were reported")
    previous = None
    for sample in samples:
        if math.isfinite(sample["incumbent"]) and sample["bound"] > sample["incumbent"] + PRINT_TOLERANCE:
            raise ValueError(f"Certified bound exceeds incumbent: {sample}")
        if previous is not None:
            if sample["epoch"] < previous["epoch"]:
                raise ValueError(f"Gap samples are not time ordered: {previous} then {sample}")
            if sample["incumbent"] > previous["incumbent"] + PRINT_TOLERANCE:
                raise ValueError(f"Incumbent increased: {previous} then {sample}")
            if math.isfinite(previous["bound"]) and sample["bound"] < previous["bound"] - PRINT_TOLERANCE:
                raise ValueError(f"Certified bound decreased: {previous} then {sample}")
        previous = sample


def gapIntegral(samples, start_epoch, budget):
    """Integral of the gap over [0, budget] seconds from start_epoch.

    Returns a dict with the integral (seconds), normalized integral (/ budget),
    final gap, final incumbent/bound, and first-incumbent elapsed time.
    """
    validateGapSamples(samples)
    if not (math.isfinite(budget) and budget > 0):
        raise ValueError("Budget must be finite and positive")
    integral = 0.0
    time_cursor = 0.0
    gap = 1.0
    first_incumbent = None
    final = {"incumbent": math.inf, "bound": -math.inf}
    for sample in samples:
        t = min(max(sample["epoch"] - start_epoch, 0.0), budget)
        integral += gap * (t - time_cursor)
        time_cursor = t
        gap = primalDualGap(sample["incumbent"], sample["bound"])
        if first_incumbent is None and math.isfinite(sample["incumbent"]):
            first_incumbent = sample["epoch"] - start_epoch
        if sample["epoch"] - start_epoch <= budget:
            final = sample
    integral += gap * (budget - time_cursor)
    return {
        "integral": integral,
        "normalized": integral / budget,
        "final_gap": primalDualGap(final["incumbent"], final["bound"]),
        "final_incumbent": final["incumbent"],
        "final_bound": final["bound"],
        "first_incumbent_elapsed": first_incumbent,
        "samples": len(samples),
    }


def formatGapReport(result):
    def number(value):
        return "none" if value is None else f"{value:.17g}"
    return ("[Program gap] primal-dual-integral=" + number(result["integral"])
            + " normalized=" + number(result["normalized"])
            + " final-gap=" + number(result["final_gap"])
            + " final-incumbent=" + number(result["final_incumbent"])
            + " final-bound=" + number(result["final_bound"])
            + " first-incumbent-elapsed=" + number(result["first_incumbent_elapsed"])
            + " samples=" + str(result["samples"]))
