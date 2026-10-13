"""Unit tests for the primal-dual gap integral (tests/search_gap.py).

Run: .venv/Scripts/python.exe -m tests.test_search_gap
"""
import math

from tests.search_gap import gapIntegral, parseGapSamples, primalDualGap


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b):
    return abs(a - b) < 1e-9


def line(epoch, incumbent, bound, event="sample", iterations=0):
    return (f"[INFO] x - [SearchEngine gap] epoch={epoch} event={event} iterations={iterations} "
            f"incumbent={incumbent} bound={bound}")


def expectFailure(output, start, budget, fragment):
    try:
        gapIntegral(parseGapSamples(output), start, budget)
    except ValueError as error:
        require(fragment in str(error), f"Unexpected error {error}")
        return
    raise AssertionError(f"Expected failure containing {fragment!r}")


def main():
    require(primalDualGap(math.inf, 3.0) == 1.0, "no incumbent must have gap 1")
    require(primalDualGap(10.0, 10.0) == 0.0, "closed gap must be 0")
    require(close(primalDualGap(10.0, 4.0), 0.6), "gap is (UB-LB)/UB for positive values")
    require(primalDualGap(1.0, -1.0) == 1.0, "opposite signs give gap 1")

    # Setup 10 s with gap 1, root at t=10 (no incumbent), incumbent 10 with
    # bound 4 at t=40, bound 5 at t=70, budget 100 s.
    output = "\n".join([
        line(1010.0, "inf", "0.500000", "root"),
        line(1040.0, "10.000000", "4.000000", "incumbent"),
        line(1070.0, "10.000000", "5.000000"),
    ])
    result = gapIntegral(parseGapSamples(output), 1000.0, 100.0)
    expected = 40.0 * 1.0 + 30.0 * 0.6 + 30.0 * 0.5
    require(close(result["integral"], expected), f"integral {result['integral']} != {expected}")
    require(close(result["normalized"], expected / 100.0), "normalized integral")
    require(close(result["final_gap"], 0.5), "final gap")
    require(close(result["first_incumbent_elapsed"], 40.0), "first incumbent time")

    # Samples after the budget (deadline reporting) do not count.
    late = output + "\n" + line(1105.0, "10.000000", "10.000000", "deadline")
    late_result = gapIntegral(parseGapSamples(late), 1000.0, 100.0)
    require(close(late_result["integral"], expected), "late sample must not change the integral")
    require(close(late_result["final_gap"], 0.5), "late sample must not change the final gap")

    # Exhausted search closes the gap.
    closed = output + "\n" + line(1080.0, "10.000000", "10.000000", "final")
    closed_result = gapIntegral(parseGapSamples(closed), 1000.0, 100.0)
    require(close(closed_result["integral"], 40.0 + 18.0 + 5.0), "closed gap integral")
    require(closed_result["final_gap"] == 0.0, "closed final gap")

    # No incumbent at all: integral equals the budget.
    none_result = gapIntegral(parseGapSamples(line(1010.0, "inf", "0.5", "root")), 1000.0, 100.0)
    require(close(none_result["normalized"], 1.0), "no incumbent normalized integral is 1")
    require(none_result["first_incumbent_elapsed"] is None, "no first incumbent")

    # Validation failures.
    expectFailure("", 1000.0, 100.0, "No [SearchEngine gap]")
    expectFailure("\n".join([line(1010.0, "10.0", "4.0"), line(1020.0, "10.0", "3.0")]), 1000.0, 100.0,
                  "bound decreased")
    expectFailure("\n".join([line(1010.0, "10.0", "4.0"), line(1020.0, "11.0", "4.0")]), 1000.0, 100.0,
                  "Incumbent increased")
    expectFailure(line(1010.0, "10.0", "12.0"), 1000.0, 100.0, "exceeds incumbent")
    print("search gap tests passed")


if __name__ == "__main__":
    main()
