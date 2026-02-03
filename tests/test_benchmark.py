"""
Benchmark tests comparing Hamming vs Hsiao SECDED schemes.

Tests XOR depth, fan-in statistics, and overall performance across
various data widths from 32 to 2048 bits.
"""
from __future__ import annotations
import statistics
from typing import List, Dict, Tuple

import pytest

from src.gen_hmatrix import generate_h_matrix_secdec, Scheme
from src.xor_depth import calculate_xor_depth, XORDepthResult
from src.metrics import compare_schemes, generate_comparison_table, analyze_scheme


# Standard data widths to test (powers of 2 and common sizes)
STANDARD_WIDTHS = [32, 64, 128, 256, 512, 1024, 2048]

# Extended widths for comprehensive testing
ALL_WIDTHS = [
    32, 33, 34, 35, 36, 37, 38, 39, 40,
    48, 56, 64, 72, 80, 96, 112, 128,
    160, 192, 224, 256, 288, 320, 384, 448, 512,
    640, 768, 896, 1024, 1280, 1536, 1792, 2048
]


def _get_fanin_stats(k: int, scheme: Scheme) -> Dict[str, float]:
    """Calculate fan-in statistics for a given scheme."""
    H, r = generate_h_matrix_secdec(k, scheme)

    fanins = []
    for i in range(r):
        fan_in = sum(H[i][r + j] for j in range(k))
        fanins.append(fan_in)

    if not fanins:
        return {
            "max": 0,
            "min": 0,
            "avg": 0.0,
            "std": 0.0
        }

    return {
        "max": max(fanins),
        "min": min(fanins),
        "avg": statistics.mean(fanins),
        "std": statistics.stdev(fanins) if len(fanins) > 1 else 0.0
    }


def test_compare_xor_depth_all_widths(capsys):
    """
    Test comparing XOR depth for Hamming vs Hsiao across all standard widths.

    Prints a detailed comparison table showing XOR depth metrics for each width.
    Use pytest -s to see the output.
    """
    print("\n" + "=" * 100)
    print("XOR DEPTH COMPARISON: Hamming vs Hsiao (All Standard Widths)")
    print("=" * 100)
    print()

    # Table header
    header = (
        f"{'k':<6} | {'r':<3} | "
        f"{'Hamming Max':<12} | {'Hsiao Max':<11} | "
        f"{'Hamming Avg':<12} | {'Hsiao Avg':<11} | "
        f"{'Improvement':<12}"
    )
    print(header)
    print("-" * 100)

    results = []

    for k in STANDARD_WIDTHS:
        # Generate both schemes
        H_hamming, r = generate_h_matrix_secdec(k, "hamming")
        H_hsiao, r_hsiao = generate_h_matrix_secdec(k, "hsiao")

        assert r == r_hsiao, f"Parity bit count mismatch for k={k}"

        # Calculate XOR depths
        depth_hamming = calculate_xor_depth(H_hamming, r)
        depth_hsiao = calculate_xor_depth(H_hsiao, r)

        # Calculate improvement (negative means Hsiao is better)
        max_improvement = depth_hsiao.max_depth - depth_hamming.max_depth
        avg_improvement = depth_hsiao.avg_depth - depth_hamming.avg_depth

        # Store results for assertion
        results.append({
            "k": k,
            "r": r,
            "hamming_max": depth_hamming.max_depth,
            "hsiao_max": depth_hsiao.max_depth,
            "hamming_avg": depth_hamming.avg_depth,
            "hsiao_avg": depth_hsiao.avg_depth,
            "max_improvement": max_improvement,
            "avg_improvement": avg_improvement
        })

        # Format improvement string
        if max_improvement < 0:
            improvement_str = f"{max_improvement:+d} (better)"
        elif max_improvement == 0:
            improvement_str = "0 (equal)"
        else:
            improvement_str = f"{max_improvement:+d} (worse)"

        # Print row
        row = (
            f"{k:<6} | {r:<3} | "
            f"{depth_hamming.max_depth:<12} | {depth_hsiao.max_depth:<11} | "
            f"{depth_hamming.avg_depth:<12.2f} | {depth_hsiao.avg_depth:<11.2f} | "
            f"{improvement_str:<12}"
        )
        print(row)

    print("=" * 100)
    print()

    # Summary statistics
    max_improvements = [r["max_improvement"] for r in results]
    avg_improvements = [r["avg_improvement"] for r in results]

    better_count = sum(1 for imp in max_improvements if imp < 0)
    equal_count = sum(1 for imp in max_improvements if imp == 0)
    worse_count = sum(1 for imp in max_improvements if imp > 0)

    print(f"Summary (Max XOR Depth):")
    print(f"  Hsiao better:  {better_count}/{len(STANDARD_WIDTHS)} cases")
    print(f"  Hsiao equal:   {equal_count}/{len(STANDARD_WIDTHS)} cases")
    print(f"  Hsiao worse:   {worse_count}/{len(STANDARD_WIDTHS)} cases")
    print(f"  Avg improvement: {statistics.mean(avg_improvements):.3f}")
    print()

    # Verify at least some improvements
    assert better_count + equal_count > 0, "Hsiao should be better or equal in at least some cases"


def test_hsiao_better_or_equal_max_depth():
    """
    Test that Hsiao's max XOR depth is better than or equal to Hamming
    for most standard widths.

    This verifies that the Hsiao construction achieves its goal of
    minimizing maximum XOR tree depth.
    """
    better_or_equal_count = 0
    worse_count = 0
    results = []

    for k in STANDARD_WIDTHS:
        H_hamming, r = generate_h_matrix_secdec(k, "hamming")
        H_hsiao, _ = generate_h_matrix_secdec(k, "hsiao")

        depth_hamming = calculate_xor_depth(H_hamming, r)
        depth_hsiao = calculate_xor_depth(H_hsiao, r)

        if depth_hsiao.max_depth <= depth_hamming.max_depth:
            better_or_equal_count += 1
        else:
            worse_count += 1
            results.append({
                "k": k,
                "hamming_max": depth_hamming.max_depth,
                "hsiao_max": depth_hsiao.max_depth
            })

    # Hsiao should be better or equal in majority of cases (at least 70%)
    success_rate = better_or_equal_count / len(STANDARD_WIDTHS)

    if worse_count > 0:
        print(f"\nCases where Hsiao is worse than Hamming:")
        for r in results:
            print(f"  k={r['k']}: Hamming max={r['hamming_max']}, Hsiao max={r['hsiao_max']}")

    assert success_rate >= 0.7, (
        f"Hsiao should be better or equal in at least 70% of cases, "
        f"but was only {success_rate*100:.1f}% ({better_or_equal_count}/{len(STANDARD_WIDTHS)})"
    )


def test_generate_comparison_report(capsys):
    """
    Generate and print a comprehensive comparison report.

    Creates a detailed table showing all metrics (XOR depth, fan-in, overhead)
    for both Hamming and Hsiao schemes across all standard widths.
    """
    print("\n" + "=" * 130)
    print("COMPREHENSIVE SECDED COMPARISON REPORT")
    print("=" * 130)
    print()

    # Table header
    header = (
        f"{'k':<6} | {'r':<3} | "
        f"{'Hamming Max':<12} | {'Hsiao Max':<11} | "
        f"{'Hamming Avg':<12} | {'Hsiao Avg':<11} | "
        f"{'Max Δ':<7} | {'Avg Δ':<7} | "
        f"{'Status':<10}"
    )
    print(header)
    print("-" * 130)

    for k in STANDARD_WIDTHS:
        results = compare_schemes(k)
        hamming = results["hamming"]
        hsiao = results["hsiao"]

        max_diff = hsiao.max_xor_depth - hamming.max_xor_depth
        avg_diff = hsiao.avg_xor_depth - hamming.avg_xor_depth

        # Determine status
        if max_diff < 0:
            status = "BETTER"
        elif max_diff == 0:
            status = "EQUAL"
        else:
            status = "WORSE"

        row = (
            f"{k:<6} | {hamming.r:<3} | "
            f"{hamming.max_xor_depth:<12} | {hsiao.max_xor_depth:<11} | "
            f"{hamming.avg_xor_depth:<12.2f} | {hsiao.avg_xor_depth:<11.2f} | "
            f"{max_diff:<+7} | {avg_diff:<+7.2f} | "
            f"{status:<10}"
        )
        print(row)

    print("=" * 130)
    print()

    # Additional detailed comparison using metrics module
    print(generate_comparison_table([32, 64, 128, 256, 512, 1024, 2048]))


def test_fanin_statistics_comparison(capsys):
    """
    Compare fan-in statistics between Hamming and Hsiao schemes.

    Tests that Hsiao provides better balanced fan-in (lower max, lower std dev)
    compared to Hamming.
    """
    print("\n" + "=" * 120)
    print("FAN-IN STATISTICS COMPARISON")
    print("=" * 120)
    print()

    header = (
        f"{'k':<6} | {'r':<3} | "
        f"{'Hamming Max':<12} | {'Hsiao Max':<11} | "
        f"{'Hamming Std':<12} | {'Hsiao Std':<11} | "
        f"{'Max Δ':<7} | {'Std Δ':<7}"
    )
    print(header)
    print("-" * 120)

    hsiao_better_max_count = 0
    hsiao_better_std_count = 0

    for k in STANDARD_WIDTHS:
        hamming_stats = _get_fanin_stats(k, "hamming")
        hsiao_stats = _get_fanin_stats(k, "hsiao")

        # Get r for display
        _, r = generate_h_matrix_secdec(k, "hamming")

        max_diff = hsiao_stats["max"] - hamming_stats["max"]
        std_diff = hsiao_stats["std"] - hamming_stats["std"]

        if max_diff <= 0:
            hsiao_better_max_count += 1
        if std_diff < 0:
            hsiao_better_std_count += 1

        row = (
            f"{k:<6} | {r:<3} | "
            f"{hamming_stats['max']:<12} | {hsiao_stats['max']:<11} | "
            f"{hamming_stats['std']:<12.2f} | {hsiao_stats['std']:<11.2f} | "
            f"{max_diff:<+7} | {std_diff:<+7.2f}"
        )
        print(row)

    print("=" * 120)
    print()
    print(f"Summary:")
    print(f"  Hsiao has lower or equal max fan-in: {hsiao_better_max_count}/{len(STANDARD_WIDTHS)}")
    print(f"  Hsiao has lower std deviation:       {hsiao_better_std_count}/{len(STANDARD_WIDTHS)}")
    print()

    # Verify Hsiao achieves better balance in majority of cases
    assert hsiao_better_std_count >= len(STANDARD_WIDTHS) * 0.6, (
        f"Hsiao should have better fan-in balance (lower std) in at least 60% of cases"
    )


def test_extended_widths_spot_check():
    """
    Spot check some non-power-of-2 widths to ensure correctness.
    """
    test_widths = [33, 72, 96, 160, 320, 640, 1280]

    for k in test_widths:
        # Both schemes should generate valid matrices
        H_hamming, r_hamming = generate_h_matrix_secdec(k, "hamming")
        H_hsiao, r_hsiao = generate_h_matrix_secdec(k, "hsiao")

        # Same parity bit count
        assert r_hamming == r_hsiao

        # Valid XOR depths can be calculated
        depth_hamming = calculate_xor_depth(H_hamming, r_hamming)
        depth_hsiao = calculate_xor_depth(H_hsiao, r_hsiao)

        # XOR depths should be reasonable (not 0, not too large)
        assert depth_hamming.max_depth > 0
        assert depth_hsiao.max_depth > 0
        assert depth_hamming.max_depth <= 12  # log2(2048) + margin
        assert depth_hsiao.max_depth <= 12


@pytest.mark.parametrize("k", STANDARD_WIDTHS)
def test_individual_width_comparison(k):
    """
    Parametrized test for individual width comparison.

    Verifies that both schemes produce valid results for each standard width.
    """
    results = compare_schemes(k)

    hamming = results["hamming"]
    hsiao = results["hsiao"]

    # Both should have same parity bit count
    assert hamming.r == hsiao.r

    # Both should have valid XOR depths
    assert hamming.max_xor_depth > 0
    assert hsiao.max_xor_depth > 0

    # Both should have reasonable fan-in
    assert hamming.max_fanin > 0
    assert hsiao.max_fanin > 0

    # Hsiao should generally have better or equal max XOR depth
    # (allowed to be worse in some cases, but not by much)
    depth_difference = hsiao.max_xor_depth - hamming.max_xor_depth
    assert depth_difference <= 1, (
        f"For k={k}, Hsiao max depth ({hsiao.max_xor_depth}) should not be "
        f"more than 1 worse than Hamming ({hamming.max_xor_depth})"
    )


def test_performance_trend_analysis(capsys):
    """
    Analyze how the XOR depth improvement trend changes with data width.
    """
    print("\n" + "=" * 80)
    print("XOR DEPTH IMPROVEMENT TREND ANALYSIS")
    print("=" * 80)
    print()

    improvements = []

    for k in STANDARD_WIDTHS:
        results = compare_schemes(k)
        hamming = results["hamming"]
        hsiao = results["hsiao"]

        max_improvement = hamming.max_xor_depth - hsiao.max_xor_depth
        avg_improvement = hamming.avg_xor_depth - hsiao.avg_xor_depth

        improvements.append({
            "k": k,
            "max_improvement": max_improvement,
            "avg_improvement": avg_improvement
        })

        print(f"k={k:4d}: Max improvement={max_improvement:+2d}, Avg improvement={avg_improvement:+.3f}")

    print("=" * 80)
    print()

    # Calculate statistics on improvements
    max_improvements = [imp["max_improvement"] for imp in improvements]
    avg_improvements = [imp["avg_improvement"] for imp in improvements]

    print(f"Overall Statistics:")
    print(f"  Max depth improvement - Mean: {statistics.mean(max_improvements):+.2f}")
    print(f"  Max depth improvement - Median: {statistics.median(max_improvements):+.2f}")
    print(f"  Avg depth improvement - Mean: {statistics.mean(avg_improvements):+.3f}")
    print(f"  Avg depth improvement - Median: {statistics.median(avg_improvements):+.3f}")
    print()

    # Most improvements should be non-negative (Hsiao equal or better)
    non_negative_count = sum(1 for imp in max_improvements if imp >= 0)
    assert non_negative_count >= len(STANDARD_WIDTHS) * 0.7, (
        "Hsiao should show improvement (or be equal) in at least 70% of cases"
    )
