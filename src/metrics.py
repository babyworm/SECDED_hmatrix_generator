from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import statistics

from src.gen_hmatrix import generate_h_matrix_secdec, Scheme
from src.xor_depth import calculate_xor_depth, XORDepthResult


@dataclass
class SchemeMetrics:
    """Metrics for a SECDED scheme with given data bit count."""
    scheme: Scheme
    k: int  # data bits
    r: int  # SEC parity bits
    total_bits: int  # k + r + 1
    overhead_percent: float  # (r+1)/k * 100
    max_fanin: int
    min_fanin: int
    avg_fanin: float
    fanin_std: float  # standard deviation
    max_xor_depth: int
    min_xor_depth: int
    avg_xor_depth: float


def analyze_scheme(k: int, scheme: Scheme) -> SchemeMetrics:
    """
    Analyze a SECDED scheme for given data bit count.

    Args:
        k: number of data bits
        scheme: "hamming" or "hsiao"

    Returns:
        SchemeMetrics containing all efficiency metrics
    """
    # Generate H matrix
    H, r = generate_h_matrix_secdec(k, scheme)

    # Calculate fan-in for each parity bit (number of 1s in data portion)
    fanins = []
    for i in range(r):
        fan_in = sum(H[i][r + j] for j in range(k))
        fanins.append(fan_in)

    # Calculate XOR depths
    xor_result = calculate_xor_depth(H, r)

    # Calculate metrics
    total_bits = k + r + 1
    overhead_percent = (r + 1) / k * 100

    return SchemeMetrics(
        scheme=scheme,
        k=k,
        r=r,
        total_bits=total_bits,
        overhead_percent=overhead_percent,
        max_fanin=max(fanins) if fanins else 0,
        min_fanin=min(fanins) if fanins else 0,
        avg_fanin=statistics.mean(fanins) if fanins else 0.0,
        fanin_std=statistics.stdev(fanins) if len(fanins) > 1 else 0.0,
        max_xor_depth=xor_result.max_depth,
        min_xor_depth=xor_result.min_depth,
        avg_xor_depth=xor_result.avg_depth
    )


def compare_schemes(k: int) -> Dict[str, SchemeMetrics]:
    """
    Compare Hamming and Hsiao schemes for given data bit count.

    Args:
        k: number of data bits

    Returns:
        Dictionary with "hamming" and "hsiao" keys mapping to SchemeMetrics
    """
    return {
        "hamming": analyze_scheme(k, "hamming"),
        "hsiao": analyze_scheme(k, "hsiao")
    }


def generate_comparison_table(k_values: List[int]) -> str:
    """
    Generate a comparison table for multiple data bit counts.

    Args:
        k_values: list of data bit counts to compare

    Returns:
        Formatted string containing comparison table
    """
    lines = []
    lines.append("=" * 120)
    lines.append("SECDED Scheme Comparison")
    lines.append("=" * 120)
    lines.append("")

    for k in k_values:
        results = compare_schemes(k)
        hamming = results["hamming"]
        hsiao = results["hsiao"]

        lines.append(f"Data bits (k): {k}")
        lines.append("-" * 120)
        lines.append(f"{'Metric':<25} {'Hamming':>20} {'Hsiao':>20} {'Difference':>20}")
        lines.append("-" * 120)

        # Basic parameters
        lines.append(f"{'Parity bits (r)':<25} {hamming.r:>20} {hsiao.r:>20} {''}")
        lines.append(f"{'Total bits':<25} {hamming.total_bits:>20} {hsiao.total_bits:>20} {''}")
        lines.append(f"{'Overhead %':<25} {hamming.overhead_percent:>20.2f} {hsiao.overhead_percent:>20.2f} {''}")

        lines.append("")
        lines.append("Fan-in Statistics:")
        lines.append(f"{'  Max fan-in':<25} {hamming.max_fanin:>20} {hsiao.max_fanin:>20} {hsiao.max_fanin - hamming.max_fanin:>+20}")
        lines.append(f"{'  Min fan-in':<25} {hamming.min_fanin:>20} {hsiao.min_fanin:>20} {hsiao.min_fanin - hamming.min_fanin:>+20}")
        lines.append(f"{'  Avg fan-in':<25} {hamming.avg_fanin:>20.2f} {hsiao.avg_fanin:>20.2f} {hsiao.avg_fanin - hamming.avg_fanin:>+20.2f}")
        lines.append(f"{'  Fan-in std dev':<25} {hamming.fanin_std:>20.2f} {hsiao.fanin_std:>20.2f} {hsiao.fanin_std - hamming.fanin_std:>+20.2f}")

        lines.append("")
        lines.append("XOR Depth Statistics:")
        lines.append(f"{'  Max XOR depth':<25} {hamming.max_xor_depth:>20} {hsiao.max_xor_depth:>20} {hsiao.max_xor_depth - hamming.max_xor_depth:>+20}")
        lines.append(f"{'  Min XOR depth':<25} {hamming.min_xor_depth:>20} {hsiao.min_xor_depth:>20} {hsiao.min_xor_depth - hamming.min_xor_depth:>+20}")
        lines.append(f"{'  Avg XOR depth':<25} {hamming.avg_xor_depth:>20.2f} {hsiao.avg_xor_depth:>20.2f} {hsiao.avg_xor_depth - hamming.avg_xor_depth:>+20.2f}")

        lines.append("")
        lines.append("=" * 120)
        lines.append("")

    return "\n".join(lines)


# Demo usage
if __name__ == "__main__":
    # Analyze single scheme
    print("Analyzing Hsiao SECDED for k=32:")
    metrics = analyze_scheme(32, "hsiao")
    print(f"  Scheme: {metrics.scheme}")
    print(f"  Data bits: {metrics.k}")
    print(f"  Parity bits: {metrics.r}")
    print(f"  Total bits: {metrics.total_bits}")
    print(f"  Overhead: {metrics.overhead_percent:.2f}%")
    print(f"  Fan-in: max={metrics.max_fanin}, min={metrics.min_fanin}, avg={metrics.avg_fanin:.2f}, std={metrics.fanin_std:.2f}")
    print(f"  XOR depth: max={metrics.max_xor_depth}, min={metrics.min_xor_depth}, avg={metrics.avg_xor_depth:.2f}")
    print()

    # Compare schemes for multiple k values
    print(generate_comparison_table([8, 16, 32, 64]))
