from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class XORDepthResult:
    """Result of XOR depth calculation for a parity check matrix."""
    depths: List[int]  # XOR depth for each parity bit
    max_depth: int
    min_depth: int
    avg_depth: float


def calculate_xor_depth(H: List[List[int]], r: int) -> XORDepthResult:
    """
    Calculate the XOR tree depth for each parity bit in the H matrix.

    For a systematic SECDED encoder:
    - Each parity bit p[i] is computed as XOR of data bits where H[i, r+j] == 1
    - XOR depth is ceil(log2(fan-in)) where fan-in is the number of data bits
    - If fan-in is 0 or 1, depth is 0

    Args:
        H: (r+1) x (k+r+1) binary matrix
        r: number of SEC parity bits

    Returns:
        XORDepthResult containing depth information
    """
    k = len(H[0]) - r - 1  # total columns - parity columns - overall parity
    depths = []

    for i in range(r):
        # Count fan-in for this parity bit (number of 1s in data portion)
        fan_in = sum(H[i][r + j] for j in range(k))

        # Calculate XOR tree depth
        if fan_in <= 1:
            depth = 0
        else:
            # depth = ceil(log2(fan_in))
            depth = (fan_in - 1).bit_length()

        depths.append(depth)

    if not depths:
        return XORDepthResult(depths=[], max_depth=0, min_depth=0, avg_depth=0.0)

    return XORDepthResult(
        depths=depths,
        max_depth=max(depths),
        min_depth=min(depths),
        avg_depth=sum(depths) / len(depths)
    )
