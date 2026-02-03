import pytest
import math
import statistics
from src.xor_depth import calculate_xor_depth, XORDepthResult
from src.gen_hmatrix import generate_h_matrix_secdec


def theoretical_min_depth(fanin: int) -> int:
    """
    Calculate the theoretical minimum XOR tree depth for a given fan-in.

    Args:
        fanin: Number of inputs to XOR together

    Returns:
        Minimum depth: 0 for fanin<=1, ceil(log2(fanin)) otherwise
    """
    if fanin <= 1:
        return 0
    return math.ceil(math.log2(fanin))


def test_xor_depth_calculation_simple():
    """Test XOR depth calculation on a simple example matrix."""
    # Simple H matrix: (3 rows) x (4 columns)
    # First 3 columns are parity (should be unit vectors in systematic form)
    # But for this test, we construct a simple non-systematic example
    # to verify depth calculation logic

    # H = [[1,0,1,0],   row 0: fan-in = 2 (columns 2,3 counting from parity position)
    #      [1,1,1,1],   row 1: fan-in = 4
    #      [0,1,0,1]]   row 2: fan-in = 2

    # Actually, let's use a proper systematic form:
    # For k=1 data bit, r=2 parity bits, overall parity -> (r+1)=3 rows, (k+r+1)=4 cols
    # Columns: [p0, p1, d0, p_overall]
    # Row 0 (p0): unit vector [1,0,0,1]
    # Row 1 (p1): unit vector [0,1,0,1]
    # Row 2 (overall): all ones [1,1,1,1]

    # Let's test with a manually crafted matrix for clear fan-in values
    # Using r=2, so columns are: [p0, p1, d0, d1, p_overall]
    # We want to test different fan-ins for each parity bit

    # Simple test case: r=2, k=2
    # H[0] = [1, 0, 1, 0, 0]  -> data portion: [1,0] -> fan-in = 1 -> depth = 0
    # H[1] = [0, 1, 1, 1, 0]  -> data portion: [1,1] -> fan-in = 2 -> depth = 1
    H_simple = [
        [1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]
    ]
    r = 2

    result = calculate_xor_depth(H_simple, r)

    assert len(result.depths) == 2
    assert result.depths[0] == 0, "Fan-in 1 should have depth 0"
    assert result.depths[1] == 1, "Fan-in 2 should have depth 1"
    assert result.max_depth == 1
    assert result.min_depth == 0
    assert result.avg_depth == 0.5

    # Test with higher fan-ins
    # r=3, k=4
    # H[0] -> fan-in 4 -> depth 2
    # H[1] -> fan-in 2 -> depth 1
    # H[2] -> fan-in 1 -> depth 0
    H_multi = [
        [1, 0, 0, 1, 1, 1, 1, 0],  # data cols: [1,1,1,1] -> fan-in 4 -> depth 2
        [0, 1, 0, 1, 1, 0, 0, 0],  # data cols: [1,1,0,0] -> fan-in 2 -> depth 1
        [0, 0, 1, 1, 0, 0, 0, 0],  # data cols: [1,0,0,0] -> fan-in 1 -> depth 0
        [1, 1, 1, 1, 1, 1, 1, 1]   # overall parity row
    ]
    r_multi = 3

    result_multi = calculate_xor_depth(H_multi, r_multi)

    assert len(result_multi.depths) == 3
    assert result_multi.depths[0] == 2, "Fan-in 4 should have depth 2"
    assert result_multi.depths[1] == 1, "Fan-in 2 should have depth 1"
    assert result_multi.depths[2] == 0, "Fan-in 1 should have depth 0"
    assert result_multi.max_depth == 2
    assert result_multi.min_depth == 0


def test_theoretical_min_depth():
    """Test the theoretical minimum depth calculation for various fan-ins."""
    # Test cases mapping fan-in to expected depth
    test_cases = [
        (0, 0),   # No inputs -> depth 0
        (1, 0),   # Single input -> depth 0
        (2, 1),   # 2 inputs -> depth 1
        (3, 2),   # 3 inputs -> depth 2
        (4, 2),   # 4 inputs -> depth 2
        (5, 3),   # 5 inputs -> depth 3
        (6, 3),   # 6 inputs -> depth 3
        (7, 3),   # 7 inputs -> depth 3
        (8, 3),   # 8 inputs -> depth 3
        (9, 4),   # 9 inputs -> depth 4
        (15, 4),  # 15 inputs -> depth 4
        (16, 4),  # 16 inputs -> depth 4
        (17, 5),  # 17 inputs -> depth 5
        (32, 5),  # 32 inputs -> depth 5
        (33, 6),  # 33 inputs -> depth 6
        (64, 6),  # 64 inputs -> depth 6
    ]

    for fanin, expected_depth in test_cases:
        actual_depth = theoretical_min_depth(fanin)
        assert actual_depth == expected_depth, \
            f"Fan-in {fanin}: expected depth {expected_depth}, got {actual_depth}"


@pytest.mark.parametrize("data_width", [32, 64, 128, 256, 512, 1024, 2048])
def test_xor_depth_for_all_widths(data_width):
    """Test that XOR depth calculation runs without errors for all standard data widths."""
    for scheme in ["hamming", "hsiao"]:
        H, r = generate_h_matrix_secdec(data_width, scheme)

        # Should not raise any exceptions
        result = calculate_xor_depth(H, r)

        # Basic sanity checks
        assert isinstance(result, XORDepthResult)
        assert len(result.depths) == r
        assert all(isinstance(d, int) for d in result.depths)
        assert all(d >= 0 for d in result.depths)
        assert result.max_depth >= result.min_depth
        assert result.min_depth <= result.avg_depth <= result.max_depth


@pytest.mark.parametrize("data_width", [32, 64, 128, 256, 512, 1024])
def test_hsiao_balanced_fanin(data_width):
    """
    Test that Hsiao scheme has more balanced fan-in (lower standard deviation)
    compared to Hamming scheme.

    This validates that Hsiao's row-load balancing approach is effective.
    """
    H_hamming, r_hamming = generate_h_matrix_secdec(data_width, "hamming")
    H_hsiao, r_hsiao = generate_h_matrix_secdec(data_width, "hsiao")

    assert r_hamming == r_hsiao, "Both schemes should have same r for same k"
    r = r_hamming
    k = data_width

    # Calculate fan-ins for each scheme
    fanin_hamming = []
    fanin_hsiao = []

    for i in range(r):
        # Count number of 1s in data portion (columns r to r+k-1)
        fanin_h = sum(H_hamming[i][r + j] for j in range(k))
        fanin_hs = sum(H_hsiao[i][r + j] for j in range(k))

        fanin_hamming.append(fanin_h)
        fanin_hsiao.append(fanin_hs)

    # Calculate standard deviations
    std_hamming = statistics.stdev(fanin_hamming) if len(fanin_hamming) > 1 else 0.0
    std_hsiao = statistics.stdev(fanin_hsiao) if len(fanin_hsiao) > 1 else 0.0

    # Hsiao should have lower or equal standard deviation (better balance)
    # Allow small tolerance for edge cases
    assert std_hsiao <= std_hamming + 0.5, \
        f"Hsiao std ({std_hsiao:.2f}) should be <= Hamming std ({std_hamming:.2f}) " \
        f"for data_width={data_width}. Fan-ins: Hamming={fanin_hamming}, Hsiao={fanin_hsiao}"

    # Additional check: Hsiao max fan-in should generally be <= Hamming max fan-in
    max_hamming = max(fanin_hamming)
    max_hsiao = max(fanin_hsiao)

    # This is a strong property of Hsiao but may not always hold strictly
    # so we just record it as informational
    assert max_hsiao <= max_hamming + 1, \
        f"Hsiao max fan-in ({max_hsiao}) should be close to or better than " \
        f"Hamming max fan-in ({max_hamming}) for data_width={data_width}"


@pytest.mark.parametrize("data_width", [32, 64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("scheme", ["hamming", "hsiao"])
def test_depth_within_expected_range(data_width, scheme):
    """
    Test that XOR depth values are within reasonable expected range.

    Depth should be:
    - Minimum: 0 (for fan-in of 0 or 1)
    - Maximum: ceil(log2(k)) where k is the number of data bits
    """
    H, r = generate_h_matrix_secdec(data_width, scheme)
    k = data_width

    result = calculate_xor_depth(H, r)

    # All depths should be non-negative
    assert all(d >= 0 for d in result.depths), \
        f"All depths should be non-negative, got {result.depths}"

    # Maximum theoretical depth is ceil(log2(k)) since worst case is
    # all k data bits XORed for one parity bit
    max_theoretical_depth = math.ceil(math.log2(k)) if k > 1 else 0

    assert result.max_depth <= max_theoretical_depth, \
        f"Max depth {result.max_depth} exceeds theoretical limit {max_theoretical_depth} " \
        f"for k={k}, scheme={scheme}"

    # Check that each individual depth is also within bounds
    for i, depth in enumerate(result.depths):
        # Calculate fan-in for this parity bit
        fanin = sum(H[i][r + j] for j in range(k))
        expected_depth = theoretical_min_depth(fanin)

        assert depth == expected_depth, \
            f"Parity bit {i} with fan-in {fanin}: expected depth {expected_depth}, " \
            f"got {depth}"


def test_xor_depth_empty_matrix():
    """Test edge case of empty or minimal matrix."""
    # Empty-ish case: r=1, k=0 (though this may not be valid in practice)
    # Just ensure no crashes
    H_empty = [[1]]
    r = 1

    # This should return empty depths or handle gracefully
    # Depending on implementation, k=0 means no data columns
    # So no depths to calculate for data bits
    # The function should handle this without crashing
    try:
        result = calculate_xor_depth(H_empty, r)
        # If it doesn't crash, verify reasonable output
        assert isinstance(result, XORDepthResult)
    except (ValueError, IndexError):
        # It's also acceptable to raise an error for invalid input
        pass


def test_xor_depth_result_consistency():
    """Test that XORDepthResult fields are internally consistent."""
    # Use a real matrix
    H, r = generate_h_matrix_secdec(64, "hsiao")
    result = calculate_xor_depth(H, r)

    # Check consistency
    assert result.max_depth == max(result.depths)
    assert result.min_depth == min(result.depths)
    assert abs(result.avg_depth - (sum(result.depths) / len(result.depths))) < 1e-9

    # Max should be >= min
    assert result.max_depth >= result.min_depth

    # Avg should be between min and max
    assert result.min_depth <= result.avg_depth <= result.max_depth
