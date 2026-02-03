"""
Property-based tests for SECDED H-matrix generator.

Tests mathematical invariants and properties that must always hold for SECDED codes.
Uses random sampling instead of hypothesis for broader compatibility.
"""
import math
import random
from typing import List, Tuple

from src.gen_hmatrix import generate_h_matrix_secdec, Scheme
from src.secded_codec import encode, decode, compute_syndrome, inject_error
from src.xor_depth import calculate_xor_depth


# Configuration
NUM_SAMPLES = 10  # Number of random k values to test per property
K_MIN = 32
K_MAX = 2048
SCHEMES: List[Scheme] = ["hamming", "hsiao"]
RANDOM_SEED = 42  # For reproducibility


def _required_r(k: int) -> int:
    """Calculate minimum parity bits for SECDED: 2^r >= k + r + 1"""
    r = 1
    while (1 << r) < (k + r + 1):
        r += 1
    return r


def _column_to_int(H: List[List[int]], col_idx: int, num_rows: int) -> int:
    """Convert a column to an integer (bit mask) for comparison."""
    val = 0
    for row in range(num_rows):
        val |= (H[row][col_idx] << row)
    return val


def _get_random_k_values(seed: int = RANDOM_SEED) -> List[int]:
    """Generate random k values for testing."""
    rng = random.Random(seed)
    # Include boundary values plus random samples
    k_values = [K_MIN, K_MAX]
    k_values.extend(rng.randint(K_MIN, K_MAX) for _ in range(NUM_SAMPLES - 2))
    return k_values


class TestHMatrixProperties:
    """Property tests for H-matrix structure."""

    def test_property_columns_unique(self) -> None:
        """Property: All columns in SEC portion (excluding overall parity) must be unique."""
        for k in _get_random_k_values():
            for scheme in SCHEMES:
                H, r = generate_h_matrix_secdec(k, scheme)
                n = k + r + 1

                # Convert each SEC column (excluding last overall parity column) to integer
                col_masks = []
                for col_idx in range(n - 1):
                    mask = _column_to_int(H, col_idx, r)
                    col_masks.append(mask)

                assert len(set(col_masks)) == len(col_masks), (
                    f"k={k}, scheme={scheme}: Found duplicate columns in top {r} rows"
                )

    def test_property_columns_nonzero(self) -> None:
        """Property: All columns must be non-zero in top r rows."""
        for k in _get_random_k_values():
            for scheme in SCHEMES:
                H, r = generate_h_matrix_secdec(k, scheme)
                n = k + r + 1

                for col_idx in range(n - 1):
                    mask = _column_to_int(H, col_idx, r)
                    assert mask != 0, (
                        f"k={k}, scheme={scheme}: Column {col_idx} is zero in top {r} rows"
                    )

    def test_property_first_r_columns_identity(self) -> None:
        """Property: First r columns must be identity matrix (unit vectors)."""
        for k in _get_random_k_values():
            for scheme in SCHEMES:
                H, r = generate_h_matrix_secdec(k, scheme)

                for i in range(r):
                    for row in range(r):
                        expected = 1 if row == i else 0
                        actual = H[row][i]
                        assert actual == expected, (
                            f"k={k}, scheme={scheme}: "
                            f"Parity column {i}, row {row}: expected {expected}, got {actual}"
                        )

    def test_property_last_row_all_ones(self) -> None:
        """Property: Last row must be all ones (overall parity)."""
        for k in _get_random_k_values():
            for scheme in SCHEMES:
                H, r = generate_h_matrix_secdec(k, scheme)
                n = k + r + 1

                last_row = H[r]
                for col_idx in range(n):
                    assert last_row[col_idx] == 1, (
                        f"k={k}, scheme={scheme}: "
                        f"Overall parity row at column {col_idx}: expected 1, got {last_row[col_idx]}"
                    )

    def test_property_overall_parity_column(self) -> None:
        """Property: Last column (overall parity bit) must be [0,0,...,0,1]."""
        for k in _get_random_k_values():
            for scheme in SCHEMES:
                H, r = generate_h_matrix_secdec(k, scheme)
                n = k + r + 1
                last_col_idx = n - 1

                # Top r rows should be 0
                for row in range(r):
                    assert H[row][last_col_idx] == 0, (
                        f"k={k}, scheme={scheme}: "
                        f"Overall parity column at row {row}: expected 0, got {H[row][last_col_idx]}"
                    )

                # Last row should be 1
                assert H[r][last_col_idx] == 1, (
                    f"k={k}, scheme={scheme}: Overall parity column last row: expected 1"
                )


class TestDimensionProperties:
    """Property tests for matrix dimensions."""

    def test_property_matrix_shape(self) -> None:
        """Property: Matrix shape must be (r+1) x (k+r+1)."""
        for k in _get_random_k_values():
            for scheme in SCHEMES:
                H, r = generate_h_matrix_secdec(k, scheme)

                expected_rows = r + 1
                expected_cols = k + r + 1
                actual_rows = len(H)
                actual_cols = len(H[0]) if H else 0

                assert actual_rows == expected_rows, (
                    f"k={k}, scheme={scheme}: Expected {expected_rows} rows, got {actual_rows}"
                )
                assert actual_cols == expected_cols, (
                    f"k={k}, scheme={scheme}: Expected {expected_cols} cols, got {actual_cols}"
                )

    def test_property_r_satisfies_hamming_bound(self) -> None:
        """Property: r must satisfy 2^r >= k + r + 1."""
        for k in _get_random_k_values():
            for scheme in SCHEMES:
                _, r = generate_h_matrix_secdec(k, scheme)

                # Check the Hamming bound
                assert (1 << r) >= (k + r + 1), (
                    f"k={k}, scheme={scheme}: "
                    f"2^{r} = {1 << r} < {k + r + 1} = k + r + 1"
                )

                # Also check that r is minimal
                if r > 1:
                    assert (1 << (r - 1)) < (k + (r - 1) + 1), (
                        f"k={k}, scheme={scheme}: r={r} is not minimal"
                    )


class TestSECDEDInvariants:
    """Property tests for SECDED encode/decode invariants."""

    def test_property_encode_decode_roundtrip(self) -> None:
        """Property: encode(decode(codeword)) should return same codeword if no error."""
        rng = random.Random(RANDOM_SEED)

        for k in _get_random_k_values():
            for scheme in SCHEMES:
                H, r = generate_h_matrix_secdec(k, scheme)

                # Test with multiple random data values
                for _ in range(5):
                    data = rng.randint(0, (1 << k) - 1)
                    codeword = encode(data, H, r, k)
                    result = decode(codeword, H, r, k)

                    assert result.corrected_data == data, (
                        f"k={k}, scheme={scheme}, data={data}: "
                        f"Roundtrip failed, got {result.corrected_data}"
                    )
                    assert result.error_type == "none", (
                        f"k={k}, scheme={scheme}: Expected no error, got {result.error_type}"
                    )

    def test_property_single_bit_flip_correctable(self) -> None:
        """Property: Single bit flip should always be correctable."""
        rng = random.Random(RANDOM_SEED)

        for k in _get_random_k_values():
            for scheme in SCHEMES:
                H, r = generate_h_matrix_secdec(k, scheme)
                n = k + r + 1

                # Test with random data
                data = rng.randint(0, (1 << k) - 1)
                codeword = encode(data, H, r, k)

                # Test flipping each bit position
                for pos in range(n):
                    corrupted = inject_error(codeword, [pos])
                    result = decode(corrupted, H, r, k)

                    assert result.corrected_data == data, (
                        f"k={k}, scheme={scheme}, pos={pos}: "
                        f"Single bit flip not corrected. Expected {data}, got {result.corrected_data}"
                    )
                    assert result.error_type == "single_corrected", (
                        f"k={k}, scheme={scheme}, pos={pos}: "
                        f"Expected single_corrected, got {result.error_type}"
                    )

    def test_property_double_bit_flip_detectable(self) -> None:
        """Property: Double bit flip should always be detectable (syndrome != 0, overall_parity == 0)."""
        rng = random.Random(RANDOM_SEED)

        for k in _get_random_k_values():
            for scheme in SCHEMES:
                H, r = generate_h_matrix_secdec(k, scheme)
                n = k + r + 1

                # Test with random data
                data = rng.randint(0, (1 << k) - 1)
                codeword = encode(data, H, r, k)

                # Test some random double bit flip positions
                for _ in range(min(10, n * (n - 1) // 2)):
                    pos1 = rng.randint(0, n - 1)
                    pos2 = rng.randint(0, n - 1)
                    if pos1 == pos2:
                        continue

                    corrupted = inject_error(codeword, [pos1, pos2])
                    result = decode(corrupted, H, r, k)

                    # Double bit error should be detected but not corrected
                    assert result.error_type == "double_detected", (
                        f"k={k}, scheme={scheme}, positions=[{pos1}, {pos2}]: "
                        f"Expected double_detected, got {result.error_type}"
                    )
                    assert result.syndrome != 0, (
                        f"k={k}, scheme={scheme}: "
                        f"Double bit error should have non-zero syndrome"
                    )
                    assert result.overall_parity == 0, (
                        f"k={k}, scheme={scheme}: "
                        f"Double bit error should have overall_parity == 0"
                    )

    def test_property_error_free_syndrome_zero(self) -> None:
        """Property: Syndrome of error-free codeword is always 0."""
        rng = random.Random(RANDOM_SEED)

        for k in _get_random_k_values():
            for scheme in SCHEMES:
                H, r = generate_h_matrix_secdec(k, scheme)

                # Test with multiple random data values
                for _ in range(5):
                    data = rng.randint(0, (1 << k) - 1)
                    codeword = encode(data, H, r, k)
                    syndrome, overall = compute_syndrome(codeword, H, r)

                    assert syndrome == 0, (
                        f"k={k}, scheme={scheme}, data={data}: "
                        f"Error-free codeword has non-zero syndrome: {syndrome}"
                    )
                    assert overall == 0, (
                        f"k={k}, scheme={scheme}, data={data}: "
                        f"Error-free codeword has non-zero overall parity: {overall}"
                    )


class TestXORDepthProperties:
    """Property tests for XOR depth calculations."""

    def test_property_depth_non_negative(self) -> None:
        """Property: depth >= 0 for all parity bits."""
        for k in _get_random_k_values():
            for scheme in SCHEMES:
                H, r = generate_h_matrix_secdec(k, scheme)
                result = calculate_xor_depth(H, r)

                for i, depth in enumerate(result.depths):
                    assert depth >= 0, (
                        f"k={k}, scheme={scheme}: Parity bit {i} has negative depth: {depth}"
                    )

    def test_property_depth_upper_bound(self) -> None:
        """Property: depth <= ceil(log2(k)) (upper bound)."""
        for k in _get_random_k_values():
            for scheme in SCHEMES:
                H, r = generate_h_matrix_secdec(k, scheme)
                result = calculate_xor_depth(H, r)

                upper_bound = math.ceil(math.log2(k)) if k > 1 else 0

                for i, depth in enumerate(result.depths):
                    assert depth <= upper_bound, (
                        f"k={k}, scheme={scheme}: "
                        f"Parity bit {i} depth {depth} exceeds upper bound {upper_bound}"
                    )

    def test_property_hsiao_more_balanced_than_hamming(self) -> None:
        """Property: Hsiao max_depth <= Hamming max_depth (Hsiao is more balanced).

        Hsiao codes distribute 1s more evenly across rows, leading to more uniform
        fan-in per parity bit. This results in lower or equal maximum XOR depth
        compared to Hamming codes which can have unbalanced fan-in distribution.
        """
        for k in _get_random_k_values():
            H_hamming, r_hamming = generate_h_matrix_secdec(k, "hamming")
            H_hsiao, r_hsiao = generate_h_matrix_secdec(k, "hsiao")

            result_hamming = calculate_xor_depth(H_hamming, r_hamming)
            result_hsiao = calculate_xor_depth(H_hsiao, r_hsiao)

            # Hsiao should have equal or lower max depth (more balanced fan-in)
            assert result_hsiao.max_depth <= result_hamming.max_depth, (
                f"k={k}: Hsiao max_depth ({result_hsiao.max_depth}) > "
                f"Hamming max_depth ({result_hamming.max_depth})"
            )

            # Hsiao should have lower variance in depths (more uniform)
            hsiao_variance = max(result_hsiao.depths) - min(result_hsiao.depths)
            hamming_variance = max(result_hamming.depths) - min(result_hamming.depths)
            assert hsiao_variance <= hamming_variance, (
                f"k={k}: Hsiao depth variance ({hsiao_variance}) > "
                f"Hamming depth variance ({hamming_variance})"
            )


class TestDeterminism:
    """Property tests for deterministic behavior."""

    def test_property_same_inputs_same_hmatrix(self) -> None:
        """Property: Same inputs always produce same H-matrix."""
        for k in _get_random_k_values():
            for scheme in SCHEMES:
                H1, r1 = generate_h_matrix_secdec(k, scheme)
                H2, r2 = generate_h_matrix_secdec(k, scheme)

                assert r1 == r2, (
                    f"k={k}, scheme={scheme}: r values differ: {r1} vs {r2}"
                )
                assert H1 == H2, (
                    f"k={k}, scheme={scheme}: H-matrices differ on repeated calls"
                )

    def test_property_same_data_same_codeword(self) -> None:
        """Property: Same data always produces same codeword."""
        rng = random.Random(RANDOM_SEED)

        for k in _get_random_k_values():
            for scheme in SCHEMES:
                H, r = generate_h_matrix_secdec(k, scheme)

                for _ in range(5):
                    data = rng.randint(0, (1 << k) - 1)
                    codeword1 = encode(data, H, r, k)
                    codeword2 = encode(data, H, r, k)

                    assert codeword1 == codeword2, (
                        f"k={k}, scheme={scheme}, data={data}: "
                        f"Codewords differ on repeated encoding"
                    )


class TestEdgeCases:
    """Property tests for edge cases and boundary conditions."""

    def test_property_all_zeros_data(self) -> None:
        """Property: All-zeros data encodes and decodes correctly."""
        for k in _get_random_k_values():
            for scheme in SCHEMES:
                H, r = generate_h_matrix_secdec(k, scheme)
                data = 0

                codeword = encode(data, H, r, k)
                result = decode(codeword, H, r, k)

                assert result.corrected_data == data, (
                    f"k={k}, scheme={scheme}: All-zeros roundtrip failed"
                )

    def test_property_all_ones_data(self) -> None:
        """Property: All-ones data encodes and decodes correctly."""
        for k in _get_random_k_values():
            for scheme in SCHEMES:
                H, r = generate_h_matrix_secdec(k, scheme)
                data = (1 << k) - 1  # All ones

                codeword = encode(data, H, r, k)
                result = decode(codeword, H, r, k)

                assert result.corrected_data == data, (
                    f"k={k}, scheme={scheme}: All-ones roundtrip failed"
                )

    def test_property_alternating_bits_data(self) -> None:
        """Property: Alternating bit patterns encode and decode correctly."""
        for k in _get_random_k_values():
            for scheme in SCHEMES:
                H, r = generate_h_matrix_secdec(k, scheme)

                # 0xAA...AA pattern (10101010...)
                pattern_aa = 0
                for i in range(k):
                    if i % 2 == 1:
                        pattern_aa |= (1 << i)

                codeword = encode(pattern_aa, H, r, k)
                result = decode(codeword, H, r, k)

                assert result.corrected_data == pattern_aa, (
                    f"k={k}, scheme={scheme}: Alternating pattern roundtrip failed"
                )

                # 0x55...55 pattern (01010101...)
                pattern_55 = 0
                for i in range(k):
                    if i % 2 == 0:
                        pattern_55 |= (1 << i)

                codeword = encode(pattern_55, H, r, k)
                result = decode(codeword, H, r, k)

                assert result.corrected_data == pattern_55, (
                    f"k={k}, scheme={scheme}: Alternating pattern (inv) roundtrip failed"
                )
