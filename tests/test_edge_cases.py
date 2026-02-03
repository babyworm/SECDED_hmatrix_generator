"""
Edge cases and boundary condition tests for SECDED H-matrix generator.

Tests boundary data widths, invalid inputs, H-matrix edge cases,
XOR depth edge cases, and SECDED codec edge cases.
"""
import math
import pytest
from typing import List

from src.gen_hmatrix import generate_h_matrix_secdec, Scheme
from src.xor_depth import calculate_xor_depth
from src.secded_codec import encode, decode, inject_error


# =============================================================================
# Test Constants
# =============================================================================

# Powers of 2 data widths
POWERS_OF_2 = [32, 64, 128, 256, 512, 1024, 2048]

# Non-power-of-2 data widths
NON_POWERS_OF_2 = [33, 63, 127, 255, 500, 1000]

# Minimum valid k values per scheme
MIN_K_HSIAO = 4
MIN_K_HAMMING = 8

SCHEMES: List[Scheme] = ["hamming", "hsiao"]


def _column_to_int(H: List[List[int]], col_idx: int, num_rows: int) -> int:
    """Convert a column to an integer (bit mask) for comparison."""
    val = 0
    for row in range(num_rows):
        val |= (H[row][col_idx] << row)
    return val


def _required_r(k: int) -> int:
    """Calculate minimum parity bits for SECDED: 2^r >= k + r + 1"""
    r = 1
    while (1 << r) < (k + r + 1):
        r += 1
    return r


# =============================================================================
# 1. Boundary Data Widths Tests
# =============================================================================

class TestBoundaryDataWidths:
    """Test boundary data width values."""

    @pytest.mark.parametrize("k", POWERS_OF_2)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_powers_of_2(self, k: int, scheme: Scheme):
        """Test that powers of 2 data widths work correctly."""
        H, r = generate_h_matrix_secdec(k, scheme)

        expected_rows = r + 1
        expected_cols = k + r + 1

        assert len(H) == expected_rows
        assert len(H[0]) == expected_cols

    @pytest.mark.parametrize("k", NON_POWERS_OF_2)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_non_powers_of_2(self, k: int, scheme: Scheme):
        """Test that non-power-of-2 data widths work correctly."""
        H, r = generate_h_matrix_secdec(k, scheme)

        expected_rows = r + 1
        expected_cols = k + r + 1

        assert len(H) == expected_rows
        assert len(H[0]) == expected_cols

    def test_minimum_k_hsiao(self):
        """Test minimum valid k value for Hsiao scheme (k=4)."""
        k = MIN_K_HSIAO
        H, r = generate_h_matrix_secdec(k, "hsiao")

        assert len(H) == r + 1
        assert len(H[0]) == k + r + 1

        # Verify the matrix is valid
        sec_cols = []
        for col_idx in range(k + r):
            mask = _column_to_int(H, col_idx, r)
            sec_cols.append(mask)

        # All columns should be unique and non-zero
        assert len(set(sec_cols)) == len(sec_cols)
        assert all(m != 0 for m in sec_cols)

    def test_minimum_k_hamming(self):
        """Test minimum valid k value for Hamming scheme (k=8)."""
        k = MIN_K_HAMMING
        H, r = generate_h_matrix_secdec(k, "hamming")

        assert len(H) == r + 1
        assert len(H[0]) == k + r + 1

        # Verify the matrix is valid
        sec_cols = []
        for col_idx in range(k + r):
            mask = _column_to_int(H, col_idx, r)
            sec_cols.append(mask)

        # All columns should be unique and non-zero
        assert len(set(sec_cols)) == len(sec_cols)
        assert all(m != 0 for m in sec_cols)

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_maximum_k_2048(self, scheme: Scheme):
        """Test maximum k=2048 data width."""
        k = 2048
        H, r = generate_h_matrix_secdec(k, scheme)

        expected_r = _required_r(k)
        assert r == expected_r
        assert len(H) == r + 1
        assert len(H[0]) == k + r + 1

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_very_small_k_hsiao(self, k: int):
        """Test very small k values with Hsiao scheme."""
        # These should work for Hsiao but may fail for Hamming
        H, r = generate_h_matrix_secdec(k, "hsiao")

        assert len(H) == r + 1
        assert len(H[0]) == k + r + 1


# =============================================================================
# 2. Invalid Inputs Tests
# =============================================================================

class TestInvalidInputs:
    """Test invalid input handling."""

    def test_k_zero_raises_value_error(self):
        """k=0 should raise ValueError."""
        with pytest.raises(ValueError, match="k must be positive"):
            generate_h_matrix_secdec(0, "hsiao")

        with pytest.raises(ValueError, match="k must be positive"):
            generate_h_matrix_secdec(0, "hamming")

    def test_k_negative_raises_value_error(self):
        """k=-1 should raise ValueError."""
        with pytest.raises(ValueError, match="k must be positive"):
            generate_h_matrix_secdec(-1, "hsiao")

        with pytest.raises(ValueError, match="k must be positive"):
            generate_h_matrix_secdec(-1, "hamming")

    @pytest.mark.parametrize("k", [-100, -10, -2])
    def test_various_negative_k_values(self, k: int):
        """Various negative k values should raise ValueError."""
        with pytest.raises(ValueError):
            generate_h_matrix_secdec(k, "hsiao")

    def test_invalid_scheme_raises_value_error(self):
        """Invalid scheme names should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scheme"):
            generate_h_matrix_secdec(32, "invalid_scheme")

        with pytest.raises(ValueError, match="Unknown scheme"):
            generate_h_matrix_secdec(32, "HSIAO")  # Case sensitive

        with pytest.raises(ValueError, match="Unknown scheme"):
            generate_h_matrix_secdec(32, "Hamming")  # Case sensitive

        with pytest.raises(ValueError, match="Unknown scheme"):
            generate_h_matrix_secdec(32, "")


# =============================================================================
# 3. H-Matrix Edge Cases Tests
# =============================================================================

class TestHMatrixEdgeCases:
    """Test H-matrix structural edge cases."""

    @pytest.mark.parametrize("k", POWERS_OF_2 + NON_POWERS_OF_2)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_all_sec_columns_unique(self, k: int, scheme: Scheme):
        """Verify all SEC columns are unique (no duplicates)."""
        H, r = generate_h_matrix_secdec(k, scheme)

        # Collect SEC columns (excluding overall parity column)
        sec_col_masks = []
        for col_idx in range(k + r):
            mask = _column_to_int(H, col_idx, r)
            sec_col_masks.append(mask)

        # Check for uniqueness
        assert len(set(sec_col_masks)) == len(sec_col_masks), \
            f"Duplicate columns found for k={k}, scheme={scheme}"

    @pytest.mark.parametrize("k", POWERS_OF_2 + NON_POWERS_OF_2)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_no_zero_columns_in_sec_portion(self, k: int, scheme: Scheme):
        """Verify no zero columns in SEC portion."""
        H, r = generate_h_matrix_secdec(k, scheme)

        for col_idx in range(k + r):
            mask = _column_to_int(H, col_idx, r)
            assert mask != 0, \
                f"Zero column found at index {col_idx} for k={k}, scheme={scheme}"

    @pytest.mark.parametrize("k", POWERS_OF_2 + NON_POWERS_OF_2)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_parity_columns_are_unit_vectors(self, k: int, scheme: Scheme):
        """Verify parity columns are proper unit vectors (identity matrix pattern)."""
        H, r = generate_h_matrix_secdec(k, scheme)

        # First r columns should be unit vectors in top r rows
        for i in range(r):
            expected_mask = 1 << i
            actual_mask = _column_to_int(H, i, r)

            assert actual_mask == expected_mask, \
                f"Parity column {i} is not unit vector: expected {expected_mask}, got {actual_mask}"

            # Also verify each element explicitly
            for row in range(r):
                expected_val = 1 if row == i else 0
                assert H[row][i] == expected_val, \
                    f"Parity column {i}, row {row}: expected {expected_val}, got {H[row][i]}"

    @pytest.mark.parametrize("k", [32, 64, 128])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_overall_parity_column_structure(self, k: int, scheme: Scheme):
        """Verify overall parity column has correct structure."""
        H, r = generate_h_matrix_secdec(k, scheme)
        n = k + r + 1
        last_col = n - 1

        # Top r rows should be 0
        for row in range(r):
            assert H[row][last_col] == 0, \
                f"Overall parity column at row {row} should be 0"

        # Last row should be 1
        assert H[r][last_col] == 1, \
            "Overall parity column at overall row should be 1"

    @pytest.mark.parametrize("k", [32, 64, 128])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_overall_parity_row_all_ones(self, k: int, scheme: Scheme):
        """Verify overall parity row (last row) is all ones."""
        H, r = generate_h_matrix_secdec(k, scheme)
        n = k + r + 1

        for col in range(n):
            assert H[r][col] == 1, \
                f"Overall parity row at column {col} should be 1"


# =============================================================================
# 4. XOR Depth Edge Cases Tests
# =============================================================================

class TestXORDepthEdgeCases:
    """Test XOR depth calculation edge cases."""

    def test_fanin_zero_depth_zero(self):
        """Fan-in of 0 should have depth 0."""
        # Construct a matrix where one parity bit has no data bit coverage
        # r=2, k=1: H[0] has fan-in 0 (no 1s in data portion)
        H = [
            [1, 0, 0, 0],  # p0: data portion [0] -> fan-in 0
            [0, 1, 1, 0],  # p1: data portion [1] -> fan-in 1
            [1, 1, 1, 1]   # overall parity
        ]
        r = 2

        result = calculate_xor_depth(H, r)

        assert result.depths[0] == 0, "Fan-in 0 should have depth 0"

    def test_fanin_one_depth_zero(self):
        """Fan-in of 1 should have depth 0."""
        # r=2, k=2: One parity has fan-in 1
        H = [
            [1, 0, 1, 0, 0],  # p0: data [1,0] -> fan-in 1
            [0, 1, 1, 1, 0],  # p1: data [1,1] -> fan-in 2
            [1, 1, 1, 1, 1]
        ]
        r = 2

        result = calculate_xor_depth(H, r)

        assert result.depths[0] == 0, "Fan-in 1 should have depth 0"
        assert result.depths[1] == 1, "Fan-in 2 should have depth 1"

    @pytest.mark.parametrize("fanin,expected_depth", [
        (0, 0),
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 2),
        (5, 3),
        (7, 3),
        (8, 3),
        (9, 4),
        (15, 4),
        (16, 4),
        (17, 5),
        (31, 5),
        (32, 5),
        (33, 6),
        (64, 6),
        (128, 7),
        (256, 8),
        (512, 9),
        (1024, 10),
    ])
    def test_depth_formula_ceil_log2(self, fanin: int, expected_depth: int):
        """Verify depth formula: ceil(log2(fan_in)) for fan_in > 1, else 0."""
        if fanin <= 1:
            computed = 0
        else:
            computed = (fanin - 1).bit_length()

        assert computed == expected_depth, \
            f"Fan-in {fanin}: expected depth {expected_depth}, got {computed}"

    @pytest.mark.parametrize("k", [32, 64, 128, 256])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_all_depths_match_fanin_formula(self, k: int, scheme: Scheme):
        """Verify all depths match the ceil(log2(fan_in)) formula."""
        H, r = generate_h_matrix_secdec(k, scheme)
        result = calculate_xor_depth(H, r)

        for i in range(r):
            # Count fan-in for this parity bit
            fanin = sum(H[i][r + j] for j in range(k))

            if fanin <= 1:
                expected = 0
            else:
                expected = (fanin - 1).bit_length()

            assert result.depths[i] == expected, \
                f"Row {i}: fan-in={fanin}, expected depth={expected}, got={result.depths[i]}"

    @pytest.mark.parametrize("k", [32, 64, 128])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_max_depth_bounded_by_data_width(self, k: int, scheme: Scheme):
        """Maximum depth should be bounded by ceil(log2(k))."""
        H, r = generate_h_matrix_secdec(k, scheme)
        result = calculate_xor_depth(H, r)

        max_theoretical = math.ceil(math.log2(k)) if k > 1 else 0

        assert result.max_depth <= max_theoretical, \
            f"Max depth {result.max_depth} exceeds theoretical max {max_theoretical}"


# =============================================================================
# 5. SECDED Codec Edge Cases Tests
# =============================================================================

class TestSECDEDCodecEdgeCases:
    """Test SECDED codec edge cases."""

    @pytest.mark.parametrize("k", [32, 64, 128, 256])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_all_zeros_data(self, k: int, scheme: Scheme):
        """Test encoding/decoding all-zeros data."""
        H, r = generate_h_matrix_secdec(k, scheme)

        data = 0
        codeword = encode(data, H, r, k)
        result = decode(codeword, H, r, k)

        assert result.corrected_data == data
        assert result.error_type == "none"
        assert result.syndrome == 0

    @pytest.mark.parametrize("k", [32, 64, 128, 256])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_all_ones_data(self, k: int, scheme: Scheme):
        """Test encoding/decoding all-ones data (max value for bit width)."""
        H, r = generate_h_matrix_secdec(k, scheme)

        data = (1 << k) - 1  # All bits set to 1
        codeword = encode(data, H, r, k)
        result = decode(codeword, H, r, k)

        assert result.corrected_data == data
        assert result.error_type == "none"
        assert result.syndrome == 0

    @pytest.mark.parametrize("k", [32, 64, 128])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_alternating_pattern_0xAA(self, k: int, scheme: Scheme):
        """Test alternating bit pattern 0xAA... (10101010...)."""
        H, r = generate_h_matrix_secdec(k, scheme)

        # Create 0xAA pattern for k bits
        pattern = 0
        for i in range(k):
            if i % 2 == 1:
                pattern |= (1 << i)

        codeword = encode(pattern, H, r, k)
        result = decode(codeword, H, r, k)

        assert result.corrected_data == pattern
        assert result.error_type == "none"
        assert result.syndrome == 0

    @pytest.mark.parametrize("k", [32, 64, 128])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_alternating_pattern_0x55(self, k: int, scheme: Scheme):
        """Test alternating bit pattern 0x55... (01010101...)."""
        H, r = generate_h_matrix_secdec(k, scheme)

        # Create 0x55 pattern for k bits
        pattern = 0
        for i in range(k):
            if i % 2 == 0:
                pattern |= (1 << i)

        codeword = encode(pattern, H, r, k)
        result = decode(codeword, H, r, k)

        assert result.corrected_data == pattern
        assert result.error_type == "none"
        assert result.syndrome == 0

    @pytest.mark.parametrize("k", [32, 64, 128])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_single_bit_set_first_position(self, k: int, scheme: Scheme):
        """Test single bit set at position 0."""
        H, r = generate_h_matrix_secdec(k, scheme)

        data = 1  # Only bit 0 set
        codeword = encode(data, H, r, k)
        result = decode(codeword, H, r, k)

        assert result.corrected_data == data
        assert result.error_type == "none"

    @pytest.mark.parametrize("k", [32, 64, 128])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_single_bit_set_last_position(self, k: int, scheme: Scheme):
        """Test single bit set at last data position."""
        H, r = generate_h_matrix_secdec(k, scheme)

        data = 1 << (k - 1)  # Only MSB set
        codeword = encode(data, H, r, k)
        result = decode(codeword, H, r, k)

        assert result.corrected_data == data
        assert result.error_type == "none"

    @pytest.mark.parametrize("k", [32, 64, 128])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_single_bit_set_middle_position(self, k: int, scheme: Scheme):
        """Test single bit set at middle position."""
        H, r = generate_h_matrix_secdec(k, scheme)

        data = 1 << (k // 2)  # Middle bit set
        codeword = encode(data, H, r, k)
        result = decode(codeword, H, r, k)

        assert result.corrected_data == data
        assert result.error_type == "none"

    @pytest.mark.parametrize("k", [32, 64])
    @pytest.mark.parametrize("scheme", SCHEMES)
    @pytest.mark.parametrize("bit_pos", [0, 1, 7, 15, 16, 31])
    def test_single_bit_set_various_positions(
        self, k: int, scheme: Scheme, bit_pos: int
    ):
        """Test single bit set at various positions."""
        if bit_pos >= k:
            pytest.skip(f"bit_pos {bit_pos} >= k {k}")

        H, r = generate_h_matrix_secdec(k, scheme)

        data = 1 << bit_pos
        codeword = encode(data, H, r, k)
        result = decode(codeword, H, r, k)

        assert result.corrected_data == data
        assert result.error_type == "none"

    @pytest.mark.parametrize("k", [32, 64])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_error_correction_all_zeros(self, k: int, scheme: Scheme):
        """Test error correction on all-zeros data."""
        H, r = generate_h_matrix_secdec(k, scheme)
        n = k + r + 1

        data = 0
        codeword = encode(data, H, r, k)

        # Test single bit error at various positions
        for pos in [0, r, r + k // 2, n - 2, n - 1]:
            corrupted = inject_error(codeword, [pos])
            result = decode(corrupted, H, r, k)

            assert result.corrected_data == data, \
                f"Failed to correct error at position {pos}"
            assert result.error_type == "single_corrected"

    @pytest.mark.parametrize("k", [32, 64])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_error_correction_all_ones(self, k: int, scheme: Scheme):
        """Test error correction on all-ones data."""
        H, r = generate_h_matrix_secdec(k, scheme)
        n = k + r + 1

        data = (1 << k) - 1
        codeword = encode(data, H, r, k)

        # Test single bit error at various positions
        for pos in [0, r, r + k // 2, n - 2, n - 1]:
            corrupted = inject_error(codeword, [pos])
            result = decode(corrupted, H, r, k)

            assert result.corrected_data == data, \
                f"Failed to correct error at position {pos}"
            assert result.error_type == "single_corrected"

    @pytest.mark.parametrize("k", [32, 64])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_double_error_detection_all_zeros(self, k: int, scheme: Scheme):
        """Test double error detection on all-zeros data."""
        H, r = generate_h_matrix_secdec(k, scheme)

        data = 0
        codeword = encode(data, H, r, k)

        # Inject double error
        corrupted = inject_error(codeword, [0, 1])
        result = decode(corrupted, H, r, k)

        assert result.error_type == "double_detected"
        assert result.syndrome != 0

    @pytest.mark.parametrize("k", [32, 64])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_double_error_detection_all_ones(self, k: int, scheme: Scheme):
        """Test double error detection on all-ones data."""
        H, r = generate_h_matrix_secdec(k, scheme)

        data = (1 << k) - 1
        codeword = encode(data, H, r, k)

        # Inject double error
        corrupted = inject_error(codeword, [0, 1])
        result = decode(corrupted, H, r, k)

        assert result.error_type == "double_detected"
        assert result.syndrome != 0


# =============================================================================
# Additional Edge Case Tests
# =============================================================================

class TestAdditionalEdgeCases:
    """Additional edge case tests."""

    @pytest.mark.parametrize("k", [32, 64, 128])
    def test_hamming_vs_hsiao_produce_different_matrices(self, k: int):
        """Verify Hamming and Hsiao produce different H-matrices."""
        H_hamming, r_hamming = generate_h_matrix_secdec(k, "hamming")
        H_hsiao, r_hsiao = generate_h_matrix_secdec(k, "hsiao")

        # r values should be the same
        assert r_hamming == r_hsiao

        # But data columns should differ
        hamming_data_cols = [
            _column_to_int(H_hamming, r_hamming + i, r_hamming)
            for i in range(k)
        ]
        hsiao_data_cols = [
            _column_to_int(H_hsiao, r_hsiao + i, r_hsiao)
            for i in range(k)
        ]

        # Sets should have same elements but order may differ
        # Actually for different schemes the sets themselves may differ
        hamming_set = set(hamming_data_cols)
        hsiao_set = set(hsiao_data_cols)

        # At minimum, the ordering should be different
        assert hamming_data_cols != hsiao_data_cols, \
            "Hamming and Hsiao should produce different column orderings"

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_deterministic_generation(self, scheme: Scheme):
        """Verify matrix generation is deterministic."""
        k = 64

        H1, r1 = generate_h_matrix_secdec(k, scheme)
        H2, r2 = generate_h_matrix_secdec(k, scheme)
        H3, r3 = generate_h_matrix_secdec(k, scheme)

        assert r1 == r2 == r3
        assert H1 == H2 == H3

    @pytest.mark.parametrize("k", [32, 64, 128, 256, 512, 1024, 2048])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_r_value_correctness(self, k: int, scheme: Scheme):
        """Verify r value satisfies 2^r >= k + r + 1."""
        H, r = generate_h_matrix_secdec(k, scheme)

        # Verify constraint
        assert (1 << r) >= (k + r + 1), \
            f"r={r} does not satisfy 2^r >= k+r+1 for k={k}"

        # Verify r is minimal
        if r > 1:
            assert (1 << (r - 1)) < (k + (r - 1) + 1), \
                f"r={r} is not minimal for k={k}"

    @pytest.mark.parametrize("k", [63, 127, 255, 511, 1023])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_boundary_r_transitions(self, k: int, scheme: Scheme):
        """Test k values at r transition boundaries (2^r - r - 1)."""
        H, r = generate_h_matrix_secdec(k, scheme)

        # These k values are just below powers of 2, testing r boundaries
        expected_r = _required_r(k)
        assert r == expected_r

        # Also test k+1
        H_next, r_next = generate_h_matrix_secdec(k + 1, scheme)
        expected_r_next = _required_r(k + 1)
        assert r_next == expected_r_next

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_inject_error_empty_positions(self, scheme: Scheme):
        """Test inject_error with empty positions list."""
        H, r = generate_h_matrix_secdec(32, scheme)
        codeword = encode(0x12345678, H, r, 32)

        # Inject no errors
        result = inject_error(codeword, [])

        assert result == codeword

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_inject_error_out_of_bounds(self, scheme: Scheme):
        """Test inject_error with out-of-bounds positions (should be ignored)."""
        H, r = generate_h_matrix_secdec(32, scheme)
        codeword = encode(0x12345678, H, r, 32)
        n = len(codeword)

        # Inject error at out-of-bounds position
        result = inject_error(codeword, [n, n + 1, n + 100])

        # Should be unchanged since all positions are out of bounds
        assert result == codeword

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_inject_error_same_position_twice(self, scheme: Scheme):
        """Test inject_error with same position twice (should cancel out)."""
        H, r = generate_h_matrix_secdec(32, scheme)
        codeword = encode(0x12345678, H, r, 32)

        # Inject error at same position twice - should cancel
        result = inject_error(codeword, [5, 5])

        assert result == codeword
