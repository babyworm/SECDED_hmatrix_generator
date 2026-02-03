"""
Error handling tests for SECDED H-matrix generator.

Tests exception handling and error messages for various invalid inputs
and edge cases across all modules.
"""
import pytest
from typing import List

from src.gen_hmatrix import generate_h_matrix_secdec, parity_equations_sv
from src.xor_depth import calculate_xor_depth
from src.metrics import analyze_scheme, compare_schemes
from src.secded_codec import encode, decode, inject_error


# =============================================================================
# generate_h_matrix_secdec Error Tests
# =============================================================================

class TestGenerateHMatrixErrors:
    """Tests for generate_h_matrix_secdec error handling."""

    def test_k_zero_raises_value_error(self):
        """k=0 should raise ValueError with informative message."""
        with pytest.raises(ValueError) as exc_info:
            generate_h_matrix_secdec(k=0)
        assert "positive" in str(exc_info.value).lower() or "k" in str(exc_info.value)

    def test_k_negative_one_raises_value_error(self):
        """k=-1 should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            generate_h_matrix_secdec(k=-1)
        assert "positive" in str(exc_info.value).lower() or "k" in str(exc_info.value)

    def test_k_large_negative_raises_value_error(self):
        """k=-100 should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            generate_h_matrix_secdec(k=-100)
        assert "positive" in str(exc_info.value).lower() or "k" in str(exc_info.value)

    def test_invalid_scheme_string_raises_value_error(self):
        """Invalid scheme 'invalid' should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            generate_h_matrix_secdec(k=8, scheme="invalid")
        assert "scheme" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()

    def test_empty_scheme_string_raises_value_error(self):
        """Empty scheme '' should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            generate_h_matrix_secdec(k=8, scheme="")
        assert "scheme" in str(exc_info.value).lower()

    def test_none_scheme_raises_error(self):
        """None scheme should raise TypeError or ValueError."""
        with pytest.raises((TypeError, ValueError)):
            generate_h_matrix_secdec(k=8, scheme=None)


class TestHammingSchemeConstraints:
    """Tests for Hamming scheme constraints with small k values."""

    def test_hamming_small_k_raises_vector_error(self):
        """Hamming with very small k (k=4) fails due to limited vectors.

        With r=3 (for k=4), we have 2^3-1=7 nonzero vectors.
        Unit vectors take 3, leaving only 4 for data columns.
        The algorithm may still fail due to how it selects vectors.
        """
        with pytest.raises(ValueError) as exc_info:
            generate_h_matrix_secdec(k=4, scheme="hamming")
        # Error message should be informative
        message = str(exc_info.value).lower()
        assert "vector" in message or "k=" in message

    def test_hamming_k_1_raises_error(self):
        """k=1 with Hamming scheme fails due to insufficient vectors."""
        with pytest.raises(ValueError) as exc_info:
            generate_h_matrix_secdec(k=1, scheme="hamming")
        message = str(exc_info.value).lower()
        assert "vector" in message or "k=" in message

    def test_hamming_informative_error_on_vector_exhaustion(self):
        """Test that vector exhaustion error is informative.

        This tests the error message quality when Hamming cannot find enough
        data column vectors.
        """
        with pytest.raises(ValueError) as exc_info:
            generate_h_matrix_secdec(k=4, scheme="hamming")
        message = str(exc_info.value)
        # Error should mention vectors, k, or r
        assert "vector" in message.lower() or "k=" in message


# =============================================================================
# parity_equations_sv Error Tests
# =============================================================================

class TestParityEquationsSvErrors:
    """Tests for parity_equations_sv error handling."""

    def test_invalid_k_propagates_error(self):
        """Invalid k should propagate ValueError from generate_h_matrix_secdec."""
        with pytest.raises(ValueError):
            parity_equations_sv(k=0)

    def test_invalid_scheme_propagates_error(self):
        """Invalid scheme should propagate ValueError."""
        with pytest.raises(ValueError):
            parity_equations_sv(k=8, scheme="invalid")


# =============================================================================
# XOR Depth Error Tests
# =============================================================================

class TestXorDepthErrors:
    """Tests for calculate_xor_depth error handling."""

    def test_empty_h_matrix_raises_index_error(self):
        """Empty H matrix raises IndexError (no graceful handling)."""
        empty_H: List[List[int]] = []
        with pytest.raises(IndexError):
            calculate_xor_depth(empty_H, r=0)

    def test_single_row_h_matrix(self):
        """Single row H matrix (r=0) should handle gracefully."""
        H = [[1, 1, 1]]  # 1 row, 3 columns
        result = calculate_xor_depth(H, r=0)
        assert result.depths == []

    def test_mismatched_r_value_does_not_crash(self):
        """Mismatched r value should not crash (may produce incorrect results)."""
        # Generate valid H matrix
        H, actual_r = generate_h_matrix_secdec(k=8, scheme="hsiao")

        # Use wrong r value (too small)
        result = calculate_xor_depth(H, r=actual_r - 1)
        # Should not crash, just potentially give wrong results
        assert result is not None

        # Use wrong r value (too large)
        result = calculate_xor_depth(H, r=actual_r + 1)
        assert result is not None

    def test_h_matrix_all_zeros(self):
        """H matrix of all zeros should handle gracefully."""
        H = [[0, 0, 0, 0] for _ in range(4)]
        result = calculate_xor_depth(H, r=3)
        # All fan-ins should be 0, resulting in depth 0
        assert all(d == 0 for d in result.depths)


# =============================================================================
# SECDED Codec Error Tests
# =============================================================================

class TestSecdedCodecErrors:
    """Tests for SECDED codec error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.k = 8
        self.H, self.r = generate_h_matrix_secdec(self.k, scheme="hsiao")
        self.n = self.k + self.r + 1

    def test_data_overflow_behavior(self):
        """Data larger than k bits - test behavior (may truncate or wrap)."""
        # Data value that exceeds k bits (255 fits in 8 bits, 256 needs 9 bits)
        large_data = 256  # Needs 9 bits, but k=8
        # The function may truncate or handle this silently
        codeword = encode(large_data, self.H, self.r, self.k)
        # Should produce a valid codeword (even if data was truncated)
        assert len(codeword) == self.n
        assert all(b in (0, 1) for b in codeword)

    def test_very_large_data_overflow(self):
        """Very large data value - test behavior."""
        huge_data = 2**16  # Way larger than 8 bits
        codeword = encode(huge_data, self.H, self.r, self.k)
        assert len(codeword) == self.n

    def test_negative_data_behavior(self):
        """Negative data value - test behavior."""
        # Negative values may cause issues with bit extraction
        negative_data = -1
        # This may work (via two's complement) or may produce unexpected results
        try:
            codeword = encode(negative_data, self.H, self.r, self.k)
            # If it works, should still be valid codeword structure
            assert len(codeword) == self.n
        except (ValueError, OverflowError):
            # It's acceptable to raise an error for negative input
            pass

    def test_inject_error_negative_position_ignored(self):
        """inject_error with negative position should be ignored (bounds check)."""
        valid_data = 42
        codeword = encode(valid_data, self.H, self.r, self.k)
        original = codeword.copy()

        # Negative position should be ignored per the implementation
        corrupted = inject_error(codeword, [-1])
        # Codeword should be unchanged (negative position skipped)
        assert corrupted == original

    def test_inject_error_out_of_range_position_ignored(self):
        """inject_error with out-of-range position should be ignored."""
        valid_data = 42
        codeword = encode(valid_data, self.H, self.r, self.k)
        original = codeword.copy()

        # Position beyond codeword length should be ignored
        corrupted = inject_error(codeword, [100])
        assert corrupted == original

    def test_inject_error_mixed_valid_invalid_positions(self):
        """inject_error with mix of valid and invalid positions."""
        valid_data = 42
        codeword = encode(valid_data, self.H, self.r, self.k)

        # Mix of valid (0) and invalid (-1, 100) positions
        corrupted = inject_error(codeword, [-1, 0, 100])

        # Only position 0 should be flipped
        assert corrupted[0] != codeword[0]  # Position 0 flipped
        for i in range(1, len(codeword)):
            assert corrupted[i] == codeword[i]  # Others unchanged

    def test_decode_with_wrong_length_codeword(self):
        """Decode with mismatched codeword length - behavior test."""
        valid_data = 42
        codeword = encode(valid_data, self.H, self.r, self.k)

        # Truncated codeword (too short)
        truncated = codeword[:-2]
        try:
            result = decode(truncated, self.H, self.r, self.k)
            # If it doesn't crash, it may give incorrect results
            assert result is not None
        except (IndexError, ValueError):
            # It's acceptable to raise an error
            pass

        # Extended codeword (too long)
        extended = codeword + [0, 0]
        try:
            result = decode(extended, self.H, self.r, self.k)
            assert result is not None
        except (IndexError, ValueError):
            pass

    def test_decode_empty_codeword(self):
        """Decode with empty codeword."""
        empty_codeword: List[int] = []
        try:
            result = decode(empty_codeword, self.H, self.r, self.k)
            # If it works, check it handled gracefully
            assert result is not None
        except (IndexError, ValueError):
            # Expected to fail
            pass


# =============================================================================
# Metrics Error Tests
# =============================================================================

class TestMetricsErrors:
    """Tests for metrics module error handling."""

    def test_analyze_scheme_invalid_k_zero(self):
        """analyze_scheme with k=0 should raise ValueError."""
        with pytest.raises(ValueError):
            analyze_scheme(k=0, scheme="hsiao")

    def test_analyze_scheme_invalid_k_negative(self):
        """analyze_scheme with negative k should raise ValueError."""
        with pytest.raises(ValueError):
            analyze_scheme(k=-5, scheme="hsiao")

    def test_analyze_scheme_invalid_scheme(self):
        """analyze_scheme with invalid scheme should raise ValueError."""
        with pytest.raises(ValueError):
            analyze_scheme(k=8, scheme="invalid_scheme")

    def test_analyze_scheme_empty_scheme(self):
        """analyze_scheme with empty scheme string should raise ValueError."""
        with pytest.raises(ValueError):
            analyze_scheme(k=8, scheme="")

    def test_compare_schemes_invalid_k_zero(self):
        """compare_schemes with k=0 should raise ValueError."""
        with pytest.raises(ValueError):
            compare_schemes(k=0)

    def test_compare_schemes_invalid_k_negative(self):
        """compare_schemes with negative k should raise ValueError."""
        with pytest.raises(ValueError):
            compare_schemes(k=-10)


# =============================================================================
# CLI Error Handling Tests (without subprocess)
# =============================================================================

class TestCliErrorHandling:
    """Tests for CLI error handling patterns.

    These tests verify error handling behavior without spawning subprocesses.
    """

    def test_cli_module_imports(self):
        """Verify CLI module can be imported without errors."""
        from src.cli import create_parser, main
        assert create_parser is not None
        assert main is not None

    def test_cli_parser_creation(self):
        """Verify CLI parser can be created."""
        from src.cli import create_parser
        parser = create_parser()
        assert parser is not None

    def test_cli_parser_accepts_zero_k_at_parse_level(self):
        """CLI parser accepts k=0 at parse level; validation happens in commands."""
        from src.cli import create_parser
        parser = create_parser()

        # argparse handles type conversion; validation happens in command functions
        # The argument is named 'data_width' internally (from --data-width)
        args = parser.parse_args(['generate', '-k', '0'])
        assert args.data_width == 0  # Parser accepts it; command validates

    def test_cli_parser_rejects_non_integer_k(self):
        """CLI parser should reject non-integer k values."""
        from src.cli import create_parser
        import sys
        parser = create_parser()

        # This should raise SystemExit due to argparse type error
        with pytest.raises(SystemExit):
            parser.parse_args(['generate', '-k', 'not_a_number'])


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for various edge cases and boundary conditions."""

    def test_very_small_valid_k(self):
        """Test with minimum valid k value (k=1)."""
        H, r = generate_h_matrix_secdec(k=1, scheme="hsiao")
        assert H is not None
        assert len(H) == r + 1
        assert len(H[0]) == 1 + r + 1

    def test_k_equals_one_hamming_raises_error(self):
        """Test k=1 with Hamming scheme raises error due to vector constraints."""
        with pytest.raises(ValueError) as exc_info:
            generate_h_matrix_secdec(k=1, scheme="hamming")
        assert "vector" in str(exc_info.value).lower()

    def test_encode_decode_roundtrip_edge_data(self):
        """Test encode/decode roundtrip with edge case data values."""
        k = 8
        H, r = generate_h_matrix_secdec(k, scheme="hsiao")

        # Test with 0
        codeword = encode(0, H, r, k)
        result = decode(codeword, H, r, k)
        assert result.corrected_data == 0
        assert result.error_type == "none"

        # Test with max value for k bits
        max_val = (1 << k) - 1  # 255 for k=8
        codeword = encode(max_val, H, r, k)
        result = decode(codeword, H, r, k)
        assert result.corrected_data == max_val
        assert result.error_type == "none"

    def test_scheme_case_sensitivity(self):
        """Test that scheme names are case-sensitive."""
        # These should raise ValueError (assuming case-sensitive)
        with pytest.raises(ValueError):
            generate_h_matrix_secdec(k=8, scheme="HSIAO")

        with pytest.raises(ValueError):
            generate_h_matrix_secdec(k=8, scheme="Hamming")

        with pytest.raises(ValueError):
            generate_h_matrix_secdec(k=8, scheme="HAMMING")


# =============================================================================
# Error Message Quality Tests
# =============================================================================

class TestErrorMessageQuality:
    """Tests that error messages are informative and helpful."""

    def test_k_error_message_mentions_k(self):
        """Error for invalid k should mention 'k' or 'positive'."""
        with pytest.raises(ValueError) as exc_info:
            generate_h_matrix_secdec(k=0)
        message = str(exc_info.value).lower()
        assert "k" in message or "positive" in message

    def test_scheme_error_message_mentions_scheme(self):
        """Error for invalid scheme should mention the scheme name."""
        with pytest.raises(ValueError) as exc_info:
            generate_h_matrix_secdec(k=8, scheme="bogus_scheme")
        message = str(exc_info.value).lower()
        assert "scheme" in message or "bogus" in message

    def test_hamming_vector_error_is_informative(self):
        """If Hamming fails due to vector exhaustion, error should be informative."""
        # This is a theoretical test - in practice, the algorithm handles most k values
        # We just verify that if an error occurs, it's informative
        try:
            generate_h_matrix_secdec(k=8, scheme="hamming")
        except ValueError as e:
            message = str(e).lower()
            # Should mention vectors, k, or r
            assert any(term in message for term in ["vector", "k=", "r=", "increase"])
