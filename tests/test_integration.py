"""
Integration tests for SECDED H-matrix generator.
Tests end-to-end workflows combining multiple modules.
"""
import pytest
import random
from typing import List

from src.gen_hmatrix import generate_h_matrix_secdec, parity_equations_sv, Scheme
from src.xor_depth import calculate_xor_depth
from src.metrics import analyze_scheme, compare_schemes, generate_comparison_table
from src.secded_codec import encode, decode, inject_error


# Test parameters
DATA_WIDTHS = [32, 64, 128, 256, 512, 1024, 2048]
SCHEMES: List[Scheme] = ["hamming", "hsiao"]


def _generate_random_data(width: int) -> int:
    """Generate random data of specified bit width."""
    return random.randint(0, (1 << width) - 1)


# =============================================================================
# 1. Full Encoding/Decoding Workflow
# =============================================================================


class TestFullEncodingDecodingWorkflow:
    """Test end-to-end encoding and decoding workflow."""

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_generate_encode_decode_verify(self, data_width: int, scheme: Scheme):
        """Generate H-matrix, encode data, decode, verify original data recovered."""
        # Generate H-matrix
        H, r = generate_h_matrix_secdec(data_width, scheme)

        # Generate random test data
        original_data = _generate_random_data(data_width)

        # Encode
        codeword = encode(original_data, H, r, data_width)

        # Decode
        result = decode(codeword, H, r, data_width)

        # Verify
        assert result.corrected_data == original_data
        assert result.error_type == "none"
        assert result.syndrome == 0

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_multiple_data_patterns(self, data_width: int, scheme: Scheme):
        """Test encode/decode workflow with multiple data patterns."""
        H, r = generate_h_matrix_secdec(data_width, scheme)

        test_patterns = [
            0,                              # All zeros
            (1 << data_width) - 1,          # All ones
            0xAAAAAAAAAAAAAAAA & ((1 << data_width) - 1),  # Alternating 10
            0x5555555555555555 & ((1 << data_width) - 1),  # Alternating 01
            _generate_random_data(data_width),  # Random
        ]

        for pattern in test_patterns:
            codeword = encode(pattern, H, r, data_width)
            result = decode(codeword, H, r, data_width)
            assert result.corrected_data == pattern
            assert result.error_type == "none"


# =============================================================================
# 2. Error Injection and Recovery
# =============================================================================


class TestErrorInjectionAndRecovery:
    """Test error injection and recovery workflow."""

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_single_bit_error_data_positions(self, data_width: int, scheme: Scheme):
        """Full workflow: Generate, Encode, Inject error at data positions, Decode, Verify."""
        H, r = generate_h_matrix_secdec(data_width, scheme)
        original_data = _generate_random_data(data_width)
        codeword = encode(original_data, H, r, data_width)

        # Test a sample of data bit positions (positions r to r+data_width-1)
        num_samples = min(10, data_width)
        data_positions = random.sample(range(r, r + data_width), num_samples)

        for pos in data_positions:
            corrupted = inject_error(codeword, [pos])
            result = decode(corrupted, H, r, data_width)

            assert result.error_type == "single_corrected"
            assert result.corrected_data == original_data
            assert result.corrected is True

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_single_bit_error_parity_positions(self, data_width: int, scheme: Scheme):
        """Full workflow: Generate, Encode, Inject error at parity positions, Decode, Verify."""
        H, r = generate_h_matrix_secdec(data_width, scheme)
        original_data = _generate_random_data(data_width)
        codeword = encode(original_data, H, r, data_width)

        # Test all parity bit positions (positions 0 to r-1)
        for pos in range(r):
            corrupted = inject_error(codeword, [pos])
            result = decode(corrupted, H, r, data_width)

            assert result.error_type == "single_corrected"
            assert result.corrected_data == original_data
            assert result.corrected is True

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_double_bit_error_detected_not_corrected(self, data_width: int, scheme: Scheme):
        """Verify double bit errors are detected but not corrected."""
        H, r = generate_h_matrix_secdec(data_width, scheme)
        original_data = _generate_random_data(data_width)
        codeword = encode(original_data, H, r, data_width)
        codeword_length = len(codeword)

        # Test several double-bit error patterns
        for _ in range(10):
            positions = random.sample(range(codeword_length), 2)
            corrupted = inject_error(codeword, positions)
            result = decode(corrupted, H, r, data_width)

            assert result.error_type == "double_detected"
            assert result.syndrome != 0

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_overall_parity_bit_error_correction(self, data_width: int, scheme: Scheme):
        """Test error correction when only overall parity bit is flipped."""
        H, r = generate_h_matrix_secdec(data_width, scheme)
        original_data = _generate_random_data(data_width)
        codeword = encode(original_data, H, r, data_width)

        # Overall parity is the last bit
        overall_parity_pos = len(codeword) - 1
        corrupted = inject_error(codeword, [overall_parity_pos])
        result = decode(corrupted, H, r, data_width)

        assert result.error_type == "single_corrected"
        assert result.corrected_data == original_data
        assert result.syndrome == 0
        assert result.overall_parity == 1


# =============================================================================
# 3. Metrics Consistency
# =============================================================================


class TestMetricsConsistency:
    """Test metrics consistency across modules."""

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_analyze_scheme_consistent_with_calculate_xor_depth(
        self, data_width: int, scheme: Scheme
    ):
        """Verify analyze_scheme produces consistent results with calculate_xor_depth."""
        # Get metrics from analyze_scheme
        metrics = analyze_scheme(data_width, scheme)

        # Calculate XOR depth directly
        H, r = generate_h_matrix_secdec(data_width, scheme)
        xor_result = calculate_xor_depth(H, r)

        # Verify consistency
        assert metrics.max_xor_depth == xor_result.max_depth
        assert metrics.min_xor_depth == xor_result.min_depth
        assert abs(metrics.avg_xor_depth - xor_result.avg_depth) < 1e-10

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    def test_compare_schemes_contains_both_results(self, data_width: int):
        """Verify compare_schemes contains both hamming and hsiao results."""
        results = compare_schemes(data_width)

        assert "hamming" in results
        assert "hsiao" in results
        assert results["hamming"].scheme == "hamming"
        assert results["hsiao"].scheme == "hsiao"
        assert results["hamming"].k == data_width
        assert results["hsiao"].k == data_width

    def test_generate_comparison_table_produces_valid_output(self):
        """Verify generate_comparison_table produces valid output string."""
        k_values = [32, 64, 128]
        table = generate_comparison_table(k_values)

        assert isinstance(table, str)
        assert len(table) > 0
        assert "SECDED Scheme Comparison" in table

        # Verify table contains info for each k value
        for k in k_values:
            assert f"Data bits (k): {k}" in table

        # Verify table contains metric headers
        assert "Max XOR depth" in table or "max_xor_depth" in table.lower()
        assert "Fan-in" in table or "fanin" in table.lower()

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_metrics_values_are_reasonable(self, data_width: int, scheme: Scheme):
        """Verify metrics values are within reasonable bounds."""
        metrics = analyze_scheme(data_width, scheme)

        # Basic sanity checks
        assert metrics.r > 0
        assert metrics.total_bits == data_width + metrics.r + 1
        assert metrics.overhead_percent > 0
        assert metrics.max_fanin >= metrics.min_fanin
        assert metrics.max_xor_depth >= metrics.min_xor_depth
        assert metrics.avg_fanin >= metrics.min_fanin
        assert metrics.avg_fanin <= metrics.max_fanin


# =============================================================================
# 4. SystemVerilog Generation Workflow
# =============================================================================


class TestSystemVerilogGenerationWorkflow:
    """Test SystemVerilog code generation workflow."""

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_generate_sv_code_structure(self, data_width: int, scheme: Scheme):
        """Generate H-matrix, generate SV code, verify code structure."""
        H, r = generate_h_matrix_secdec(data_width, scheme)
        sv_code = parity_equations_sv(data_width, scheme)

        # Verify basic structure
        assert isinstance(sv_code, str)
        assert len(sv_code) > 0

        # Verify header comments
        assert scheme.upper() in sv_code
        assert f"k={data_width}" in sv_code
        assert f"r={r}" in sv_code

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_assign_statements_present(self, data_width: int, scheme: Scheme):
        """Verify assign statements present for each parity bit."""
        H, r = generate_h_matrix_secdec(data_width, scheme)
        sv_code = parity_equations_sv(data_width, scheme)

        # Verify assign statement for each parity bit
        for i in range(r):
            assert f"assign p[{i}]" in sv_code

        # Verify overall parity assignment
        assert "assign p_overall" in sv_code

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_data_bit_references_correct(self, data_width: int, scheme: Scheme):
        """Verify correct data bit references in XOR equations."""
        H, r = generate_h_matrix_secdec(data_width, scheme)
        sv_code = parity_equations_sv(data_width, scheme)

        # Verify data bit references exist
        for line in sv_code.split("\n"):
            if line.startswith("assign p["):
                # Each parity line should reference d[] bits or be 1'b0
                assert "d[" in line or "1'b0" in line

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_custom_signal_names(self, scheme: Scheme):
        """Verify custom signal names are used correctly."""
        data_width = 32
        d_name = "data_in"
        p_name = "parity"
        po_name = "overall"

        sv_code = parity_equations_sv(
            data_width, scheme, d_name=d_name, p_name=p_name, po_name=po_name
        )

        assert f"{d_name}[" in sv_code
        assert f"{p_name}[" in sv_code
        assert f"assign {po_name}" in sv_code


# =============================================================================
# 5. Round-Trip Consistency
# =============================================================================


class TestRoundTripConsistency:
    """Test reproducibility and consistency of operations."""

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_h_matrix_identical_multiple_generations(
        self, data_width: int, scheme: Scheme
    ):
        """Generate same H-matrix multiple times, verify identical."""
        H1, r1 = generate_h_matrix_secdec(data_width, scheme)
        H2, r2 = generate_h_matrix_secdec(data_width, scheme)
        H3, r3 = generate_h_matrix_secdec(data_width, scheme)

        assert r1 == r2 == r3
        assert H1 == H2 == H3

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_encode_identical_codewords(self, data_width: int, scheme: Scheme):
        """Encode same data multiple times, verify identical codewords."""
        H, r = generate_h_matrix_secdec(data_width, scheme)
        data = _generate_random_data(data_width)

        codeword1 = encode(data, H, r, data_width)
        codeword2 = encode(data, H, r, data_width)
        codeword3 = encode(data, H, r, data_width)

        assert codeword1 == codeword2 == codeword3

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_metrics_reproducible(self, data_width: int, scheme: Scheme):
        """Verify metrics are reproducible across multiple calls."""
        metrics1 = analyze_scheme(data_width, scheme)
        metrics2 = analyze_scheme(data_width, scheme)

        assert metrics1.r == metrics2.r
        assert metrics1.total_bits == metrics2.total_bits
        assert metrics1.max_fanin == metrics2.max_fanin
        assert metrics1.min_fanin == metrics2.min_fanin
        assert metrics1.max_xor_depth == metrics2.max_xor_depth
        assert metrics1.min_xor_depth == metrics2.min_xor_depth
        assert abs(metrics1.avg_fanin - metrics2.avg_fanin) < 1e-10
        assert abs(metrics1.avg_xor_depth - metrics2.avg_xor_depth) < 1e-10

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_sv_code_reproducible(self, scheme: Scheme):
        """Verify SystemVerilog code is reproducible."""
        data_width = 64

        sv1 = parity_equations_sv(data_width, scheme)
        sv2 = parity_equations_sv(data_width, scheme)
        sv3 = parity_equations_sv(data_width, scheme)

        assert sv1 == sv2 == sv3


# =============================================================================
# 6. Cross-Scheme Comparison
# =============================================================================


class TestCrossSchemeComparison:
    """Test cross-scheme comparisons and properties."""

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    def test_both_schemes_produce_valid_h_matrices(self, data_width: int):
        """For same k, verify both schemes produce valid H-matrices."""
        H_hamming, r_hamming = generate_h_matrix_secdec(data_width, "hamming")
        H_hsiao, r_hsiao = generate_h_matrix_secdec(data_width, "hsiao")

        # Both should have same r for same k
        assert r_hamming == r_hsiao

        # Both should have same dimensions
        assert len(H_hamming) == len(H_hsiao)
        assert len(H_hamming[0]) == len(H_hsiao[0])

        # Verify dimensions: (r+1) x (k+r+1)
        expected_rows = r_hamming + 1
        expected_cols = data_width + r_hamming + 1
        assert len(H_hamming) == expected_rows
        assert len(H_hamming[0]) == expected_cols

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    def test_both_schemes_encode_decode_same_data(self, data_width: int):
        """Verify both schemes can encode/decode same data correctly."""
        H_hamming, r_hamming = generate_h_matrix_secdec(data_width, "hamming")
        H_hsiao, r_hsiao = generate_h_matrix_secdec(data_width, "hsiao")

        # Test with several random data values
        for _ in range(5):
            original_data = _generate_random_data(data_width)

            # Encode with both schemes
            codeword_hamming = encode(original_data, H_hamming, r_hamming, data_width)
            codeword_hsiao = encode(original_data, H_hsiao, r_hsiao, data_width)

            # Decode with both schemes
            result_hamming = decode(codeword_hamming, H_hamming, r_hamming, data_width)
            result_hsiao = decode(codeword_hsiao, H_hsiao, r_hsiao, data_width)

            # Both should recover original data
            assert result_hamming.corrected_data == original_data
            assert result_hsiao.corrected_data == original_data
            assert result_hamming.error_type == "none"
            assert result_hsiao.error_type == "none"

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    def test_hsiao_xor_depth_characteristics(self, data_width: int):
        """Verify Hsiao has better or equal XOR depth characteristics."""
        hamming_metrics = analyze_scheme(data_width, "hamming")
        hsiao_metrics = analyze_scheme(data_width, "hsiao")

        # Hsiao should have better balanced fan-in (lower std deviation)
        # or at least not significantly worse
        # Note: This is a design goal of Hsiao, allowing some tolerance
        assert hsiao_metrics.fanin_std <= hamming_metrics.fanin_std + 1.0

        # Hsiao should generally have comparable or better max XOR depth
        # Allow some tolerance as this depends on specific k values
        assert hsiao_metrics.max_xor_depth <= hamming_metrics.max_xor_depth + 1

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    def test_both_schemes_correct_single_errors(self, data_width: int):
        """Verify both schemes can correct single-bit errors."""
        H_hamming, r_hamming = generate_h_matrix_secdec(data_width, "hamming")
        H_hsiao, r_hsiao = generate_h_matrix_secdec(data_width, "hsiao")

        original_data = _generate_random_data(data_width)
        codeword_hamming = encode(original_data, H_hamming, r_hamming, data_width)
        codeword_hsiao = encode(original_data, H_hsiao, r_hsiao, data_width)

        codeword_length = len(codeword_hamming)

        # Test a few random error positions
        for _ in range(5):
            error_pos = random.randint(0, codeword_length - 1)

            corrupted_hamming = inject_error(codeword_hamming, [error_pos])
            corrupted_hsiao = inject_error(codeword_hsiao, [error_pos])

            result_hamming = decode(corrupted_hamming, H_hamming, r_hamming, data_width)
            result_hsiao = decode(corrupted_hsiao, H_hsiao, r_hsiao, data_width)

            assert result_hamming.corrected_data == original_data
            assert result_hsiao.corrected_data == original_data
            assert result_hamming.error_type == "single_corrected"
            assert result_hsiao.error_type == "single_corrected"

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    def test_both_schemes_detect_double_errors(self, data_width: int):
        """Verify both schemes can detect double-bit errors."""
        H_hamming, r_hamming = generate_h_matrix_secdec(data_width, "hamming")
        H_hsiao, r_hsiao = generate_h_matrix_secdec(data_width, "hsiao")

        original_data = _generate_random_data(data_width)
        codeword_hamming = encode(original_data, H_hamming, r_hamming, data_width)
        codeword_hsiao = encode(original_data, H_hsiao, r_hsiao, data_width)

        codeword_length = len(codeword_hamming)

        # Test a few random double-bit error patterns
        for _ in range(5):
            error_positions = random.sample(range(codeword_length), 2)

            corrupted_hamming = inject_error(codeword_hamming, error_positions)
            corrupted_hsiao = inject_error(codeword_hsiao, error_positions)

            result_hamming = decode(corrupted_hamming, H_hamming, r_hamming, data_width)
            result_hsiao = decode(corrupted_hsiao, H_hsiao, r_hsiao, data_width)

            assert result_hamming.error_type == "double_detected"
            assert result_hsiao.error_type == "double_detected"


# =============================================================================
# Additional Integration Tests
# =============================================================================


class TestEndToEndWorkflows:
    """Additional end-to-end workflow tests."""

    def test_full_pipeline_all_components(self):
        """Test a complete pipeline using all components together."""
        data_width = 64
        scheme: Scheme = "hsiao"

        # 1. Generate H-matrix
        H, r = generate_h_matrix_secdec(data_width, scheme)

        # 2. Analyze metrics
        metrics = analyze_scheme(data_width, scheme)
        assert metrics.k == data_width
        assert metrics.r == r

        # 3. Calculate XOR depth
        xor_result = calculate_xor_depth(H, r)
        assert xor_result.max_depth == metrics.max_xor_depth

        # 4. Generate SystemVerilog
        sv_code = parity_equations_sv(data_width, scheme)
        assert f"k={data_width}" in sv_code

        # 5. Encode data
        test_data = 0xDEADBEEFCAFEBABE & ((1 << data_width) - 1)
        codeword = encode(test_data, H, r, data_width)

        # 6. Decode without error
        result = decode(codeword, H, r, data_width)
        assert result.corrected_data == test_data

        # 7. Inject and correct error
        corrupted = inject_error(codeword, [r + 5])  # Error in data bit 5
        result = decode(corrupted, H, r, data_width)
        assert result.corrected_data == test_data
        assert result.error_type == "single_corrected"

    def test_comparison_workflow(self):
        """Test complete comparison workflow across multiple widths."""
        k_values = [32, 64, 128]

        # Generate comparison table
        table = generate_comparison_table(k_values)
        assert len(table) > 0

        # Verify comparison data for each width
        for k in k_values:
            results = compare_schemes(k)

            # Verify both schemes work
            for scheme in ["hamming", "hsiao"]:
                H, r = generate_h_matrix_secdec(k, scheme)
                data = _generate_random_data(k)
                codeword = encode(data, H, r, k)
                decoded = decode(codeword, H, r, k)
                assert decoded.corrected_data == data

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_stress_test_multiple_operations(self, scheme: Scheme):
        """Stress test with many sequential operations."""
        data_width = 128
        H, r = generate_h_matrix_secdec(data_width, scheme)

        # Perform many encode/decode cycles
        for i in range(100):
            data = random.randint(0, (1 << data_width) - 1)
            codeword = encode(data, H, r, data_width)
            result = decode(codeword, H, r, data_width)
            assert result.corrected_data == data

            # Occasionally inject and correct an error
            if i % 10 == 0:
                error_pos = random.randint(0, len(codeword) - 1)
                corrupted = inject_error(codeword, [error_pos])
                result = decode(corrupted, H, r, data_width)
                assert result.corrected_data == data
