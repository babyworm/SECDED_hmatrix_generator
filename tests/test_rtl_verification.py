"""
Tests for RTL verification module.

Tests cover:
- Test vector generation
- Coverage tracking
- Verilator integration (requires verilator installed)
"""

import pytest
import os
import tempfile
import shutil
from typing import List

from src.rtl_verification import (
    RTLVerifier,
    TestVector,
    CoveragePoint,
    CoverageReport,
    VerificationResult,
    verify_rtl,
)


class TestCoverageReport:
    """Tests for CoverageReport class."""

    def test_add_point(self):
        """Test adding coverage points."""
        report = CoverageReport()
        report.add_point("test_point")

        assert report.total_points == 1
        assert report.covered_points == 0
        assert "test_point" in report.points

    def test_add_duplicate_point(self):
        """Test that duplicate points are not added."""
        report = CoverageReport()
        report.add_point("test_point")
        report.add_point("test_point")

        assert report.total_points == 1

    def test_hit_point(self):
        """Test hitting coverage points."""
        report = CoverageReport()
        report.add_point("test_point")
        report.hit_point("test_point")

        assert report.covered_points == 1
        assert report.points["test_point"].hit
        assert report.points["test_point"].count == 1
        assert report.coverage_percent == 100.0

    def test_hit_point_multiple_times(self):
        """Test hitting the same point multiple times."""
        report = CoverageReport()
        report.add_point("test_point")
        report.hit_point("test_point")
        report.hit_point("test_point")
        report.hit_point("test_point")

        assert report.covered_points == 1  # Only counted once
        assert report.points["test_point"].count == 3  # But tracked

    def test_coverage_percent(self):
        """Test coverage percentage calculation."""
        report = CoverageReport()
        report.add_point("point1")
        report.add_point("point2")
        report.add_point("point3")
        report.add_point("point4")

        report.hit_point("point1")
        report.hit_point("point2")

        assert report.coverage_percent == 50.0


class TestTestVector:
    """Tests for TestVector dataclass."""

    def test_create_vector(self):
        """Test creating a test vector."""
        vector = TestVector(
            data=0x1234,
            error_positions=[1, 5],
            expected_corrected_data=0x1234,
            expected_error_type=2,
            description="test vector"
        )

        assert vector.data == 0x1234
        assert vector.error_positions == [1, 5]
        assert vector.expected_error_type == 2

    def test_vector_no_error(self):
        """Test vector with no error."""
        vector = TestVector(
            data=0xABCD,
            error_positions=[],
            expected_corrected_data=0xABCD,
            expected_error_type=0,
            description="no error"
        )

        assert len(vector.error_positions) == 0
        assert vector.expected_error_type == 0


class TestRTLVerifier:
    """Tests for RTLVerifier class."""

    def test_init_hsiao(self):
        """Test initializing verifier with Hsiao scheme."""
        verifier = RTLVerifier(k=32, scheme="hsiao")

        assert verifier.k == 32
        assert verifier.scheme == "hsiao"
        assert verifier.r > 0
        assert verifier.n == 32 + verifier.r + 1

    def test_init_hamming(self):
        """Test initializing verifier with Hamming scheme."""
        verifier = RTLVerifier(k=64, scheme="hamming")

        assert verifier.k == 64
        assert verifier.scheme == "hamming"
        assert verifier.r > 0

    def test_coverage_points_initialized(self):
        """Test that coverage points are initialized."""
        verifier = RTLVerifier(k=32, scheme="hsiao")

        # Data pattern coverage
        assert "data_all_zeros" in verifier.coverage.points
        assert "data_all_ones" in verifier.coverage.points
        assert "data_random" in verifier.coverage.points

        # Error type coverage
        assert "error_none" in verifier.coverage.points
        assert "error_single_data_bit" in verifier.coverage.points
        assert "error_double_bit" in verifier.coverage.points

        # Bit position coverage
        assert "bit_pos_0" in verifier.coverage.points

    def test_generate_test_vectors(self):
        """Test generating test vectors."""
        verifier = RTLVerifier(k=32, scheme="hsiao")
        vectors = verifier.generate_test_vectors(num_random=10)

        # Should have deterministic vectors plus random ones
        assert len(vectors) > 10

        # Check coverage was updated
        assert verifier.coverage.covered_points > 0

    def test_generate_test_vectors_types(self):
        """Test that generated vectors have correct types."""
        verifier = RTLVerifier(k=32, scheme="hsiao")
        vectors = verifier.generate_test_vectors(num_random=5)

        for v in vectors:
            assert isinstance(v, TestVector)
            assert isinstance(v.data, int)
            assert isinstance(v.error_positions, list)
            assert isinstance(v.expected_error_type, int)
            assert 0 <= v.expected_error_type <= 3

    def test_generate_exhaustive_errors(self):
        """Test exhaustive error position testing."""
        verifier = RTLVerifier(k=32, scheme="hsiao")
        vectors_normal = verifier.generate_test_vectors(num_random=0, include_exhaustive_errors=False)

        verifier2 = RTLVerifier(k=32, scheme="hsiao")
        vectors_exhaustive = verifier2.generate_test_vectors(num_random=0, include_exhaustive_errors=True)

        # Exhaustive should have more vectors
        assert len(vectors_exhaustive) > len(vectors_normal)

    def test_create_vector_no_error(self):
        """Test creating vector with no error."""
        verifier = RTLVerifier(k=32, scheme="hsiao")
        vector = verifier._create_vector(0x12345678, [], "test")

        assert vector.data == 0x12345678
        assert vector.expected_error_type == 0  # no error
        assert vector.expected_corrected_data == 0x12345678

    def test_create_vector_single_error(self):
        """Test creating vector with single bit error."""
        verifier = RTLVerifier(k=32, scheme="hsiao")
        # Error in data region
        vector = verifier._create_vector(0x12345678, [verifier.r + 5], "test")

        assert vector.expected_error_type == 1  # corrected
        assert vector.expected_corrected_data == 0x12345678

    def test_create_vector_double_error(self):
        """Test creating vector with double bit error."""
        verifier = RTLVerifier(k=32, scheme="hsiao")
        vector = verifier._create_vector(0x12345678, [0, 1], "test")

        assert vector.expected_error_type == 2  # detected (not correctable)

    def test_int_to_word_array(self):
        """Test integer to word array conversion."""
        verifier = RTLVerifier(k=32, scheme="hsiao")

        # Test simple case
        words = verifier._int_to_word_array(0x12345678, 1)
        assert words == [0x12345678]

        # Test multi-word case
        words = verifier._int_to_word_array(0x123456789ABCDEF0, 3)
        assert words[0] == 0x9ABCDEF0  # LSW
        assert words[1] == 0x12345678
        assert words[2] == 0x00000000

    def test_int_to_word_array_zero(self):
        """Test conversion of zero."""
        verifier = RTLVerifier(k=32, scheme="hsiao")
        words = verifier._int_to_word_array(0, 2)
        assert words == [0, 0]


class TestVerifyRTL:
    """Integration tests for verify_rtl function (requires Verilator)."""

    @pytest.fixture
    def check_verilator(self):
        """Check if Verilator is available."""
        result = shutil.which("verilator")
        if result is None:
            pytest.skip("Verilator not installed")

    def test_verify_rtl_32bit_hsiao(self, check_verilator):
        """Test RTL verification for 32-bit Hsiao scheme."""
        with tempfile.TemporaryDirectory() as work_dir:
            result = verify_rtl(
                k=32,
                scheme="hsiao",
                num_random_tests=5,
                work_dir=work_dir,
                keep_files=False
            )

            assert isinstance(result, VerificationResult)
            assert result.total_tests > 0
            assert result.passed
            assert result.failed_tests == 0

    def test_verify_rtl_32bit_hamming(self, check_verilator):
        """Test RTL verification for 32-bit Hamming scheme."""
        with tempfile.TemporaryDirectory() as work_dir:
            result = verify_rtl(
                k=32,
                scheme="hamming",
                num_random_tests=5,
                work_dir=work_dir,
                keep_files=False
            )

            assert result.passed
            assert result.passed_tests == result.total_tests

    def test_verify_rtl_64bit_hsiao(self, check_verilator):
        """Test RTL verification for 64-bit Hsiao scheme (mixed types)."""
        with tempfile.TemporaryDirectory() as work_dir:
            result = verify_rtl(
                k=64,
                scheme="hsiao",
                num_random_tests=5,
                work_dir=work_dir,
                keep_files=False
            )

            assert result.passed
            assert len(result.errors) == 0

    def test_verify_rtl_coverage(self, check_verilator):
        """Test that coverage is tracked during verification."""
        with tempfile.TemporaryDirectory() as work_dir:
            result = verify_rtl(
                k=32,
                scheme="hsiao",
                num_random_tests=10,
                work_dir=work_dir,
                keep_files=False
            )

            assert result.coverage.total_points > 0
            assert result.coverage.covered_points > 0
            assert result.coverage.coverage_percent > 0

    def test_verify_rtl_keep_files(self, check_verilator):
        """Test that files are kept when requested."""
        with tempfile.TemporaryDirectory() as work_dir:
            result = verify_rtl(
                k=32,
                scheme="hsiao",
                num_random_tests=2,
                work_dir=work_dir,
                keep_files=True
            )

            # Check that RTL files exist
            assert os.path.exists(os.path.join(work_dir, "secded_encoder_32.sv"))
            assert os.path.exists(os.path.join(work_dir, "secded_decoder_32.sv"))
            assert os.path.exists(os.path.join(work_dir, "tb_main.cpp"))


class TestVerifierTestbenchGeneration:
    """Tests for testbench generation methods."""

    def test_simple_testbench_generated(self):
        """Test that simple testbench is generated for k <= 64, n <= 64."""
        verifier = RTLVerifier(k=32, scheme="hsiao")
        vectors = [verifier._create_vector(0x1234, [], "test")]

        with tempfile.TemporaryDirectory() as work_dir:
            tb_path = verifier._generate_wrapper_testbench(vectors, work_dir)

            assert os.path.exists(tb_path)
            with open(tb_path, 'r') as f:
                content = f.read()
                assert "uint64_t data" in content
                assert "uint64_t error_mask" in content
                # Should not use VlWide for 32-bit
                assert "VlWide" not in content

    def test_mixed_testbench_generated(self):
        """Test that mixed testbench is generated for k <= 64, n > 64."""
        verifier = RTLVerifier(k=64, scheme="hsiao")  # n = 72 > 64
        vectors = [verifier._create_vector(0x1234, [], "test")]

        with tempfile.TemporaryDirectory() as work_dir:
            tb_path = verifier._generate_wrapper_testbench(vectors, work_dir)

            assert os.path.exists(tb_path)
            with open(tb_path, 'r') as f:
                content = f.read()
                # Data is uint64_t but codeword uses VlWide
                assert "uint64_t data" in content
                assert "VlWide" in content
                assert "set_wide" in content

    def test_wide_testbench_generated(self):
        """Test that wide testbench is generated for k > 64."""
        verifier = RTLVerifier(k=128, scheme="hsiao")
        vectors = [verifier._create_vector(0x1234, [], "test")]

        with tempfile.TemporaryDirectory() as work_dir:
            tb_path = verifier._generate_wrapper_testbench(vectors, work_dir)

            assert os.path.exists(tb_path)
            with open(tb_path, 'r') as f:
                content = f.read()
                assert "VlWide" in content
                assert "set_wide" in content
                assert "eq_wide" in content


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_create_pass_result(self):
        """Test creating a passing result."""
        coverage = CoverageReport()
        result = VerificationResult(
            passed=True,
            total_tests=10,
            passed_tests=10,
            failed_tests=0,
            coverage=coverage
        )

        assert result.passed
        assert result.total_tests == 10
        assert len(result.errors) == 0

    def test_create_fail_result(self):
        """Test creating a failing result."""
        coverage = CoverageReport()
        result = VerificationResult(
            passed=False,
            total_tests=10,
            passed_tests=7,
            failed_tests=3,
            coverage=coverage,
            errors=["Test 1 failed", "Test 2 failed"]
        )

        assert not result.passed
        assert result.failed_tests == 3
        assert len(result.errors) == 2
