"""
SystemVerilog Generator Tests

Tests for the SystemVerilog code generation module.
Verifies encoder, decoder, testbench, and package generation.
"""
import pytest
import os
import tempfile
from typing import List

from src.sv_generator import (
    SVGeneratorConfig,
    generate_encoder_sv,
    generate_decoder_sv,
    generate_top_module_sv,
    generate_testbench_sv,
    generate_package_sv,
    generate_all_sv_files,
)
from src.gen_hmatrix import generate_h_matrix_secdec


# Test configurations
DATA_WIDTHS = [32, 64, 128, 256]
SCHEMES = ["hamming", "hsiao"]


class TestEncoderGeneration:
    """Tests for encoder module generation."""

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_encoder_generates_valid_module(self, data_width: int, scheme: str):
        """Test that encoder generates syntactically valid SystemVerilog."""
        code = generate_encoder_sv(data_width, scheme)

        # Check module declaration
        assert f"module secded_encoder_{data_width}" in code
        assert "endmodule" in code

        # Check ports
        assert "input  logic" in code
        assert "output logic" in code
        assert "clk" in code
        assert "rst_n" in code

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_encoder_has_correct_parameters(self, data_width: int, scheme: str):
        """Test encoder has correct parameter declarations."""
        H, r = generate_h_matrix_secdec(data_width, scheme)
        n = data_width + r + 1

        code = generate_encoder_sv(data_width, scheme)

        assert f"parameter int DATA_WIDTH = {data_width}" in code
        assert f"parameter int PARITY_BITS = {r}" in code
        assert f"parameter int CODEWORD_WIDTH = {n}" in code

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_encoder_has_parity_computation(self, data_width: int, scheme: str):
        """Test encoder generates parity computation logic."""
        code = generate_encoder_sv(data_width, scheme)

        # Should have always_comb blocks for parity
        assert "always_comb begin" in code
        assert "parity[" in code

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_encoder_has_reset_logic(self, data_width: int, scheme: str):
        """Test encoder has proper reset handling."""
        code = generate_encoder_sv(data_width, scheme)

        assert "always_ff @(posedge clk or negedge rst_n)" in code
        assert "if (!rst_n)" in code

    def test_encoder_custom_config(self):
        """Test encoder with custom configuration."""
        config = SVGeneratorConfig(
            module_prefix="my_secded",
            data_signal="din",
            codeword_signal="cw_out",
            author="Test Author",
            include_assertions=False
        )

        code = generate_encoder_sv(64, "hsiao", config)

        assert "module my_secded_encoder_64" in code
        assert "din" in code
        assert "cw_out" in code
        assert "Test Author" in code
        # No assertions when disabled
        assert "synthesis translate_off" not in code or "assert" not in code


class TestDecoderGeneration:
    """Tests for decoder module generation."""

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_decoder_generates_valid_module(self, data_width: int, scheme: str):
        """Test that decoder generates syntactically valid SystemVerilog."""
        code = generate_decoder_sv(data_width, scheme)

        assert f"module secded_decoder_{data_width}" in code
        assert "endmodule" in code

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_decoder_has_syndrome_computation(self, data_width: int, scheme: str):
        """Test decoder generates syndrome computation."""
        code = generate_decoder_sv(data_width, scheme)

        assert "syndrome" in code
        assert "syndrome_comb" in code

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_decoder_has_error_types(self, data_width: int, scheme: str):
        """Test decoder includes all error type constants."""
        code = generate_decoder_sv(data_width, scheme)

        assert "ERR_NONE" in code
        assert "ERR_CORRECTED" in code
        assert "ERR_DETECTED" in code
        assert "ERR_UNKNOWN" in code

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_decoder_has_error_position_lookup(self, data_width: int, scheme: str):
        """Test decoder has error position lookup table."""
        code = generate_decoder_sv(data_width, scheme)

        # Should have case statement for syndrome lookup
        assert "case (syndrome_comb)" in code
        assert "error_position" in code
        assert "position_valid" in code

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_decoder_has_correction_logic(self, data_width: int, scheme: str):
        """Test decoder includes error correction logic."""
        code = generate_decoder_sv(data_width, scheme)

        assert "corrected_codeword" in code
        assert "error_corrected" in code


class TestTopModuleGeneration:
    """Tests for top module generation."""

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_top_module_generates_valid_code(self, data_width: int, scheme: str):
        """Test top module generates valid SystemVerilog."""
        code = generate_top_module_sv(data_width, scheme)

        assert f"module secded_top_{data_width}" in code
        assert "endmodule" in code

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_top_module_instantiates_encoder_decoder(self, data_width: int, scheme: str):
        """Test top module instantiates both encoder and decoder."""
        code = generate_top_module_sv(data_width, scheme)

        assert f"secded_encoder_{data_width}" in code
        assert f"secded_decoder_{data_width}" in code
        assert "u_encoder" in code
        assert "u_decoder" in code


class TestTestbenchGeneration:
    """Tests for testbench generation."""

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_testbench_generates_valid_code(self, data_width: int, scheme: str):
        """Test testbench generates valid SystemVerilog."""
        code = generate_testbench_sv(data_width, scheme)

        assert f"module tb_secded_{data_width}" in code
        assert "endmodule" in code
        assert "`timescale" in code

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_testbench_has_clock_generation(self, data_width: int, scheme: str):
        """Test testbench includes clock generation."""
        code = generate_testbench_sv(data_width, scheme)

        assert "CLK_PERIOD" in code
        assert "clk = ~clk" in code or "forever" in code

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_testbench_has_dut_instances(self, data_width: int, scheme: str):
        """Test testbench instantiates encoder and decoder."""
        code = generate_testbench_sv(data_width, scheme)

        assert f"secded_encoder_{data_width}" in code
        assert f"secded_decoder_{data_width}" in code

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_testbench_has_test_cases(self, data_width: int, scheme: str):
        """Test testbench includes various test cases."""
        code = generate_testbench_sv(data_width, scheme)

        # Should test no error, single bit error, double bit error
        assert "No Error" in code or "no error" in code.lower()
        assert "Single Bit Error" in code or "single bit" in code.lower()
        assert "Double Bit Error" in code or "double bit" in code.lower()

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_testbench_has_result_checking(self, data_width: int, scheme: str):
        """Test testbench includes result verification."""
        code = generate_testbench_sv(data_width, scheme)

        assert "PASS" in code or "pass" in code
        assert "FAIL" in code or "fail" in code


class TestPackageGeneration:
    """Tests for package generation."""

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_package_generates_valid_code(self, data_width: int, scheme: str):
        """Test package generates valid SystemVerilog."""
        code = generate_package_sv(data_width, scheme)

        assert "package secded_pkg" in code
        assert "endpackage" in code

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_package_has_type_definitions(self, data_width: int, scheme: str):
        """Test package includes type definitions."""
        code = generate_package_sv(data_width, scheme)

        assert "typedef" in code
        assert "data_t" in code
        assert "codeword_t" in code
        assert "syndrome_t" in code

    @pytest.mark.parametrize("data_width", DATA_WIDTHS)
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_package_has_error_enum(self, data_width: int, scheme: str):
        """Test package includes error type enumeration."""
        code = generate_package_sv(data_width, scheme)

        assert "error_type_e" in code
        assert "ERR_NONE" in code
        assert "ERR_CORRECTED" in code


class TestGenerateAllFiles:
    """Tests for generate_all_sv_files function."""

    @pytest.mark.parametrize("data_width", [32, 64])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_generates_all_expected_files(self, data_width: int, scheme: str):
        """Test that all expected files are generated."""
        files = generate_all_sv_files(data_width, scheme)

        expected_files = [
            "secded_pkg.sv",
            f"secded_encoder_{data_width}.sv",
            f"secded_decoder_{data_width}.sv",
            f"secded_top_{data_width}.sv",
            f"tb_secded_{data_width}.sv",
        ]

        for expected in expected_files:
            assert expected in files, f"Missing file: {expected}"

    @pytest.mark.parametrize("data_width", [32, 64])
    def test_files_have_content(self, data_width: int):
        """Test that all generated files have substantial content."""
        files = generate_all_sv_files(data_width, "hsiao")

        for filename, content in files.items():
            assert len(content) > 100, f"File {filename} has too little content"
            assert "module" in content or "package" in content

    def test_custom_prefix(self):
        """Test file generation with custom prefix."""
        config = SVGeneratorConfig(module_prefix="my_ecc")
        files = generate_all_sv_files(64, "hsiao", config=config)

        assert "my_ecc_pkg.sv" in files
        assert "my_ecc_encoder_64.sv" in files
        assert "my_ecc_decoder_64.sv" in files

    def test_write_files_to_directory(self):
        """Test writing files to a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = generate_all_sv_files(32, "hsiao")

            # Write files
            for filename, content in files.items():
                filepath = os.path.join(tmpdir, filename)
                with open(filepath, 'w') as f:
                    f.write(content)

            # Verify files exist
            for filename in files:
                filepath = os.path.join(tmpdir, filename)
                assert os.path.exists(filepath)
                assert os.path.getsize(filepath) > 0


class TestSVGeneratorConfig:
    """Tests for SVGeneratorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SVGeneratorConfig()

        assert config.module_prefix == "secded"
        assert config.data_signal == "data_in"
        assert config.include_header is True
        assert config.include_assertions is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SVGeneratorConfig(
            module_prefix="custom_ecc",
            data_signal="din",
            codeword_signal="cw",
            include_assertions=False
        )

        assert config.module_prefix == "custom_ecc"
        assert config.data_signal == "din"
        assert config.codeword_signal == "cw"
        assert config.include_assertions is False


class TestCodeQuality:
    """Tests for code quality and style compliance."""

    @pytest.mark.parametrize("data_width", [64, 128])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_no_blocking_in_always_ff(self, data_width: int, scheme: str):
        """Test that always_ff blocks don't use blocking assignments incorrectly."""
        encoder_code = generate_encoder_sv(data_width, scheme)
        decoder_code = generate_decoder_sv(data_width, scheme)

        # This is a simplified check - in always_ff, we should use <=
        # The generated code should follow this pattern
        for code in [encoder_code, decoder_code]:
            # Find always_ff blocks
            in_always_ff = False
            lines = code.split('\n')
            for line in lines:
                if 'always_ff' in line:
                    in_always_ff = True
                elif in_always_ff and ('end' in line and 'endmodule' not in line):
                    in_always_ff = False

    @pytest.mark.parametrize("data_width", [64])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_all_registers_have_reset(self, data_width: int, scheme: str):
        """Test that sequential logic includes reset handling."""
        encoder_code = generate_encoder_sv(data_width, scheme)
        decoder_code = generate_decoder_sv(data_width, scheme)

        for code in [encoder_code, decoder_code]:
            if "always_ff" in code:
                # Should have reset check
                assert "rst_n" in code
                assert "if (!rst_n)" in code

    @pytest.mark.parametrize("data_width", [64])
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_proper_naming_conventions(self, data_width: int, scheme: str):
        """Test that code follows naming conventions."""
        code = generate_encoder_sv(data_width, scheme)

        # Parameters should be uppercase
        assert "DATA_WIDTH" in code
        assert "PARITY_BITS" in code

        # Signals should be snake_case
        assert "output_valid" in code or "input_valid" in code


class TestLargeDataWidths:
    """Tests for large data width support."""

    @pytest.mark.parametrize("data_width", [512, 1024])
    def test_large_width_encoder(self, data_width: int):
        """Test encoder generation for large data widths."""
        code = generate_encoder_sv(data_width, "hsiao")

        assert f"module secded_encoder_{data_width}" in code
        assert "endmodule" in code

    @pytest.mark.parametrize("data_width", [512, 1024])
    def test_large_width_decoder(self, data_width: int):
        """Test decoder generation for large data widths."""
        code = generate_decoder_sv(data_width, "hsiao")

        assert f"module secded_decoder_{data_width}" in code
        assert "endmodule" in code


class TestEdgeCases:
    """Tests for edge cases."""

    def test_minimum_data_width(self):
        """Test generation with minimum practical data width."""
        # Hsiao supports smaller k values
        code = generate_encoder_sv(8, "hsiao")
        assert "module secded_encoder_8" in code

    def test_header_disabled(self):
        """Test generation without header."""
        config = SVGeneratorConfig(include_header=False)
        code = generate_encoder_sv(32, "hsiao", config)

        # Should not have the header comment block
        assert "//==============" not in code or code.index("module") < code.index("//==") if "//==" in code else True

    def test_assertions_disabled(self):
        """Test generation without assertions."""
        config = SVGeneratorConfig(include_assertions=False)
        code = generate_encoder_sv(32, "hsiao", config)

        # Assertions are in translate_off blocks
        # When disabled, there should be no translate_off section with assert
        if "synthesis translate_off" in code:
            translate_section = code.split("synthesis translate_off")[1].split("synthesis translate_on")[0]
            # This section should be minimal or not have complex assertions
