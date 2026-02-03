"""
CLI interface tests for SECDED H-matrix generator.
Tests argument parsing, command handlers, and output formats.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.cli import (
    __version__,
    cmd_analyze,
    cmd_compare,
    cmd_generate,
    cmd_verilog,
    create_parser,
)


# =============================================================================
# Argument Parsing Tests
# =============================================================================


class TestArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "python -m src.cli"

    def test_generate_required_k_argument(self):
        """Test that generate command requires -k argument."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["generate"])
        assert exc_info.value.code != 0

    def test_verilog_required_k_argument(self):
        """Test that verilog command requires -k argument."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["verilog"])
        assert exc_info.value.code != 0

    def test_analyze_required_k_argument(self):
        """Test that analyze command requires -k argument."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["analyze"])
        assert exc_info.value.code != 0

    def test_compare_required_k_argument(self):
        """Test that compare command requires -k argument."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["compare"])
        assert exc_info.value.code != 0

    def test_generate_default_scheme(self):
        """Test that generate command defaults to hsiao scheme."""
        parser = create_parser()
        args = parser.parse_args(["generate", "-k", "64"])
        assert args.scheme == "hsiao"

    def test_generate_default_format(self):
        """Test that generate command defaults to matrix format."""
        parser = create_parser()
        args = parser.parse_args(["generate", "-k", "64"])
        assert args.format == "matrix"

    def test_verilog_default_scheme(self):
        """Test that verilog command defaults to hsiao scheme."""
        parser = create_parser()
        args = parser.parse_args(["verilog", "-k", "64"])
        assert args.scheme == "hsiao"

    def test_analyze_default_scheme(self):
        """Test that analyze command defaults to hsiao scheme."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "-k", "64"])
        assert args.scheme == "hsiao"

    def test_invalid_scheme_name(self):
        """Test that invalid scheme name is rejected."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["generate", "-k", "64", "--scheme", "invalid"])
        assert exc_info.value.code != 0

    def test_invalid_format_name(self):
        """Test that invalid format name is rejected."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["generate", "-k", "64", "--format", "invalid"])
        assert exc_info.value.code != 0

    def test_version_flag(self):
        """Test that --version flag works."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_help_flag_main(self):
        """Test that --help flag works on main parser."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_help_flag_generate(self):
        """Test that --help flag works on generate subcommand."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["generate", "--help"])
        assert exc_info.value.code == 0

    def test_help_flag_verilog(self):
        """Test that --help flag works on verilog subcommand."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["verilog", "--help"])
        assert exc_info.value.code == 0

    def test_help_flag_analyze(self):
        """Test that --help flag works on analyze subcommand."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["analyze", "--help"])
        assert exc_info.value.code == 0

    def test_help_flag_compare(self):
        """Test that --help flag works on compare subcommand."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["compare", "--help"])
        assert exc_info.value.code == 0

    def test_generate_long_form_data_width(self):
        """Test that --data-width works as alternative to -k."""
        parser = create_parser()
        args = parser.parse_args(["generate", "--data-width", "128"])
        assert args.data_width == 128

    def test_verilog_output_flag(self):
        """Test that -o flag is parsed for verilog command."""
        parser = create_parser()
        args = parser.parse_args(["verilog", "-k", "64", "-o", "output.sv"])
        assert args.output == "output.sv"

    def test_verilog_long_output_flag(self):
        """Test that --output flag is parsed for verilog command."""
        parser = create_parser()
        args = parser.parse_args(["verilog", "-k", "64", "--output", "output.sv"])
        assert args.output == "output.sv"

    def test_scheme_choices(self):
        """Test that both hamming and hsiao schemes are accepted."""
        parser = create_parser()

        args_hamming = parser.parse_args(["generate", "-k", "64", "--scheme", "hamming"])
        assert args_hamming.scheme == "hamming"

        args_hsiao = parser.parse_args(["generate", "-k", "64", "--scheme", "hsiao"])
        assert args_hsiao.scheme == "hsiao"

    def test_format_choices(self):
        """Test that both matrix and json formats are accepted."""
        parser = create_parser()

        args_matrix = parser.parse_args(["generate", "-k", "64", "--format", "matrix"])
        assert args_matrix.format == "matrix"

        args_json = parser.parse_args(["generate", "-k", "64", "--format", "json"])
        assert args_json.format == "json"


# =============================================================================
# Generate Command Tests
# =============================================================================


class TestGenerateCommand:
    """Tests for the generate command."""

    def test_generate_matrix_format_hsiao(self, capsys):
        """Test generate command with matrix format and hsiao scheme."""
        parser = create_parser()
        args = parser.parse_args(["generate", "-k", "32", "--scheme", "hsiao", "--format", "matrix"])

        result = cmd_generate(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "H-matrix" in captured.out
        assert "Data bits (k): 32" in captured.out
        assert "SEC parity bits (r):" in captured.out

    def test_generate_matrix_format_hamming(self, capsys):
        """Test generate command with matrix format and hamming scheme."""
        parser = create_parser()
        args = parser.parse_args(["generate", "-k", "32", "--scheme", "hamming", "--format", "matrix"])

        result = cmd_generate(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "H-matrix" in captured.out
        assert "Data bits (k): 32" in captured.out

    def test_generate_json_format_hsiao(self, capsys):
        """Test generate command with json format and hsiao scheme."""
        parser = create_parser()
        args = parser.parse_args(["generate", "-k", "32", "--scheme", "hsiao", "--format", "json"])

        result = cmd_generate(args)

        assert result == 0
        captured = capsys.readouterr()

        data = json.loads(captured.out)
        assert data["scheme"] == "hsiao"
        assert data["k"] == 32
        assert "r" in data
        assert "total_bits" in data
        assert "matrix" in data
        assert isinstance(data["matrix"], list)

    def test_generate_json_format_hamming(self, capsys):
        """Test generate command with json format and hamming scheme."""
        parser = create_parser()
        args = parser.parse_args(["generate", "-k", "32", "--scheme", "hamming", "--format", "json"])

        result = cmd_generate(args)

        assert result == 0
        captured = capsys.readouterr()

        data = json.loads(captured.out)
        assert data["scheme"] == "hamming"
        assert data["k"] == 32

    def test_generate_json_structure(self, capsys):
        """Test that JSON output has expected structure."""
        parser = create_parser()
        args = parser.parse_args(["generate", "-k", "64", "--format", "json"])

        result = cmd_generate(args)

        assert result == 0
        captured = capsys.readouterr()

        data = json.loads(captured.out)
        assert "rows" in data
        assert "cols" in data
        assert data["rows"] == len(data["matrix"])
        assert data["cols"] == len(data["matrix"][0])
        assert data["total_bits"] == data["k"] + data["r"] + 1

    def test_generate_matrix_dimensions_in_output(self, capsys):
        """Test that matrix output contains dimension information."""
        parser = create_parser()
        args = parser.parse_args(["generate", "-k", "64", "--format", "matrix"])

        result = cmd_generate(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Total bits:" in captured.out

    @pytest.mark.parametrize("k", [32, 64, 128])
    def test_generate_various_data_widths(self, k, capsys):
        """Test generate command with various data widths."""
        parser = create_parser()
        args = parser.parse_args(["generate", "-k", str(k), "--format", "json"])

        result = cmd_generate(args)

        assert result == 0
        captured = capsys.readouterr()

        data = json.loads(captured.out)
        assert data["k"] == k


# =============================================================================
# Verilog Command Tests
# =============================================================================


class TestVerilogCommand:
    """Tests for the verilog command."""

    def test_verilog_stdout_output(self, capsys):
        """Test verilog command outputs to stdout by default."""
        parser = create_parser()
        args = parser.parse_args(["verilog", "-k", "32", "--scheme", "hsiao"])

        result = cmd_verilog(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "assign" in captured.out

    def test_verilog_contains_assign_statements(self, capsys):
        """Test that verilog output contains assign statements."""
        parser = create_parser()
        args = parser.parse_args(["verilog", "-k", "32"])

        result = cmd_verilog(args)

        assert result == 0
        captured = capsys.readouterr()
        assert captured.out.count("assign") >= 1

    def test_verilog_file_output(self, capsys):
        """Test verilog command with file output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sv", delete=False) as f:
            output_path = f.name

        try:
            parser = create_parser()
            args = parser.parse_args(["verilog", "-k", "32", "-o", output_path])

            result = cmd_verilog(args)

            assert result == 0

            captured = capsys.readouterr()
            assert output_path in captured.err

            with open(output_path, "r") as f:
                content = f.read()
            assert "assign" in content
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_verilog_hamming_scheme(self, capsys):
        """Test verilog command with hamming scheme."""
        parser = create_parser()
        args = parser.parse_args(["verilog", "-k", "32", "--scheme", "hamming"])

        result = cmd_verilog(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "assign" in captured.out

    def test_verilog_hsiao_scheme(self, capsys):
        """Test verilog command with hsiao scheme."""
        parser = create_parser()
        args = parser.parse_args(["verilog", "-k", "32", "--scheme", "hsiao"])

        result = cmd_verilog(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "assign" in captured.out

    def test_verilog_xor_operators(self, capsys):
        """Test that verilog output contains XOR operators."""
        parser = create_parser()
        args = parser.parse_args(["verilog", "-k", "32"])

        result = cmd_verilog(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "^" in captured.out

    @pytest.mark.parametrize("k", [32, 64, 128])
    def test_verilog_various_data_widths(self, k, capsys):
        """Test verilog command with various data widths."""
        parser = create_parser()
        args = parser.parse_args(["verilog", "-k", str(k)])

        result = cmd_verilog(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "assign" in captured.out


# =============================================================================
# Analyze Command Tests
# =============================================================================


class TestAnalyzeCommand:
    """Tests for the analyze command."""

    def test_analyze_output_contains_metrics(self, capsys):
        """Test that analyze output contains expected metrics."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "-k", "64", "--scheme", "hsiao"])

        result = cmd_analyze(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "HSIAO SECDED Analysis" in captured.out
        assert "k=64" in captured.out

    def test_analyze_fanin_statistics(self, capsys):
        """Test that analyze output includes fan-in statistics."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "-k", "64"])

        result = cmd_analyze(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Fan-in Statistics:" in captured.out
        assert "Maximum fan-in:" in captured.out
        assert "Minimum fan-in:" in captured.out
        assert "Average fan-in:" in captured.out

    def test_analyze_xor_depth_statistics(self, capsys):
        """Test that analyze output includes XOR depth statistics."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "-k", "64"])

        result = cmd_analyze(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "XOR Depth Statistics:" in captured.out
        assert "Maximum XOR depth:" in captured.out
        assert "Minimum XOR depth:" in captured.out
        assert "Average XOR depth:" in captured.out

    def test_analyze_basic_parameters(self, capsys):
        """Test that analyze output includes basic parameters."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "-k", "64"])

        result = cmd_analyze(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Basic Parameters:" in captured.out
        assert "Data bits (k):" in captured.out
        assert "SEC parity bits (r):" in captured.out
        assert "Total bits:" in captured.out
        assert "Overhead:" in captured.out

    def test_analyze_hamming_scheme(self, capsys):
        """Test analyze command with hamming scheme."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "-k", "64", "--scheme", "hamming"])

        result = cmd_analyze(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "HAMMING SECDED Analysis" in captured.out

    def test_analyze_hsiao_scheme(self, capsys):
        """Test analyze command with hsiao scheme."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "-k", "64", "--scheme", "hsiao"])

        result = cmd_analyze(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "HSIAO SECDED Analysis" in captured.out

    @pytest.mark.parametrize("k", [32, 64, 128])
    def test_analyze_various_data_widths(self, k, capsys):
        """Test analyze command with various data widths."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "-k", str(k)])

        result = cmd_analyze(args)

        assert result == 0
        captured = capsys.readouterr()
        assert f"k={k}" in captured.out


# =============================================================================
# Compare Command Tests
# =============================================================================


class TestCompareCommand:
    """Tests for the compare command."""

    def test_compare_single_k_value(self, capsys):
        """Test compare command with single k value."""
        parser = create_parser()
        args = parser.parse_args(["compare", "-k", "64"])

        result = cmd_compare(args)

        assert result == 0
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_compare_multiple_k_values(self, capsys):
        """Test compare command with multiple k values."""
        parser = create_parser()
        args = parser.parse_args(["compare", "-k", "32,64,128"])

        result = cmd_compare(args)

        assert result == 0
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_compare_invalid_format_non_numeric(self, capsys):
        """Test compare command with invalid non-numeric format."""
        parser = create_parser()
        args = parser.parse_args(["compare", "-k", "abc,def"])

        result = cmd_compare(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_compare_invalid_format_mixed(self, capsys):
        """Test compare command with mixed valid/invalid values."""
        parser = create_parser()
        args = parser.parse_args(["compare", "-k", "32,abc,64"])

        result = cmd_compare(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_compare_negative_k_value(self, capsys):
        """Test compare command with negative k value."""
        parser = create_parser()
        args = parser.parse_args(["compare", "-k", "-32"])

        result = cmd_compare(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_compare_zero_k_value(self, capsys):
        """Test compare command with zero k value."""
        parser = create_parser()
        args = parser.parse_args(["compare", "-k", "0"])

        result = cmd_compare(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_compare_whitespace_handling(self, capsys):
        """Test compare command handles whitespace in k values."""
        parser = create_parser()
        args = parser.parse_args(["compare", "-k", "32, 64, 128"])

        result = cmd_compare(args)

        assert result == 0
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# =============================================================================
# Subprocess Integration Tests
# =============================================================================


class TestSubprocessIntegration:
    """Integration tests using subprocess to test actual CLI execution."""

    def test_cli_version_subprocess(self):
        """Test --version flag via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "--version"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0
        assert __version__ in result.stdout

    def test_cli_help_subprocess(self):
        """Test --help flag via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0
        assert "SECDED" in result.stdout

    def test_cli_generate_subprocess(self):
        """Test generate command via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "generate", "-k", "32"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0
        assert "H-matrix" in result.stdout

    def test_cli_generate_json_subprocess(self):
        """Test generate command with JSON format via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "generate", "-k", "32", "--format", "json"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["k"] == 32

    def test_cli_verilog_subprocess(self):
        """Test verilog command via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "verilog", "-k", "32"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0
        assert "assign" in result.stdout

    def test_cli_analyze_subprocess(self):
        """Test analyze command via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "analyze", "-k", "32"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0
        assert "Fan-in" in result.stdout
        assert "XOR" in result.stdout

    def test_cli_compare_subprocess(self):
        """Test compare command via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "compare", "-k", "32,64"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0

    def test_cli_missing_command_subprocess(self):
        """Test CLI with missing command via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode != 0

    def test_cli_invalid_scheme_subprocess(self):
        """Test CLI with invalid scheme via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "generate", "-k", "32", "--scheme", "invalid"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode != 0
        assert "invalid" in result.stderr.lower() or "invalid" in result.stdout.lower()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_generate_small_k(self, capsys):
        """Test generate with small k value."""
        parser = create_parser()
        args = parser.parse_args(["generate", "-k", "8", "--format", "json"])

        result = cmd_generate(args)

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["k"] == 8

    def test_generate_large_k(self, capsys):
        """Test generate with large k value."""
        parser = create_parser()
        args = parser.parse_args(["generate", "-k", "256", "--format", "json"])

        result = cmd_generate(args)

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["k"] == 256

    def test_verilog_file_creation(self):
        """Test that verilog command creates output file correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.sv"

            parser = create_parser()
            args = parser.parse_args(["verilog", "-k", "32", "-o", str(output_path)])

            result = cmd_verilog(args)

            assert result == 0
            assert output_path.exists()
            content = output_path.read_text()
            assert len(content) > 0
            assert "assign" in content

    def test_compare_single_value_no_comma(self, capsys):
        """Test compare with single value without comma."""
        parser = create_parser()
        args = parser.parse_args(["compare", "-k", "64"])

        result = cmd_compare(args)

        assert result == 0

    def test_compare_many_values(self, capsys):
        """Test compare with many k values."""
        parser = create_parser()
        args = parser.parse_args(["compare", "-k", "32,64,128,256,512"])

        result = cmd_compare(args)

        assert result == 0
        captured = capsys.readouterr()
        assert len(captured.out) > 0
