#!/usr/bin/env python3
"""
SECDED H-Matrix Generator CLI

Command-line interface for generating SECDED H-matrices, analyzing schemes,
and generating SystemVerilog encoder code.
"""

from __future__ import annotations
import argparse
import json
import sys
from typing import List

from src.gen_hmatrix import generate_h_matrix_secdec, parity_equations_sv, Scheme
from src.metrics import analyze_scheme, compare_schemes, generate_comparison_table
from src.xor_depth import calculate_xor_depth
from src.sv_generator import (
    SVGeneratorConfig,
    generate_encoder_sv,
    generate_decoder_sv,
    generate_top_module_sv,
    generate_testbench_sv,
    generate_package_sv,
    generate_all_sv_files,
)
from src.rtl_verification import (
    verify_rtl,
    print_coverage_report,
)


__version__ = "1.0.0"


def format_matrix(H: List[List[int]], r: int, k: int) -> str:
    """
    Format H-matrix as human-readable text.

    Args:
        H: H-matrix to format
        r: number of SEC parity bits
        k: number of data bits

    Returns:
        Formatted matrix string
    """
    lines = []
    rows = len(H)
    cols = len(H[0])

    lines.append(f"H-matrix ({rows} x {cols}):")
    lines.append(f"  Data bits (k): {k}")
    lines.append(f"  SEC parity bits (r): {r}")
    lines.append(f"  Total bits: {k + r + 1}")
    lines.append("")

    # Column header
    header_parts = []
    header_parts.append("  p0")
    for i in range(1, r):
        header_parts.append(f"p{i}")
    for i in range(k):
        header_parts.append(f"d{i}")
    header_parts.append("po")

    lines.append("Row  " + " ".join(f"{h:>3}" for h in header_parts))
    lines.append("-" * (5 + 4 * cols))

    # Matrix rows
    for i in range(r):
        row_str = f"s{i:>2}  "
        row_str += " ".join(f"{H[i][j]:>3}" for j in range(cols))
        lines.append(row_str)

    # Overall parity row
    row_str = f"ov  "
    row_str += " ".join(f"{H[r][j]:>3}" for j in range(cols))
    lines.append(row_str)

    return "\n".join(lines)


def format_matrix_json(H: List[List[int]], r: int, k: int, scheme: str) -> str:
    """
    Format H-matrix as JSON.

    Args:
        H: H-matrix to format
        r: number of SEC parity bits
        k: number of data bits
        scheme: scheme name

    Returns:
        JSON string
    """
    data = {
        "scheme": scheme,
        "k": k,
        "r": r,
        "total_bits": k + r + 1,
        "rows": len(H),
        "cols": len(H[0]),
        "matrix": H
    }
    return json.dumps(data, indent=2)


def cmd_generate(args: argparse.Namespace) -> int:
    """
    Generate H-matrix command handler.

    Args:
        args: parsed command-line arguments

    Returns:
        exit code (0 for success)
    """
    try:
        k = args.data_width
        scheme: Scheme = args.scheme

        # Generate H-matrix
        H, r = generate_h_matrix_secdec(k, scheme)

        # Format output
        if args.format == "matrix":
            output = format_matrix(H, r, k)
        elif args.format == "json":
            output = format_matrix_json(H, r, k, scheme)
        else:
            print(f"Error: Unknown format '{args.format}'", file=sys.stderr)
            return 1

        print(output)
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_verilog(args: argparse.Namespace) -> int:
    """
    Generate SystemVerilog code command handler.

    Args:
        args: parsed command-line arguments

    Returns:
        exit code (0 for success)
    """
    try:
        k = args.data_width
        scheme: Scheme = args.scheme

        # Generate SystemVerilog code
        sv_code = parity_equations_sv(k, scheme)

        # Output to file or stdout
        if args.output:
            with open(args.output, 'w') as f:
                f.write(sv_code)
                f.write('\n')
            print(f"Generated SystemVerilog code written to: {args.output}", file=sys.stderr)
        else:
            print(sv_code)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_analyze(args: argparse.Namespace) -> int:
    """
    Analyze scheme command handler.

    Args:
        args: parsed command-line arguments

    Returns:
        exit code (0 for success)
    """
    try:
        k = args.data_width
        scheme: Scheme = args.scheme

        # Analyze scheme
        metrics = analyze_scheme(k, scheme)

        # Print analysis
        print(f"{scheme.upper()} SECDED Analysis (k={k})")
        print("=" * 60)
        print()

        print("Basic Parameters:")
        print(f"  Data bits (k):        {metrics.k}")
        print(f"  SEC parity bits (r):  {metrics.r}")
        print(f"  Total bits:           {metrics.total_bits}")
        print(f"  Overhead:             {metrics.overhead_percent:.2f}%")
        print()

        print("Fan-in Statistics:")
        print(f"  Maximum fan-in:       {metrics.max_fanin}")
        print(f"  Minimum fan-in:       {metrics.min_fanin}")
        print(f"  Average fan-in:       {metrics.avg_fanin:.2f}")
        print(f"  Std deviation:        {metrics.fanin_std:.2f}")
        print()

        print("XOR Depth Statistics:")
        print(f"  Maximum XOR depth:    {metrics.max_xor_depth}")
        print(f"  Minimum XOR depth:    {metrics.min_xor_depth}")
        print(f"  Average XOR depth:    {metrics.avg_xor_depth:.2f}")
        print()

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """
    Compare schemes command handler.

    Args:
        args: parsed command-line arguments

    Returns:
        exit code (0 for success)
    """
    try:
        # Parse k values
        k_values = [int(k.strip()) for k in args.data_width.split(',')]

        # Validate k values
        if not k_values:
            print("Error: No data width values provided", file=sys.stderr)
            return 1

        for k in k_values:
            if k <= 0:
                print(f"Error: Invalid data width {k} (must be positive)", file=sys.stderr)
                return 1

        # Generate comparison table
        table = generate_comparison_table(k_values)
        print(table)

        return 0

    except ValueError as e:
        print(f"Error: Invalid data width format - {e}", file=sys.stderr)
        print("Expected format: -k 32,64,128,256", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_sv(args: argparse.Namespace) -> int:
    """
    Generate SystemVerilog modules command handler.

    Args:
        args: parsed command-line arguments

    Returns:
        exit code (0 for success)
    """
    import os

    try:
        k = args.data_width
        scheme: Scheme = args.scheme
        output_dir = args.output_dir or "."
        module_type = args.type

        # Create config
        config = SVGeneratorConfig(
            module_prefix=args.prefix or "secded",
            include_assertions=not args.no_assertions,
            author=args.author or "SECDED Generator"
        )

        # Generate code based on type
        if module_type == "encoder":
            code = generate_encoder_sv(k, scheme, config)
            filename = f"{config.module_prefix}_encoder_{k}.sv"
        elif module_type == "decoder":
            code = generate_decoder_sv(k, scheme, config)
            filename = f"{config.module_prefix}_decoder_{k}.sv"
        elif module_type == "top":
            code = generate_top_module_sv(k, scheme, config)
            filename = f"{config.module_prefix}_top_{k}.sv"
        elif module_type == "testbench":
            code = generate_testbench_sv(k, scheme, config)
            filename = f"tb_{config.module_prefix}_{k}.sv"
        elif module_type == "package":
            code = generate_package_sv(k, scheme, config)
            filename = f"{config.module_prefix}_pkg.sv"
        elif module_type == "all":
            # Generate all files
            files = generate_all_sv_files(k, scheme, output_dir, config)

            # Create output directory if needed
            if output_dir != ".":
                os.makedirs(output_dir, exist_ok=True)

            # Write all files
            for filename, content in files.items():
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w') as f:
                    f.write(content)
                    f.write('\n')
                print(f"Generated: {filepath}", file=sys.stderr)

            print(f"\nGenerated {len(files)} SystemVerilog files in '{output_dir}/'", file=sys.stderr)
            return 0
        else:
            print(f"Error: Unknown module type '{module_type}'", file=sys.stderr)
            return 1

        # Output single file
        if args.output:
            # Create directory if needed
            dir_path = os.path.dirname(args.output)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            with open(args.output, 'w') as f:
                f.write(code)
                f.write('\n')
            print(f"Generated: {args.output}", file=sys.stderr)
        else:
            print(code)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser with all subcommands.

    Returns:
        configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="SECDED H-Matrix Generator",
        epilog="""
Examples:
  %(prog)s generate -k 64 --scheme hsiao
  %(prog)s verilog -k 64 -o encoder.sv
  %(prog)s analyze -k 64
  %(prog)s compare -k 32,64,128,256
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        title='commands',
        description='available commands',
        dest='command',
        required=True
    )

    # Generate subcommand
    parser_generate = subparsers.add_parser(
        'generate',
        help='Generate SECDED H-matrix',
        description='Generate and display SECDED H-matrix for given parameters'
    )
    parser_generate.add_argument(
        '-k', '--data-width',
        type=int,
        required=True,
        metavar='K',
        help='number of data bits (required)'
    )
    parser_generate.add_argument(
        '--scheme',
        type=str,
        choices=['hamming', 'hsiao'],
        default='hsiao',
        help='encoding scheme (default: hsiao)'
    )
    parser_generate.add_argument(
        '--format',
        type=str,
        choices=['matrix', 'json'],
        default='matrix',
        help='output format (default: matrix)'
    )
    parser_generate.set_defaults(func=cmd_generate)

    # Verilog subcommand
    parser_verilog = subparsers.add_parser(
        'verilog',
        help='Generate SystemVerilog encoder code',
        description='Generate SystemVerilog parity encoder equations'
    )
    parser_verilog.add_argument(
        '-k', '--data-width',
        type=int,
        required=True,
        metavar='K',
        help='number of data bits (required)'
    )
    parser_verilog.add_argument(
        '--scheme',
        type=str,
        choices=['hamming', 'hsiao'],
        default='hsiao',
        help='encoding scheme (default: hsiao)'
    )
    parser_verilog.add_argument(
        '-o', '--output',
        type=str,
        metavar='FILE',
        help='output file (default: stdout)'
    )
    parser_verilog.set_defaults(func=cmd_verilog)

    # Analyze subcommand
    parser_analyze = subparsers.add_parser(
        'analyze',
        help='Analyze XOR depth and efficiency',
        description='Analyze XOR depth, fan-in statistics, and efficiency metrics'
    )
    parser_analyze.add_argument(
        '-k', '--data-width',
        type=int,
        required=True,
        metavar='K',
        help='number of data bits (required)'
    )
    parser_analyze.add_argument(
        '--scheme',
        type=str,
        choices=['hamming', 'hsiao'],
        default='hsiao',
        help='encoding scheme (default: hsiao)'
    )
    parser_analyze.set_defaults(func=cmd_analyze)

    # Compare subcommand
    parser_compare = subparsers.add_parser(
        'compare',
        help='Compare Hamming vs Hsiao schemes',
        description='Generate comparison table for Hamming vs Hsiao schemes'
    )
    parser_compare.add_argument(
        '-k', '--data-width',
        type=str,
        required=True,
        metavar='K1,K2,...',
        help='comma-separated list of data bit counts (e.g., 32,64,128,256)'
    )
    parser_compare.set_defaults(func=cmd_compare)

    # SV (SystemVerilog) subcommand
    parser_sv = subparsers.add_parser(
        'sv',
        help='Generate SystemVerilog encoder/decoder modules',
        description='Generate complete SystemVerilog RTL modules for SECDED encoder/decoder',
        epilog='''
Module types:
  encoder    - SECDED encoder module
  decoder    - SECDED decoder module with error correction
  top        - Top module containing both encoder and decoder
  testbench  - SystemVerilog testbench
  package    - Type definitions and constants
  all        - Generate all modules (encoder, decoder, top, testbench, package)

Examples:
  %(prog)s -k 64 --type encoder
  %(prog)s -k 64 --type all --output-dir ./rtl
  %(prog)s -k 128 --scheme hamming --type decoder -o decoder_128.sv
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser_sv.add_argument(
        '-k', '--data-width',
        type=int,
        required=True,
        metavar='K',
        help='number of data bits (required)'
    )
    parser_sv.add_argument(
        '--scheme',
        type=str,
        choices=['hamming', 'hsiao'],
        default='hsiao',
        help='encoding scheme (default: hsiao)'
    )
    parser_sv.add_argument(
        '--type',
        type=str,
        choices=['encoder', 'decoder', 'top', 'testbench', 'package', 'all'],
        default='all',
        help='module type to generate (default: all)'
    )
    parser_sv.add_argument(
        '-o', '--output',
        type=str,
        metavar='FILE',
        help='output file path (for single module)'
    )
    parser_sv.add_argument(
        '--output-dir',
        type=str,
        metavar='DIR',
        help='output directory (for --type all)'
    )
    parser_sv.add_argument(
        '--prefix',
        type=str,
        default='secded',
        help='module name prefix (default: secded)'
    )
    parser_sv.add_argument(
        '--author',
        type=str,
        default='SECDED Generator',
        help='author name for file headers'
    )
    parser_sv.add_argument(
        '--no-assertions',
        action='store_true',
        help='disable assertion generation'
    )
    parser_sv.set_defaults(func=cmd_sv)

    # Verify subcommand
    parser_verify = subparsers.add_parser(
        'verify',
        help='Verify RTL against Python reference using Verilator',
        description='Run coverage-driven RTL verification comparing SystemVerilog vs Python',
        epilog='''
This command requires Verilator to be installed.

Examples:
  %(prog)s -k 64 --scheme hsiao
  %(prog)s -k 32 --exhaustive --keep-files
  %(prog)s -k 128 -n 200 --work-dir ./verify_output
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser_verify.add_argument(
        '-k', '--data-width',
        type=int,
        required=True,
        metavar='K',
        help='number of data bits (required)'
    )
    parser_verify.add_argument(
        '--scheme',
        type=str,
        choices=['hamming', 'hsiao'],
        default='hsiao',
        help='encoding scheme (default: hsiao)'
    )
    parser_verify.add_argument(
        '-n', '--num-random',
        type=int,
        default=100,
        metavar='N',
        help='number of random test vectors (default: 100)'
    )
    parser_verify.add_argument(
        '--exhaustive',
        action='store_true',
        help='test all single-bit error positions (slower)'
    )
    parser_verify.add_argument(
        '--work-dir',
        type=str,
        metavar='DIR',
        help='working directory for generated files'
    )
    parser_verify.add_argument(
        '--keep-files',
        action='store_true',
        help='keep generated files after verification'
    )
    parser_verify.set_defaults(func=cmd_verify)

    return parser


def cmd_verify(args: argparse.Namespace) -> int:
    """
    Verify RTL against Python reference.

    Args:
        args: parsed command-line arguments

    Returns:
        exit code (0 for success)
    """
    import shutil

    # Check for Verilator
    if not shutil.which("verilator"):
        print("Error: Verilator not found. Please install Verilator.", file=sys.stderr)
        return 1

    try:
        k = args.data_width
        scheme = args.scheme

        print(f"SECDED RTL Verification (k={k}, scheme={scheme})")
        print("=" * 60)

        # Run verification
        result = verify_rtl(
            k=k,
            scheme=scheme,
            num_random_tests=args.num_random,
            exhaustive_errors=args.exhaustive,
            work_dir=args.work_dir,
            keep_files=args.keep_files
        )

        # Print errors if any
        if result.errors:
            print("\nErrors:")
            for e in result.errors:
                print(f"  - {e}")

        # Print results
        print(f"\nResults: {result.passed_tests}/{result.total_tests} passed")
        print(f"Status: {'PASS' if result.passed else 'FAIL'}")

        # Print coverage report
        print_coverage_report(result.coverage)

        return 0 if result.passed else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """
    Main entry point for CLI.

    Returns:
        exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    args = parser.parse_args()

    # Call appropriate command handler
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
