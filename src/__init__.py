"""SECDED H-Matrix Generator Package"""
from .gen_hmatrix import generate_h_matrix_secdec, parity_equations_sv, Scheme
from .xor_depth import calculate_xor_depth, XORDepthResult
from .metrics import analyze_scheme, compare_schemes, SchemeMetrics
from .secded_codec import encode, decode, inject_error, SECDEDResult
from .sv_generator import (
    SVGeneratorConfig,
    generate_encoder_sv,
    generate_decoder_sv,
    generate_top_module_sv,
    generate_testbench_sv,
    generate_package_sv,
    generate_all_sv_files,
)
from .rtl_verification import (
    RTLVerifier,
    TestVector,
    CoverageReport,
    VerificationResult,
    verify_rtl,
    print_coverage_report,
)

__all__ = [
    # H-matrix generation
    "generate_h_matrix_secdec",
    "parity_equations_sv",
    "Scheme",
    # XOR depth analysis
    "calculate_xor_depth",
    "XORDepthResult",
    # Efficiency metrics
    "analyze_scheme",
    "compare_schemes",
    "SchemeMetrics",
    # SECDED codec
    "encode",
    "decode",
    "inject_error",
    "SECDEDResult",
    # SystemVerilog generation
    "SVGeneratorConfig",
    "generate_encoder_sv",
    "generate_decoder_sv",
    "generate_top_module_sv",
    "generate_testbench_sv",
    "generate_package_sv",
    "generate_all_sv_files",
    # RTL verification
    "RTLVerifier",
    "TestVector",
    "CoverageReport",
    "VerificationResult",
    "verify_rtl",
    "print_coverage_report",
]
