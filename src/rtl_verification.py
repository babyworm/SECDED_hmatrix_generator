"""
RTL Verification Module

Verifies SystemVerilog RTL implementation against Python reference model
using Verilator for simulation.

Features:
- Coverage-driven test vector generation
- Verilator compilation and simulation
- Python vs RTL result comparison
- Detailed coverage reporting
"""

from __future__ import annotations
import os
import subprocess
import tempfile
import shutil
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from src.gen_hmatrix import generate_h_matrix_secdec, Scheme
from src.secded_codec import encode, decode, inject_error, SECDEDResult
from src.sv_generator import (
    generate_encoder_sv,
    generate_decoder_sv,
    SVGeneratorConfig,
)


@dataclass
class TestVector:
    """Single test vector for RTL verification."""
    data: int
    error_positions: List[int]  # bit positions to flip (empty = no error)
    expected_corrected_data: int
    expected_error_type: int  # 0=none, 1=corrected, 2=detected
    description: str = ""


@dataclass
class CoveragePoint:
    """Coverage point tracking."""
    name: str
    hit: bool = False
    count: int = 0


@dataclass
class CoverageReport:
    """Coverage report for verification."""
    total_points: int = 0
    covered_points: int = 0
    coverage_percent: float = 0.0
    points: Dict[str, CoveragePoint] = field(default_factory=dict)

    def add_point(self, name: str):
        if name not in self.points:
            self.points[name] = CoveragePoint(name=name)
            self.total_points += 1

    def hit_point(self, name: str):
        if name in self.points:
            if not self.points[name].hit:
                self.points[name].hit = True
                self.covered_points += 1
            self.points[name].count += 1
            self.coverage_percent = (self.covered_points / self.total_points * 100) if self.total_points > 0 else 0.0


@dataclass
class VerificationResult:
    """Result of RTL verification."""
    passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    coverage: CoverageReport
    errors: List[str] = field(default_factory=list)


class RTLVerifier:
    """Verifies SystemVerilog RTL against Python reference."""

    def __init__(self, k: int, scheme: Scheme = "hsiao"):
        self.k = k
        self.scheme = scheme
        self.H, self.r = generate_h_matrix_secdec(k, scheme)
        self.n = k + self.r + 1  # codeword length

        # Coverage tracking
        self.coverage = CoverageReport()
        self._init_coverage_points()

    def _init_coverage_points(self):
        """Initialize coverage points."""
        # Data pattern coverage
        self.coverage.add_point("data_all_zeros")
        self.coverage.add_point("data_all_ones")
        self.coverage.add_point("data_alternating_01")
        self.coverage.add_point("data_alternating_10")
        self.coverage.add_point("data_random")
        self.coverage.add_point("data_single_bit_set")
        self.coverage.add_point("data_single_bit_clear")

        # Error type coverage
        self.coverage.add_point("error_none")
        self.coverage.add_point("error_single_data_bit")
        self.coverage.add_point("error_single_parity_bit")
        self.coverage.add_point("error_single_overall_parity")
        self.coverage.add_point("error_double_bit")

        # Bit position coverage (sampled)
        for i in range(min(self.n, 16)):  # First 16 positions
            self.coverage.add_point(f"bit_pos_{i}")
        if self.n > 16:
            self.coverage.add_point(f"bit_pos_{self.n-1}")  # Last position

    def generate_test_vectors(self,
                               num_random: int = 100,
                               include_exhaustive_errors: bool = False) -> List[TestVector]:
        """
        Generate coverage-driven test vectors.

        Args:
            num_random: Number of random test vectors
            include_exhaustive_errors: Test all single-bit error positions

        Returns:
            List of test vectors
        """
        vectors = []

        # 1. All zeros
        vectors.append(self._create_vector(0, [], "all_zeros"))
        self.coverage.hit_point("data_all_zeros")
        self.coverage.hit_point("error_none")

        # 2. All ones
        all_ones = (1 << self.k) - 1
        vectors.append(self._create_vector(all_ones, [], "all_ones"))
        self.coverage.hit_point("data_all_ones")

        # 3. Alternating patterns
        alt_01 = int("01" * (self.k // 2), 2) & ((1 << self.k) - 1)
        alt_10 = int("10" * (self.k // 2), 2) & ((1 << self.k) - 1)
        vectors.append(self._create_vector(alt_01, [], "alternating_01"))
        vectors.append(self._create_vector(alt_10, [], "alternating_10"))
        self.coverage.hit_point("data_alternating_01")
        self.coverage.hit_point("data_alternating_10")

        # 4. Single bit set at various positions
        for pos in [0, self.k // 4, self.k // 2, self.k - 1]:
            if pos < self.k:
                data = 1 << pos
                vectors.append(self._create_vector(data, [], f"single_bit_{pos}"))
                self.coverage.hit_point("data_single_bit_set")

        # 5. Single bit clear (all ones except one bit)
        for pos in [0, self.k // 2, self.k - 1]:
            if pos < self.k:
                data = all_ones ^ (1 << pos)
                vectors.append(self._create_vector(data, [], f"single_clear_{pos}"))
                self.coverage.hit_point("data_single_bit_clear")

        # 6. Single-bit errors in data region
        test_data = random.randint(0, (1 << self.k) - 1)
        error_positions_to_test = list(range(self.r, self.r + min(16, self.k)))  # Data bits
        if include_exhaustive_errors:
            error_positions_to_test = list(range(self.r, self.r + self.k))

        for pos in error_positions_to_test:
            vectors.append(self._create_vector(test_data, [pos], f"single_error_data_{pos}"))
            self.coverage.hit_point("error_single_data_bit")
            if pos < 16:
                self.coverage.hit_point(f"bit_pos_{pos}")

        # 7. Single-bit errors in parity region
        for pos in range(self.r):
            vectors.append(self._create_vector(test_data, [pos], f"single_error_parity_{pos}"))
            self.coverage.hit_point("error_single_parity_bit")
            self.coverage.hit_point(f"bit_pos_{pos}")

        # 8. Single-bit error in overall parity bit
        vectors.append(self._create_vector(test_data, [self.n - 1], "single_error_overall"))
        self.coverage.hit_point("error_single_overall_parity")
        self.coverage.hit_point(f"bit_pos_{self.n-1}")

        # 9. Double-bit errors
        double_error_pairs = [
            (0, 1),
            (self.r, self.r + 1),
            (0, self.n - 1),
            (self.r, self.n - 2),
        ]
        for pos1, pos2 in double_error_pairs:
            if pos1 < self.n and pos2 < self.n:
                vectors.append(self._create_vector(test_data, [pos1, pos2], f"double_error_{pos1}_{pos2}"))
                self.coverage.hit_point("error_double_bit")

        # 10. Random test vectors
        for i in range(num_random):
            data = random.randint(0, (1 << self.k) - 1)

            # Random error pattern: 70% no error, 20% single, 10% double
            r = random.random()
            if r < 0.7:
                error_pos = []
            elif r < 0.9:
                error_pos = [random.randint(0, self.n - 1)]
            else:
                p1 = random.randint(0, self.n - 1)
                p2 = random.randint(0, self.n - 1)
                while p2 == p1:
                    p2 = random.randint(0, self.n - 1)
                error_pos = [p1, p2]

            vectors.append(self._create_vector(data, error_pos, f"random_{i}"))
            self.coverage.hit_point("data_random")

        return vectors

    def _create_vector(self, data: int, error_positions: List[int], description: str) -> TestVector:
        """Create a test vector with expected results from Python model."""
        # Encode data
        codeword = encode(data, self.H, self.r, self.k)

        # Inject errors
        if error_positions:
            corrupted = inject_error(codeword, error_positions)
        else:
            corrupted = codeword

        # Decode and get expected result
        result = decode(corrupted, self.H, self.r, self.k)

        # Map error type
        error_type_map = {
            "none": 0,
            "single_corrected": 1,
            "double_detected": 2,
            "unknown": 3,
        }
        expected_error_type = error_type_map.get(result.error_type, 3)

        return TestVector(
            data=data,
            error_positions=error_positions,
            expected_corrected_data=result.corrected_data,
            expected_error_type=expected_error_type,
            description=description,
        )

    def generate_verilator_testbench(self, vectors: List[TestVector], output_dir: str) -> str:
        """
        Generate C++ testbench for Verilator simulation (legacy, not used).
        """
        return self._generate_wrapper_testbench(vectors, output_dir)

    def _generate_wrapper_testbench(self, vectors: List[TestVector], output_dir: str) -> str:
        """
        Generate C++ testbench for Verilator simulation using wrapper module.

        Args:
            vectors: Test vectors to run
            output_dir: Directory to write files

        Returns:
            Path to generated testbench
        """
        # Determine if we need wide types (>64 bits)
        use_wide_codeword = self.n > 64
        use_wide_data = self.k > 64

        # Calculate number of 32-bit words needed for VlWide
        codeword_words = (self.n + 31) // 32
        data_words = (self.k + 31) // 32

        if use_wide_codeword and use_wide_data:
            # Both data and codeword need VlWide
            tb_code = self._generate_wide_testbench(vectors, codeword_words, data_words)
        elif use_wide_codeword:
            # Only codeword needs VlWide, data fits in uint64_t
            tb_code = self._generate_mixed_testbench(vectors, codeword_words)
        else:
            # Both fit in uint64_t
            tb_code = self._generate_simple_testbench(vectors)

        tb_path = os.path.join(output_dir, "tb_main.cpp")
        with open(tb_path, 'w') as f:
            f.write(tb_code)

        return tb_path

    def _generate_simple_testbench(self, vectors: List[TestVector]) -> str:
        """Generate testbench for data widths <= 64 bits."""
        tb_code = f'''
#include <iostream>
#include <fstream>
#include <cstdint>
#include "Vsecded_wrapper_{self.k}.h"
#include "verilated.h"

// Test vector structure
struct TestVector {{
    uint64_t data;
    uint64_t error_mask;  // bits to flip in codeword
    uint64_t expected_data;
    int expected_error_type;
    const char* description;
}};

int main(int argc, char** argv) {{
    Verilated::commandArgs(argc, argv);

    // Instantiate wrapper (contains encoder + decoder)
    Vsecded_wrapper_{self.k}* dut = new Vsecded_wrapper_{self.k};

    int pass_count = 0;
    int fail_count = 0;

    // Test vectors
    TestVector vectors[] = {{
'''

        # Add test vectors
        for v in vectors:
            # Calculate error mask from positions
            error_mask = 0
            for pos in v.error_positions:
                error_mask |= (1 << pos)

            # Escape description for C string
            desc = v.description.replace('"', '\\"')
            tb_code += f'        {{{v.data}ULL, {error_mask}ULL, {v.expected_corrected_data}ULL, {v.expected_error_type}, "{desc}"}},\n'

        tb_code += f'''    }};

    int num_tests = sizeof(vectors) / sizeof(vectors[0]);

    for (int i = 0; i < num_tests; i++) {{
        TestVector& tv = vectors[i];

        // Reset
        dut->rst_n = 0;
        dut->clk = 0;
        dut->eval();
        dut->clk = 1;
        dut->eval();
        dut->clk = 0;
        dut->eval();

        // Release reset
        dut->rst_n = 1;

        // Encode: set data input
        dut->enc_data_in = tv.data;
        dut->enc_input_valid = 1;
        dut->enc_output_ready = 1;
        dut->dec_input_valid = 0;
        dut->dec_output_ready = 1;

        // Clock cycle for encoding
        dut->clk = 1;
        dut->eval();
        dut->clk = 0;
        dut->eval();

        // Wait for encoder output
        dut->clk = 1;
        dut->eval();
        dut->clk = 0;
        dut->eval();

        // Get codeword and inject error
        uint64_t codeword = dut->enc_codeword;
        uint64_t corrupted = codeword ^ tv.error_mask;

        // Decode: set codeword input
        dut->enc_input_valid = 0;
        dut->dec_codeword = corrupted;
        dut->dec_input_valid = 1;

        // Clock cycle for decoding
        dut->clk = 1;
        dut->eval();
        dut->clk = 0;
        dut->eval();

        // Wait for decoder output
        dut->clk = 1;
        dut->eval();
        dut->clk = 0;
        dut->eval();

        // Check results
        uint64_t actual_data = dut->dec_corrected_data;
        int actual_error_type = dut->dec_error_type;

        bool data_match = (actual_data == tv.expected_data);
        bool error_match = (actual_error_type == tv.expected_error_type);

        // For double-bit errors, we don't check data (it's undefined)
        if (tv.expected_error_type == 2) {{
            data_match = true;  // Don't care about data for double errors
        }}

        if (data_match && error_match) {{
            pass_count++;
        }} else {{
            fail_count++;
            std::cout << "[FAIL] " << tv.description << std::endl;
            std::cout << "  Data: expected=0x" << std::hex << tv.expected_data
                      << ", actual=0x" << actual_data << std::endl;
            std::cout << "  Error type: expected=" << std::dec << tv.expected_error_type
                      << ", actual=" << actual_error_type << std::endl;
        }}
    }}

    // Summary
    std::cout << "\\n========================================" << std::endl;
    std::cout << "Verification Results (k={self.k}, {self.scheme})" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total: " << num_tests << ", Pass: " << pass_count << ", Fail: " << fail_count << std::endl;

    // Write results to file for Python to read
    std::ofstream result_file("verification_result.txt");
    result_file << pass_count << " " << fail_count << std::endl;
    result_file.close();

    delete dut;

    return (fail_count == 0) ? 0 : 1;
}}
'''
        return tb_code

    def _generate_mixed_testbench(self, vectors: List[TestVector], codeword_words: int) -> str:
        """Generate testbench for data <= 64 bits but codeword > 64 bits."""
        tb_code = f'''
#include <iostream>
#include <fstream>
#include <cstdint>
#include "Vsecded_wrapper_{self.k}.h"
#include "verilated.h"

// Helper to set VlWide from array of 32-bit words
template<size_t N>
void set_wide(VlWide<N>& dest, const uint32_t* src) {{
    for (size_t i = 0; i < N; i++) {{
        dest[i] = src[i];
    }}
}}

// Helper to XOR VlWide with another VlWide
template<size_t N>
void xor_wide(VlWide<N>& dest, const VlWide<N>& a, const VlWide<N>& b) {{
    for (size_t i = 0; i < N; i++) {{
        dest[i] = a[i] ^ b[i];
    }}
}}

// Test vector structure
struct TestVector {{
    uint64_t data;
    uint32_t error_mask[{codeword_words}];  // bits to flip in codeword (wide)
    uint64_t expected_data;
    int expected_error_type;
    const char* description;
}};

int main(int argc, char** argv) {{
    Verilated::commandArgs(argc, argv);

    // Instantiate wrapper (contains encoder + decoder)
    Vsecded_wrapper_{self.k}* dut = new Vsecded_wrapper_{self.k};

    int pass_count = 0;
    int fail_count = 0;

    // Test vectors
    TestVector vectors[] = {{
'''

        # Add test vectors
        for v in vectors:
            # Calculate error mask from positions
            error_mask = 0
            for pos in v.error_positions:
                error_mask |= (1 << pos)
            error_words_arr = self._int_to_word_array(error_mask, codeword_words)
            error_str = ", ".join(f"0x{w:08x}U" for w in error_words_arr)

            desc = v.description.replace('"', '\\"')
            tb_code += f'        {{{v.data}ULL, {{{error_str}}}, {v.expected_corrected_data}ULL, {v.expected_error_type}, "{desc}"}},\n'

        tb_code += f'''    }};

    int num_tests = sizeof(vectors) / sizeof(vectors[0]);

    for (int i = 0; i < num_tests; i++) {{
        TestVector& tv = vectors[i];

        // Reset
        dut->rst_n = 0;
        dut->clk = 0;
        dut->eval();
        dut->clk = 1;
        dut->eval();
        dut->clk = 0;
        dut->eval();

        // Release reset
        dut->rst_n = 1;

        // Encode: set data input (uint64_t)
        dut->enc_data_in = tv.data;
        dut->enc_input_valid = 1;
        dut->enc_output_ready = 1;
        dut->dec_input_valid = 0;
        dut->dec_output_ready = 1;

        // Clock cycle for encoding
        dut->clk = 1;
        dut->eval();
        dut->clk = 0;
        dut->eval();

        // Wait for encoder output
        dut->clk = 1;
        dut->eval();
        dut->clk = 0;
        dut->eval();

        // Get codeword (VlWide) and inject error
        VlWide<{codeword_words}> error_mask_wide;
        set_wide(error_mask_wide, tv.error_mask);

        VlWide<{codeword_words}> corrupted;
        xor_wide(corrupted, dut->enc_codeword, error_mask_wide);

        // Decode: set codeword input (VlWide)
        dut->enc_input_valid = 0;
        set_wide(dut->dec_codeword, (const uint32_t*)corrupted.data());
        dut->dec_input_valid = 1;

        // Clock cycle for decoding
        dut->clk = 1;
        dut->eval();
        dut->clk = 0;
        dut->eval();

        // Wait for decoder output
        dut->clk = 1;
        dut->eval();
        dut->clk = 0;
        dut->eval();

        // Check results (data is uint64_t)
        uint64_t actual_data = dut->dec_corrected_data;
        int actual_error_type = dut->dec_error_type;

        bool data_match = (actual_data == tv.expected_data);
        bool error_match = (actual_error_type == tv.expected_error_type);

        // For double-bit errors, we don't check data (it's undefined)
        if (tv.expected_error_type == 2) {{
            data_match = true;  // Don't care about data for double errors
        }}

        if (data_match && error_match) {{
            pass_count++;
        }} else {{
            fail_count++;
            std::cout << "[FAIL] " << tv.description << std::endl;
            std::cout << "  Data: expected=0x" << std::hex << tv.expected_data
                      << ", actual=0x" << actual_data << std::endl;
            std::cout << "  Error type: expected=" << std::dec << tv.expected_error_type
                      << ", actual=" << actual_error_type << std::endl;
        }}
    }}

    // Summary
    std::cout << "\\n========================================" << std::endl;
    std::cout << "Verification Results (k={self.k}, {self.scheme})" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total: " << num_tests << ", Pass: " << pass_count << ", Fail: " << fail_count << std::endl;

    // Write results to file for Python to read
    std::ofstream result_file("verification_result.txt");
    result_file << pass_count << " " << fail_count << std::endl;
    result_file.close();

    delete dut;

    return (fail_count == 0) ? 0 : 1;
}}
'''
        return tb_code

    def _generate_wide_testbench(self, vectors: List[TestVector], codeword_words: int, data_words: int) -> str:
        """Generate testbench for data widths > 64 bits using VlWide."""
        # For codeword > 64 bits, we need to use VlWide<N>
        # VlWide uses 32-bit words internally

        tb_code = f'''
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include "Vsecded_wrapper_{self.k}.h"
#include "verilated.h"

// Helper to set VlWide from array of 32-bit words
template<size_t N>
void set_wide(VlWide<N>& dest, const uint32_t* src) {{
    for (size_t i = 0; i < N; i++) {{
        dest[i] = src[i];
    }}
}}

// Helper to XOR VlWide with another VlWide
template<size_t N>
void xor_wide(VlWide<N>& dest, const VlWide<N>& a, const VlWide<N>& b) {{
    for (size_t i = 0; i < N; i++) {{
        dest[i] = a[i] ^ b[i];
    }}
}}

// Helper to compare VlWide
template<size_t N>
bool eq_wide(const VlWide<N>& a, const VlWide<N>& b) {{
    for (size_t i = 0; i < N; i++) {{
        if (a[i] != b[i]) return false;
    }}
    return true;
}}

// Test vector structure with 32-bit word arrays
struct TestVector {{
    uint32_t data[{data_words}];
    uint32_t error_mask[{codeword_words}];  // bits to flip in codeword
    uint32_t expected_data[{data_words}];
    int expected_error_type;
    const char* description;
}};

int main(int argc, char** argv) {{
    Verilated::commandArgs(argc, argv);

    // Instantiate wrapper (contains encoder + decoder)
    Vsecded_wrapper_{self.k}* dut = new Vsecded_wrapper_{self.k};

    int pass_count = 0;
    int fail_count = 0;

    // Test vectors (data stored as 32-bit word arrays, LSW first)
    TestVector vectors[] = {{
'''

        # Add test vectors with wide integer support
        for v in vectors:
            # Convert data to 32-bit word arrays
            data_words_arr = self._int_to_word_array(v.data, data_words)
            expected_words_arr = self._int_to_word_array(v.expected_corrected_data, data_words)

            # Calculate error mask from positions
            error_mask = 0
            for pos in v.error_positions:
                error_mask |= (1 << pos)
            error_words_arr = self._int_to_word_array(error_mask, codeword_words)

            # Format arrays for C++
            data_str = ", ".join(f"0x{w:08x}U" for w in data_words_arr)
            error_str = ", ".join(f"0x{w:08x}U" for w in error_words_arr)
            expected_str = ", ".join(f"0x{w:08x}U" for w in expected_words_arr)

            desc = v.description.replace('"', '\\"')
            tb_code += f'        {{{{{data_str}}}, {{{error_str}}}, {{{expected_str}}}, {v.expected_error_type}, "{desc}"}},\n'

        tb_code += f'''    }};

    int num_tests = sizeof(vectors) / sizeof(vectors[0]);

    for (int i = 0; i < num_tests; i++) {{
        TestVector& tv = vectors[i];

        // Reset
        dut->rst_n = 0;
        dut->clk = 0;
        dut->eval();
        dut->clk = 1;
        dut->eval();
        dut->clk = 0;
        dut->eval();

        // Release reset
        dut->rst_n = 1;

        // Encode: set data input
        set_wide(dut->enc_data_in, tv.data);
        dut->enc_input_valid = 1;
        dut->enc_output_ready = 1;
        dut->dec_input_valid = 0;
        dut->dec_output_ready = 1;

        // Clock cycle for encoding
        dut->clk = 1;
        dut->eval();
        dut->clk = 0;
        dut->eval();

        // Wait for encoder output
        dut->clk = 1;
        dut->eval();
        dut->clk = 0;
        dut->eval();

        // Get codeword and inject error
        VlWide<{codeword_words}> error_mask_wide;
        set_wide(error_mask_wide, tv.error_mask);

        VlWide<{codeword_words}> corrupted;
        xor_wide(corrupted, dut->enc_codeword, error_mask_wide);

        // Decode: set codeword input
        dut->enc_input_valid = 0;
        set_wide(dut->dec_codeword, (const uint32_t*)corrupted.data());
        dut->dec_input_valid = 1;

        // Clock cycle for decoding
        dut->clk = 1;
        dut->eval();
        dut->clk = 0;
        dut->eval();

        // Wait for decoder output
        dut->clk = 1;
        dut->eval();
        dut->clk = 0;
        dut->eval();

        // Check results
        VlWide<{data_words}> expected_wide;
        set_wide(expected_wide, tv.expected_data);

        bool data_match = eq_wide(dut->dec_corrected_data, expected_wide);
        int actual_error_type = dut->dec_error_type;
        bool error_match = (actual_error_type == tv.expected_error_type);

        // For double-bit errors, we don't check data (it's undefined)
        if (tv.expected_error_type == 2) {{
            data_match = true;  // Don't care about data for double errors
        }}

        if (data_match && error_match) {{
            pass_count++;
        }} else {{
            fail_count++;
            std::cout << "[FAIL] " << tv.description << std::endl;
            std::cout << "  Error type: expected=" << tv.expected_error_type
                      << ", actual=" << actual_error_type << std::endl;
        }}
    }}

    // Summary
    std::cout << "\\n========================================" << std::endl;
    std::cout << "Verification Results (k={self.k}, {self.scheme})" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total: " << num_tests << ", Pass: " << pass_count << ", Fail: " << fail_count << std::endl;

    // Write results to file for Python to read
    std::ofstream result_file("verification_result.txt");
    result_file << pass_count << " " << fail_count << std::endl;
    result_file.close();

    delete dut;

    return (fail_count == 0) ? 0 : 1;
}}
'''
        return tb_code

    def _int_to_word_array(self, value: int, num_words: int) -> List[int]:
        """Convert integer to array of 32-bit words (LSW first)."""
        words = []
        for i in range(num_words):
            words.append((value >> (i * 32)) & 0xFFFFFFFF)
        return words

    def run_verilator_verification(self,
                                    vectors: List[TestVector],
                                    work_dir: Optional[str] = None,
                                    keep_files: bool = False) -> VerificationResult:
        """
        Run full Verilator verification.

        Args:
            vectors: Test vectors to run
            work_dir: Working directory (temp if None)
            keep_files: Keep generated files after verification

        Returns:
            VerificationResult with pass/fail status and coverage
        """
        # Create working directory
        if work_dir:
            os.makedirs(work_dir, exist_ok=True)
            temp_dir = None
        else:
            temp_dir = tempfile.mkdtemp(prefix="secded_verify_")
            work_dir = temp_dir

        errors = []

        try:
            # Generate RTL files
            config = SVGeneratorConfig(include_assertions=False)

            encoder_sv = generate_encoder_sv(self.k, self.scheme, config)
            decoder_sv = generate_decoder_sv(self.k, self.scheme, config)

            encoder_path = os.path.join(work_dir, f"secded_encoder_{self.k}.sv")
            decoder_path = os.path.join(work_dir, f"secded_decoder_{self.k}.sv")

            with open(encoder_path, 'w') as f:
                f.write(encoder_sv)
            with open(decoder_path, 'w') as f:
                f.write(decoder_sv)

            # Generate testbench
            tb_path = self.generate_verilator_testbench(vectors, work_dir)

            # Compile with Verilator - use --top-module for each and create wrapper
            # First, create a wrapper module that instantiates both
            wrapper_sv = f'''
module secded_wrapper_{self.k} (
    input  logic                      clk,
    input  logic                      rst_n,
    // Encoder
    input  logic                      enc_input_valid,
    output logic                      enc_input_ready,
    input  logic [{self.k-1}:0]       enc_data_in,
    output logic                      enc_output_valid,
    input  logic                      enc_output_ready,
    output logic [{self.n-1}:0]       enc_codeword,
    // Decoder
    input  logic                      dec_input_valid,
    output logic                      dec_input_ready,
    input  logic [{self.n-1}:0]       dec_codeword,
    output logic                      dec_output_valid,
    input  logic                      dec_output_ready,
    output logic [{self.k-1}:0]       dec_corrected_data,
    output logic [{self.r-1}:0]       dec_syndrome,
    output logic                      dec_overall_parity,
    output logic [1:0]                dec_error_type,
    output logic                      dec_error_corrected
);

    secded_encoder_{self.k} u_encoder (
        .clk(clk),
        .rst_n(rst_n),
        .input_valid(enc_input_valid),
        .input_ready(enc_input_ready),
        .data_in(enc_data_in),
        .output_valid(enc_output_valid),
        .output_ready(enc_output_ready),
        .codeword(enc_codeword)
    );

    secded_decoder_{self.k} u_decoder (
        .clk(clk),
        .rst_n(rst_n),
        .input_valid(dec_input_valid),
        .input_ready(dec_input_ready),
        .codeword(dec_codeword),
        .output_valid(dec_output_valid),
        .output_ready(dec_output_ready),
        .corrected_data(dec_corrected_data),
        .syndrome(dec_syndrome),
        .overall_parity(dec_overall_parity),
        .error_type(dec_error_type),
        .error_corrected(dec_error_corrected)
    );

endmodule
'''
            wrapper_path = os.path.join(work_dir, f"secded_wrapper_{self.k}.sv")
            with open(wrapper_path, 'w') as f:
                f.write(wrapper_sv)

            # Update testbench to use wrapper
            self._generate_wrapper_testbench(vectors, work_dir)

            # Compile with Verilator
            compile_cmd = [
                "verilator",
                "--cc",
                "--exe",
                "--build",
                "-j", "4",
                "-Wno-fatal",
                "-Wno-WIDTH",
                "-Wno-CASEINCOMPLETE",
                "-Wno-UNOPTFLAT",
                f"--top-module", f"secded_wrapper_{self.k}",
                f"secded_encoder_{self.k}.sv",
                f"secded_decoder_{self.k}.sv",
                f"secded_wrapper_{self.k}.sv",
                "tb_main.cpp",
                "-o", "secded_verify"
            ]

            result = subprocess.run(
                compile_cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                errors.append(f"Verilator compilation failed:\n{result.stderr}")
                return VerificationResult(
                    passed=False,
                    total_tests=len(vectors),
                    passed_tests=0,
                    failed_tests=len(vectors),
                    coverage=self.coverage,
                    errors=errors
                )

            # Run simulation
            sim_path = os.path.join(work_dir, "obj_dir", "secded_verify")

            result = subprocess.run(
                [sim_path],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300
            )

            # Parse results
            result_file = os.path.join(work_dir, "verification_result.txt")
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    parts = f.read().strip().split()
                    pass_count = int(parts[0])
                    fail_count = int(parts[1])
            else:
                errors.append("Result file not found")
                pass_count = 0
                fail_count = len(vectors)

            return VerificationResult(
                passed=(fail_count == 0),
                total_tests=len(vectors),
                passed_tests=pass_count,
                failed_tests=fail_count,
                coverage=self.coverage,
                errors=errors
            )

        except subprocess.TimeoutExpired:
            errors.append("Simulation timed out")
            return VerificationResult(
                passed=False,
                total_tests=len(vectors),
                passed_tests=0,
                failed_tests=len(vectors),
                coverage=self.coverage,
                errors=errors
            )
        except Exception as e:
            errors.append(f"Verification error: {str(e)}")
            return VerificationResult(
                passed=False,
                total_tests=len(vectors),
                passed_tests=0,
                failed_tests=len(vectors),
                coverage=self.coverage,
                errors=errors
            )
        finally:
            if temp_dir and not keep_files:
                shutil.rmtree(temp_dir, ignore_errors=True)


def verify_rtl(k: int,
               scheme: Scheme = "hsiao",
               num_random_tests: int = 100,
               exhaustive_errors: bool = False,
               work_dir: Optional[str] = None,
               keep_files: bool = False) -> VerificationResult:
    """
    Convenience function to verify RTL implementation.

    Args:
        k: Data width in bits
        scheme: "hamming" or "hsiao"
        num_random_tests: Number of random test vectors
        exhaustive_errors: Test all single-bit error positions
        work_dir: Working directory (uses temp if None)
        keep_files: Keep generated files

    Returns:
        VerificationResult
    """
    verifier = RTLVerifier(k, scheme)
    vectors = verifier.generate_test_vectors(
        num_random=num_random_tests,
        include_exhaustive_errors=exhaustive_errors
    )
    return verifier.run_verilator_verification(
        vectors,
        work_dir=work_dir,
        keep_files=keep_files
    )


def print_coverage_report(coverage: CoverageReport):
    """Print detailed coverage report."""
    print("\n" + "=" * 60)
    print("Coverage Report")
    print("=" * 60)
    print(f"Coverage: {coverage.covered_points}/{coverage.total_points} ({coverage.coverage_percent:.1f}%)")
    print("-" * 60)

    # Group by prefix
    groups = {}
    for name, point in coverage.points.items():
        prefix = name.split("_")[0]
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(point)

    for group_name, points in sorted(groups.items()):
        hit_count = sum(1 for p in points if p.hit)
        print(f"\n{group_name}: {hit_count}/{len(points)}")
        for p in points:
            status = "✓" if p.hit else "✗"
            print(f"  {status} {p.name} (hits: {p.count})")


# Demo / CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RTL Verification")
    parser.add_argument("-k", "--data-width", type=int, default=32, help="Data width")
    parser.add_argument("--scheme", choices=["hamming", "hsiao"], default="hsiao")
    parser.add_argument("-n", "--num-random", type=int, default=50, help="Random tests")
    parser.add_argument("--exhaustive", action="store_true", help="Test all error positions")
    parser.add_argument("--keep-files", action="store_true", help="Keep generated files")
    parser.add_argument("--work-dir", type=str, help="Working directory")

    args = parser.parse_args()

    print(f"Verifying SECDED RTL (k={args.data_width}, scheme={args.scheme})")
    print("-" * 60)

    result = verify_rtl(
        k=args.data_width,
        scheme=args.scheme,
        num_random_tests=args.num_random,
        exhaustive_errors=args.exhaustive,
        work_dir=args.work_dir,
        keep_files=args.keep_files
    )

    if result.errors:
        print("\nErrors:")
        for e in result.errors:
            print(f"  - {e}")

    print(f"\nResults: {result.passed_tests}/{result.total_tests} passed")
    print(f"Status: {'PASS' if result.passed else 'FAIL'}")

    print_coverage_report(result.coverage)
