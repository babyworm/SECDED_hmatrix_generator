"""
H-matrix correctness tests for SECDED generator.
Tests dimensions, uniqueness, non-zero columns, unit vectors, and overall parity.
"""
import pytest
import math
from src.gen_hmatrix import generate_h_matrix_secdec


# Test data widths as specified: 32 to 2048
DATA_WIDTHS = [32, 64, 128, 256, 512, 1024, 2048]
SCHEMES = ["hamming", "hsiao"]


def _required_r(k: int) -> int:
    """Calculate minimum parity bits for SECDED: 2^r >= k + r + 1"""
    r = 1
    while (1 << r) < (k + r + 1):
        r += 1
    return r


def _column_to_int(H, col_idx: int, num_rows: int) -> int:
    """Convert a column to an integer (bit mask) for comparison."""
    val = 0
    for row in range(num_rows):
        val |= (H[row][col_idx] << row)
    return val


@pytest.mark.parametrize("data_width", DATA_WIDTHS)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_matrix_dimensions(data_width, scheme):
    """Test that the H-matrix has correct dimensions (r+1) x (k+r+1)."""
    H, r = generate_h_matrix_secdec(data_width, scheme)

    k = data_width
    expected_rows = r + 1
    expected_cols = k + r + 1

    actual_rows = len(H)
    actual_cols = len(H[0]) if H else 0

    assert actual_rows == expected_rows, \
        f"Expected {expected_rows} rows, got {actual_rows}"
    assert actual_cols == expected_cols, \
        f"Expected {expected_cols} cols, got {actual_cols}"


@pytest.mark.parametrize("data_width", DATA_WIDTHS)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_columns_unique(data_width, scheme):
    """Test that all SEC columns (excluding overall parity) are unique in top r rows."""
    H, r = generate_h_matrix_secdec(data_width, scheme)

    k = data_width
    n = k + r + 1

    # Convert each column (excluding last overall parity column) to integer
    col_masks = []
    for col_idx in range(n - 1):  # exclude last column (overall parity)
        mask = _column_to_int(H, col_idx, r)  # only top r rows
        col_masks.append(mask)

    # Check uniqueness
    assert len(set(col_masks)) == len(col_masks), \
        f"Found duplicate columns in top {r} rows"


@pytest.mark.parametrize("data_width", DATA_WIDTHS)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_columns_nonzero(data_width, scheme):
    """Test that all SEC columns (excluding overall parity) are non-zero in top r rows."""
    H, r = generate_h_matrix_secdec(data_width, scheme)

    k = data_width
    n = k + r + 1

    # Check each column (excluding last overall parity column)
    for col_idx in range(n - 1):
        mask = _column_to_int(H, col_idx, r)
        assert mask != 0, \
            f"Column {col_idx} is zero vector in top {r} rows"


@pytest.mark.parametrize("data_width", DATA_WIDTHS)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_parity_columns_unit_vectors(data_width, scheme):
    """Test that the first r columns are unit vectors (identity matrix pattern)."""
    H, r = generate_h_matrix_secdec(data_width, scheme)

    # First r columns should be unit vectors in top r rows
    for i in range(r):
        # Expected: only H[i][i] = 1, others = 0 in top r rows
        for row in range(r):
            expected = 1 if row == i else 0
            actual = H[row][i]
            assert actual == expected, \
                f"Parity column {i}, row {row}: expected {expected}, got {actual}"

        # Last row (overall parity) should be 1
        assert H[r][i] == 1, \
            f"Parity column {i}, overall row should be 1"


@pytest.mark.parametrize("data_width", DATA_WIDTHS)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_overall_parity_row(data_width, scheme):
    """Test that the last row (overall parity row) is all ones."""
    H, r = generate_h_matrix_secdec(data_width, scheme)

    k = data_width
    n = k + r + 1

    # Check that all elements in last row are 1
    last_row = H[r]
    for col_idx, val in enumerate(last_row):
        assert val == 1, \
            f"Overall parity row at column {col_idx}: expected 1, got {val}"


@pytest.mark.parametrize("data_width", DATA_WIDTHS)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_data_columns_multi_bit(data_width, scheme):
    """Test that data columns have weight >= 2 (multi-bit for error correction)."""
    H, r = generate_h_matrix_secdec(data_width, scheme)

    k = data_width

    # Data columns start at index r (after parity columns)
    for col_idx in range(r, r + k):
        weight = sum(H[row][col_idx] for row in range(r))
        assert weight >= 2, \
            f"Data column {col_idx} has weight {weight} < 2 in top {r} rows"


@pytest.mark.parametrize("data_width", DATA_WIDTHS)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_overall_parity_column(data_width, scheme):
    """Test that the last column (overall parity bit) is correct."""
    H, r = generate_h_matrix_secdec(data_width, scheme)

    k = data_width
    n = k + r + 1
    last_col_idx = n - 1

    # Top r rows should be 0 for overall parity column
    for row in range(r):
        assert H[row][last_col_idx] == 0, \
            f"Overall parity column at row {row}: expected 0, got {H[row][last_col_idx]}"

    # Last row should be 1
    assert H[r][last_col_idx] == 1, \
        f"Overall parity column at overall row: expected 1"
