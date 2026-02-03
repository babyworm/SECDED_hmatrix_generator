from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple, Literal

Scheme = Literal["hamming", "hsiao"]

def _popcount(x: int) -> int:
    return x.bit_count()

def _required_r(k: int) -> int:
    # Find smallest r such that 2^r >= k + r + 1
    r = 1
    while (1 << r) < (k + r + 1):
        r += 1
    return r

def _unit_vectors(r: int) -> List[int]:
    return [1 << i for i in range(r)]

def _hamming_data_columns(r: int, k: int) -> List[int]:
    """
    Classic Hamming SEC columns: use ascending nonzero integers excluding powers of two.
    We reserve unit vectors as parity columns in systematic form, then fill data columns.
    """
    units = set(_unit_vectors(r))
    cols: List[int] = []
    v = 1
    while len(cols) < k:
        if v != 0 and v not in units and v < (1 << r):
            cols.append(v)
        v += 1
        if v >= (1 << r):  # ran out of nonzero vectors
            raise ValueError(f"Not enough vectors for k={k} with r={r} (increase r?)")
    return cols

def _all_vectors_of_weight(r: int, w: int, exclude: set[int]) -> List[int]:
    out = []
    for idxs in combinations(range(r), w):
        v = 0
        for i in idxs:
            v |= (1 << i)
        if v not in exclude:
            out.append(v)
    return out

def _hsiao_like_data_columns(r: int, k: int) -> List[int]:
    """
    Hsiao-family style (implementation-oriented):
      - choose distinct nonzero r-bit columns
      - minimize column weight (start from weight=2, then 3, 4, ...)
      - greedy row-load balancing to reduce max XOR fan-in imbalance
    Note: multiple valid solutions exist; this generator returns one deterministic solution.
    """
    units = set(_unit_vectors(r))
    chosen: List[int] = []
    exclude = set(units)  # don't use unit vectors for data (reserved for parity)

    # row_load[i] = number of selected data columns that include row i
    row_load = [0] * r

    def score(v: int) -> Tuple[int, int, int, int]:
        # Lower is better.
        # 1) primary: resulting max row load after adding v
        # 2) secondary: resulting sum row load (encourage balance)
        # 3) tertiary: popcount(v)  (keep low weight; though we iterate by weight anyway)
        # 4) tie-break: numeric value
        new_load = row_load.copy()
        for i in range(r):
            if (v >> i) & 1:
                new_load[i] += 1
        return (max(new_load), sum(new_load), _popcount(v), v)

    w = 2  # start at 2 to reduce XOR fan-in (often beats weight-3 first for larger k)
    while len(chosen) < k:
        candidates = _all_vectors_of_weight(r, w, exclude=set(exclude) | set(chosen))
        if not candidates:
            w += 1
            if w > r:
                raise ValueError(f"Ran out of candidates for k={k} with r={r}")
            continue

        # pick greedily one-by-one by best score (deterministic)
        candidates.sort(key=score)
        for v in candidates:
            if v in exclude or v in chosen:
                continue
            # accept v
            chosen.append(v)
            for i in range(r):
                if (v >> i) & 1:
                    row_load[i] += 1
            if len(chosen) >= k:
                break

        if len(chosen) < k:
            w += 1
            if w > r:
                raise ValueError(f"Ran out of candidates for k={k} with r={r}")

    return chosen

def generate_h_matrix_secdec(k: int, scheme: Scheme = "hsiao") -> Tuple[List[List[int]], int]:
    """
    Returns:
      H: (r+1) x (k+r+1) binary matrix as 0/1 ints
      r: number of SEC parity bits (overall parity adds +1 row and +1 column)
    Column order (systematic):
      [ p0..p{r-1} | d0..d{k-1} | p_overall ]
    Row order:
      [ s0..s{r-1} | overall_row ]
    """
    if k <= 0:
        raise ValueError("k must be positive")
    r = _required_r(k)

    # Build SEC part columns (length r)
    parity_cols = _unit_vectors(r)

    if scheme == "hamming":
        data_cols = _hamming_data_columns(r, k)
    elif scheme == "hsiao":
        data_cols = _hsiao_like_data_columns(r, k)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    # total columns excluding overall parity bit: k + r
    sec_cols = parity_cols + data_cols  # each is r-bit mask

    # Build full H with (r+1) rows and (k+r+1) cols
    n = k + r + 1
    H = [[0 for _ in range(n)] for _ in range(r + 1)]

    # Fill upper r rows from masks
    for col_idx, mask in enumerate(sec_cols):
        for row in range(r):
            H[row][col_idx] = (mask >> row) & 1

    # overall parity column (last column): upper r rows are 0
    # H[row][n-1] already 0

    # overall parity row: all ones across all n columns (extended SECDED)
    for col in range(n):
        H[r][col] = 1

    return H, r

def parity_equations_sv(k: int, scheme: Scheme = "hsiao",
                        d_name: str = "d", p_name: str = "p", po_name: str = "p_overall") -> str:
    """
    Emit SystemVerilog parity equations for systematic encoder:
      p[i] = XOR of d[j] where H[i, (r + j)] == 1   (data columns region)
      p_overall = ^{d, p}
    """
    H, r = generate_h_matrix_secdec(k, scheme)
    lines = []
    lines.append(f"// {scheme.upper()} SECDED encoder (k={k}, r={r})")
    lines.append(f"// input  logic [{k-1}:0] {d_name};")
    lines.append(f"// output logic [{r-1}:0] {p_name};")
    lines.append(f"// output logic         {po_name};")
    lines.append("")

    # parity bits from data portion only (systematic)
    for i in range(r):
        taps = []
        for j in range(k):
            if H[i][r + j] == 1:
                taps.append(f"{d_name}[{j}]")
        if not taps:
            rhs = "1'b0"
        else:
            rhs = "^{ " + ", ".join(taps) + " }"
        lines.append(f"assign {p_name}[{i}] = {rhs};")

    lines.append("")
    lines.append(f"assign {po_name} = ^{{ {d_name}, {p_name} }};")
    return "\n".join(lines)

# ---- quick self-test / demo ----
if __name__ == "__main__":
    for k in (32, 63):
        for scheme in ("hamming", "hsiao"):
            H, r = generate_h_matrix_secdec(k, scheme)
            # sanity: SEC columns (excluding overall) must be nonzero and unique in upper r rows
            masks = []
            for c in range(k + r):
                m = 0
                for i in range(r):
                    m |= (H[i][c] << i)
                masks.append(m)
            assert len(set(masks)) == len(masks)
            assert all(m != 0 for m in masks)
            print(parity_equations_sv(k, scheme))
            print()

