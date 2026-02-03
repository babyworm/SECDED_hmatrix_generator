import pytest
from typing import List, Tuple
from src.gen_hmatrix import generate_h_matrix_secdec, Scheme

# Data width 파라미터 (32 ~ 2048)
DATA_WIDTHS = [32, 64, 128, 256, 512, 1024, 2048]
SCHEMES = ["hamming", "hsiao"]

@pytest.fixture(params=DATA_WIDTHS)
def data_width(request) -> int:
    return request.param

@pytest.fixture(params=SCHEMES)
def scheme(request) -> Scheme:
    return request.param

@pytest.fixture
def h_matrix(data_width: int, scheme: Scheme) -> Tuple[List[List[int]], int]:
    return generate_h_matrix_secdec(data_width, scheme)
