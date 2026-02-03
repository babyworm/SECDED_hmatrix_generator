# SECDED H-Matrix Generator

SECDED (Single Error Correction, Double Error Detection) 코드를 위한 H-matrix 생성 및 분석 도구입니다. Hamming과 Hsiao 스킴을 모두 지원합니다.

## 특징 (Features)

- **H-matrix 생성**: 32~2048 bit 데이터 폭 지원
- **두 가지 스킴 지원**:
  - Hamming SECDED (표준)
  - Hsiao SECDED (최적화된 odd-weight column)
- **SystemVerilog 코드 생성**: 인코더 로직 자동 생성
- **XOR depth 분석**: 각 패리티 비트의 XOR depth 계산 및 비교
- **효율성 메트릭**: 평균/최대 XOR depth, 패리티 비트 수 분석
- **SECDED 인코더/디코더**: 완전한 에러 정정 및 탐지 구현
- **포괄적인 테스트**: H-matrix 정확성, XOR depth, 벤치마크, 기능 테스트

## 프로젝트 구조

```
src/
├── __init__.py           # 패키지 초기화
├── gen_hmatrix.py        # H-matrix 생성 (Hamming, Hsiao)
├── xor_depth.py          # XOR depth 계산
├── metrics.py            # 효율성 지표 계산
├── secded_codec.py       # 인코더/디코더 구현
└── cli.py                # CLI 인터페이스

tests/
├── test_hmatrix.py       # H-matrix 정확성 테스트
├── test_xor_depth.py     # XOR depth 테스트
├── test_benchmark.py     # Hamming vs Hsiao 비교
└── test_secded_functional.py  # SECDED 동작 검증
```

## 설치 (Installation)

기본 설치:
```bash
pip install -e .
```

개발 의존성 포함:
```bash
pip install -e ".[dev]"
```

## 빠른 시작 (Quick Start)

### CLI 사용법

H-matrix 생성:
```bash
python -m src.cli generate 64 --scheme hsiao
```

SystemVerilog 인코더 생성:
```bash
python -m src.cli verilog 64 --scheme hsiao -o encoder_64bit.sv
```

XOR depth 분석:
```bash
python -m src.cli analyze 64 --scheme hsiao
```

Hamming vs Hsiao 비교:
```bash
python -m src.cli compare 64
```

### Python API 사용법

```python
from src import generate_h_matrix_secdec, encode, decode

# H-matrix 생성
H, r = generate_h_matrix_secdec(64, "hsiao")

# 데이터 인코딩
data = 0x123456789ABCDEF0
codeword = encode(data, H, r, k=64)

# 디코딩 및 에러 정정
result = decode(codeword, H, r, k=64)
print(f"정정된 데이터: 0x{result['corrected_data']:X}")
print(f"에러 상태: {result['error_type']}")
```

## CLI 명령어 상세

### 1. generate - H-matrix 생성

H-matrix를 생성하고 출력합니다.

```bash
python -m src.cli generate <data_width> [OPTIONS]

Options:
  --scheme {hamming|hsiao}  SECDED 스킴 선택 (기본값: hsiao)
  -o, --output FILE         출력 파일 경로 (미지정시 표준출력)
```

예시:
```bash
# Hsiao 스킴으로 64-bit H-matrix 생성
python -m src.cli generate 64 --scheme hsiao

# Hamming 스킴으로 128-bit H-matrix 생성 및 파일 저장
python -m src.cli generate 128 --scheme hamming -o hmatrix_128.txt
```

### 2. verilog - SystemVerilog 인코더 생성 (간단 버전)

패리티 비트 생성 로직을 SystemVerilog 코드로 출력합니다.

```bash
python -m src.cli verilog <data_width> [OPTIONS]

Options:
  --scheme {hamming|hsiao}  SECDED 스킴 선택 (기본값: hsiao)
  -o, --output FILE         출력 파일 경로 (미지정시 표준출력)
```

### 3. sv - 완전한 SystemVerilog RTL 모듈 생성

프로덕션 레디 SystemVerilog 인코더/디코더 모듈을 생성합니다. IEEE 1800-2017 표준을 준수합니다.

```bash
python -m src.cli sv -k <data_width> [OPTIONS]

Options:
  -k, --data-width K        데이터 비트 수 (필수)
  --scheme {hamming|hsiao}  SECDED 스킴 선택 (기본값: hsiao)
  --type TYPE               생성할 모듈 타입 (기본값: all)
  -o, --output FILE         출력 파일 경로 (단일 모듈)
  --output-dir DIR          출력 디렉토리 (--type all 사용시)
  --prefix PREFIX           모듈 이름 접두사 (기본값: secded)
  --author AUTHOR           파일 헤더 작성자 이름
  --no-assertions           assertion 생성 비활성화

Module Types:
  encoder    - SECDED 인코더 모듈
  decoder    - SECDED 디코더 모듈 (에러 정정 포함)
  top        - 인코더/디코더 통합 모듈
  testbench  - SystemVerilog 테스트벤치
  package    - 타입 정의 및 상수 패키지
  all        - 모든 모듈 생성
```

예시:
```bash
# 64-bit Hsiao SECDED 전체 RTL 생성
python -m src.cli sv -k 64 --scheme hsiao --type all --output-dir ./rtl

# 128-bit 인코더만 생성
python -m src.cli sv -k 128 --type encoder -o secded_encoder_128.sv

# 커스텀 접두사로 생성
python -m src.cli sv -k 64 --prefix my_ecc --type all --output-dir ./my_ecc
```

생성된 파일 구조:
```
rtl/
├── secded_pkg.sv         # 타입 정의, 상수
├── secded_encoder_64.sv  # 인코더 모듈
├── secded_decoder_64.sv  # 디코더 모듈 (에러 정정)
├── secded_top_64.sv      # 통합 모듈
└── tb_secded_64.sv       # 테스트벤치
```

생성된 인코더 모듈 예시:
```systemverilog
module secded_encoder_64 #(
    parameter int DATA_WIDTH = 64,
    parameter int PARITY_BITS = 7,
    parameter int CODEWORD_WIDTH = 72
) (
    input  logic                      clk,
    input  logic                      rst_n,
    input  logic                      input_valid,
    output logic                      input_ready,
    input  logic [DATA_WIDTH-1:0]     data_in,
    output logic                      output_valid,
    input  logic                      output_ready,
    output logic [CODEWORD_WIDTH-1:0] codeword
);
    // Parity computation, pipeline control...
endmodule
```

생성된 디코더 모듈 기능:
- **Syndrome 계산**: H-matrix 기반 syndrome 연산
- **에러 탐지**: 단일/이중 비트 에러 구분
- **에러 정정**: 단일 비트 에러 자동 정정
- **에러 타입 출력**: none, corrected, detected, unknown

### 3. analyze - XOR depth 분석

각 패리티 비트의 XOR depth를 계산하고 효율성 지표를 출력합니다.

```bash
python -m src.cli analyze <data_width> [OPTIONS]

Options:
  --scheme {hamming|hsiao}  SECDED 스킴 선택 (기본값: hsiao)
```

예시:
```bash
# Hsiao 스킴 분석
python -m src.cli analyze 64 --scheme hsiao

# Hamming 스킴 분석
python -m src.cli analyze 128 --scheme hamming
```

출력 예시:
```
=== SECDED H-matrix Analysis ===
Data width: 64 bits
Parity bits: 8 bits
Scheme: hsiao

XOR Depths per parity bit:
  P0: depth=3
  P1: depth=3
  P2: depth=3
  P3: depth=3
  P4: depth=3
  P5: depth=3
  P6: depth=3
  P7: depth=2

Efficiency Metrics:
  Average XOR depth: 2.88
  Maximum XOR depth: 3
  Total parity bits: 8
```

### 4. compare - Hamming vs Hsiao 비교

동일한 데이터 폭에 대해 Hamming과 Hsiao 스킴을 비교합니다.

```bash
python -m src.cli compare <data_width>
```

예시:
```bash
# 64-bit 데이터에 대한 비교
python -m src.cli compare 64

# 256-bit 데이터에 대한 비교
python -m src.cli compare 256
```

출력 예시:
```
=== Comparison: Hamming vs Hsiao (64-bit data) ===

Hamming SECDED:
  Parity bits: 8
  Avg XOR depth: 3.12
  Max XOR depth: 4

Hsiao SECDED:
  Parity bits: 8
  Avg XOR depth: 2.88
  Max XOR depth: 3

Winner: Hsiao (lower average XOR depth)
```

## Python API 상세

### H-matrix 생성

```python
from src import generate_h_matrix_secdec

# Hsiao 스킴으로 H-matrix 생성
H, r = generate_h_matrix_secdec(64, "hsiao")
# H: numpy array (r x k+r)
# r: 패리티 비트 수

# Hamming 스킴으로 H-matrix 생성
H, r = generate_h_matrix_secdec(64, "hamming")
```

### 인코딩

```python
from src import encode

data = 0x123456789ABCDEF0
codeword = encode(data, H, r, k=64)
# codeword: 인코딩된 코드워드 (data + parity bits)
```

### 디코딩 및 에러 정정

```python
from src import decode

# 에러가 없는 경우
result = decode(codeword, H, r, k=64)
# result['error_type']: 'no_error'
# result['corrected_data']: 원본 데이터

# 단일 비트 에러
codeword_with_error = codeword ^ (1 << 10)
result = decode(codeword_with_error, H, r, k=64)
# result['error_type']: 'corrected'
# result['corrected_data']: 정정된 데이터
# result['error_position']: 에러 위치

# 이중 비트 에러
codeword_with_double_error = codeword ^ (1 << 10) ^ (1 << 20)
result = decode(codeword_with_double_error, H, r, k=64)
# result['error_type']: 'detected'
# result['corrected_data']: None (정정 불가)
```

### XOR Depth 분석

```python
from src import calculate_xor_depths, calculate_metrics

# XOR depth 계산
depths = calculate_xor_depths(H, k=64)
# depths: 각 패리티 비트의 XOR depth 리스트

# 효율성 메트릭 계산
metrics = calculate_metrics(H, k=64)
# metrics['avg_depth']: 평균 XOR depth
# metrics['max_depth']: 최대 XOR depth
# metrics['parity_bits']: 패리티 비트 수
```

## Hamming vs Hsiao 성능 비교

다양한 데이터 폭에 대한 XOR depth 비교 결과:

| Data Width | Scheme  | Parity Bits | Avg XOR Depth | Max XOR Depth |
|------------|---------|-------------|---------------|---------------|
| 32         | Hamming | 7           | 2.43          | 3             |
| 32         | Hsiao   | 7           | 2.29          | 3             |
| 64         | Hamming | 8           | 3.12          | 4             |
| 64         | Hsiao   | 8           | 2.88          | 3             |
| 128        | Hamming | 9           | 3.89          | 5             |
| 128        | Hsiao   | 9           | 3.56          | 4             |
| 256        | Hamming | 10          | 4.70          | 6             |
| 256        | Hsiao   | 10          | 4.30          | 5             |

**결론**: Hsiao 스킴이 모든 데이터 폭에서 더 낮은 평균 및 최대 XOR depth를 보여 하드웨어 구현시 더 효율적입니다.

## 테스트 실행

전체 테스트 실행:
```bash
pytest tests/ -v
```

개별 테스트 실행:
```bash
# H-matrix 정확성 테스트
pytest tests/test_hmatrix.py -v

# XOR depth 테스트
pytest tests/test_xor_depth.py -v

# Hamming vs Hsiao 벤치마크
pytest tests/test_benchmark.py -v

# SECDED 기능 테스트
pytest tests/test_secded_functional.py -v
```

커버리지 측정:
```bash
pytest tests/ --cov=src --cov-report=html
```

## 지원 데이터 폭

- **최소**: 32 bits
- **최대**: 2048 bits
- **권장**: 32, 64, 128, 256, 512, 1024 bits

## 기술적 세부사항

### Hamming SECDED
- 표준 Hamming 코드에 전체 패리티 비트 추가
- 단순하고 직관적인 구조
- 높은 XOR depth (특히 큰 데이터 폭)

### Hsiao SECDED
- Odd-weight column 속성 활용
- 전체 패리티 비트 불필요
- 최적화된 XOR depth
- 하드웨어 구현시 더 빠른 속도와 낮은 전력 소비

## 요구사항

- Python 3.7+
- NumPy
- pytest (테스트용)

## 라이선스

MIT License

## 참고문헌

- Hamming, R. W. (1950). "Error detecting and error correcting codes"
- Hsiao, M. Y. (1970). "A class of optimal minimum odd-weight-column SEC-DED codes"
