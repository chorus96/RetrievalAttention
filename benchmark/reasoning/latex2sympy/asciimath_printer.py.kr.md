# `benchmark/reasoning/latex2sympy/asciimath_printer.py` 분석

## 개요
SymPy 표현식을 **AsciiMath 표기 문자열로 출력하는 프린터**입니다. SymPy의 `StrPrinter`를 상속해 극한·적분·합·곱·미분·절댓값·거듭제곱 등의 출력 형식을 AsciiMath 스타일로 오버라이드합니다.

## 오버라이드 메서드
| 메서드 | 출력 예 |
|---|---|
| `_print_Limit` | `lim_(z -> z) e` |
| `_print_Integral` | `int_(a)^(b) e dx` |
| `_print_Sum` / `_print_Product` | `sum_(i = a)^(b) e` / `prod_...` |
| `_print_Derivative` | `d/dx e` |
| `_print_Abs` | `|x|` |
| `_print_Pow` | `sqrt(x)`, `1/x`, `x^(n)` |

## 블록 다이어그램
```mermaid
flowchart LR
    E[SymPy 표현식] --> P[AsciiMathPrinter<br/>StrPrinter 상속]
    P --> A[AsciiMath 문자열]
```

## 주의
- `sympy`에만 의존. GPU 무관.
- 변환 파이프라인의 출력 포맷터로, 핵심 파싱(`latex2sympy2.py`)과는 독립적입니다.
