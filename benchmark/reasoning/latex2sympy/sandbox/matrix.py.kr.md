# `benchmark/reasoning/latex2sympy/sandbox/matrix.py` 분석

## 개요
LaTeX **행렬(matrix) 파싱을 실험하는 개발용 스크립트**입니다. `\begin{matrix}...\end{matrix}` 같은 입력을 `process_sympy`로 SymPy Matrix로 변환해 결과를 확인합니다.

## 블록 다이어그램
```mermaid
flowchart LR
    M[LaTeX 행렬] --> P[process_sympy]
    P --> S[SymPy Matrix]
    S --> Print[출력]
```

## 주의
개발용 스크래치. 행렬 변환의 정식 검증은 `tests/linalg_test.py`가 담당합니다.
