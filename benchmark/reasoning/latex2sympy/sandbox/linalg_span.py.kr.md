# `benchmark/reasoning/latex2sympy/sandbox/linalg_span.py` 분석

## 개요
**열벡터/부분공간(span) 관련 LaTeX 파싱을 실험하는 개발용 스크립트**입니다. `\begin{pmatrix}...\end{pmatrix}` 형태의 벡터를 변환해 결과를 확인합니다.

## 블록 다이어그램
```mermaid
flowchart LR
    L[LaTeX 열벡터] --> P[process_sympy]
    P --> S[SymPy Matrix]
    S --> Print[출력]
```

## 주의
개발용 스크래치. `sympy`에 의존, GPU 무관.
