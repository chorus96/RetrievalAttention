# `benchmark/reasoning/latex2sympy/sandbox/linalg_equations.py` 분석

## 개요
**선형대수 방정식/수식 파싱을 실험하는 개발용 스크립트**입니다. 행렬 곱, 삼각함수가 섞인 복잡한 LaTeX 식을 `process_sympy`로 변환해 결과를 확인합니다.

## 블록 다이어그램
```mermaid
flowchart LR
    L[LaTeX 선형대수 식] --> P[process_sympy]
    P --> S[SymPy 표현식]
    S --> Print[출력]
```

## 주의
개발용 스크래치(자동 테스트 아님). 선형대수 정식 검증은 `tests/linalg_test.py`.
