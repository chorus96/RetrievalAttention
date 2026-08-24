# `benchmark/reasoning/latex2sympy/scripts/coverage.sh` 분석

## 개요
latex2sympy의 **테스트 커버리지를 측정하고 HTML 리포트를 생성하는 스크립트**입니다(로컬 개발용).

## 핵심 로직
1. 프로젝트 루트 이동 후 `.env` venv 활성화.
2. `pytest --doctest-modules --cov-report=html --cov-config=.coveragerc --cov=latex2sympy tests` 실행.

## 블록 다이어그램
```mermaid
flowchart LR
    V[venv 활성화] --> P[pytest --cov<br/>doctest 포함]
    P --> H[HTML 커버리지 리포트]
```

## 의존성 · 주의
- `pytest`, `pytest-cov`, `.coveragerc` 필요. GPU 무관.
- CI용 변형은 `coverage-ci.sh`(XML 리포트).
