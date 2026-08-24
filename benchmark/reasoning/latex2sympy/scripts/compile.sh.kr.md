# `benchmark/reasoning/latex2sympy/scripts/compile.sh` 분석

## 개요
latex2sympy의 **ANTLR 문법을 파서 코드로 컴파일하는 스크립트**입니다. `PS.g4` 문법 정의로부터 `gen/` 디렉터리의 파서·렉서·리스너 Python 파일을 생성합니다.

## 핵심 로직
1. `git rev-parse`로 프로젝트 루트 이동.
2. `java -jar antlr-4.11.1-complete.jar PS.g4 -o gen` 실행.

## 블록 다이어그램
```mermaid
flowchart LR
    G[PS.g4 문법] --> A[ANTLR 4.11.1 jar]
    A --> Gen[gen/ PSLexer·PSParser·PSListener 생성]
```

## 의존성 · 주의
- Java + `antlr-4.11.1-complete.jar` 필요. GPU 무관.
- 문법을 바꿀 때만 재실행하며, 생성된 `gen/*.py`는 직접 수정하지 않습니다(ANTLR 런타임 버전 4.11.1과 일치해야 함).
