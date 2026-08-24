# `benchmark/reasoning/latex2sympy/gen/__init__.py` 분석

## 개요
ANTLR로 자동 생성된 파서 모듈들(`PSLexer`, `PSParser`, `PSListener`)을 담는 `gen` 패키지의 초기화 파일입니다. 내용은 비어 있습니다(패키지 표식).

## 블록 다이어그램
```mermaid
flowchart LR
    I[gen/__init__.py] --> G[PSLexer / PSParser / PSListener]
```

## 주의
자동 생성 산출물 디렉터리의 패키지 마커입니다. 직접 수정 대상이 아닙니다(문법 `.g4` 수정 후 재생성).
