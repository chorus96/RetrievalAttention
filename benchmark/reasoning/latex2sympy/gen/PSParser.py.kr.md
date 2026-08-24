# `benchmark/reasoning/latex2sympy/gen/PSParser.py` 분석

## 개요
**ANTLR이 자동 생성한 파서(parser)** 입니다(약 6,486줄, 가장 큰 생성 파일). 렉서가 만든 토큰 스트림을 LaTeX 수식 문법에 따라 **파스 트리(parse tree)** 로 구성합니다. `latex2sympy2.py`의 `convert_*` 함수들이 이 트리의 컨텍스트 노드를 순회합니다.

## 역할
- 토큰 스트림 → 문법 규칙별 컨텍스트(예: `RelationContext`, `ExprContext`, `FracContext`, `FuncContext`, `MatrixContext` 등) 트리 생성.
- 각 규칙에 대응하는 `*Context` 클래스와 파싱 메서드를 포함.

## 블록 다이어그램
```mermaid
flowchart LR
    Tok[토큰 스트림] --> P[PSParser<br/>문법 규칙]
    P --> Tree[파스 트리<br/>*Context 노드]
    Tree --> C[latex2sympy2 convert_*]
    C --> S[SymPy 표현식]
```

## 주의
- ANTLR 런타임(4.11.1)에 의존. GPU 무관.
- 자동 생성 산출물이므로 직접 편집하지 않습니다(문법 `PS.g4` 수정 후 재생성).
- `*Context` 클래스 이름이 `latex2sympy2.py`의 변환 함수와 1:1로 대응합니다.
