# `benchmark/reasoning/latex2sympy/gen/PSListener.py` 분석

## 개요
**ANTLR이 자동 생성한 리스너(listener) 베이스 클래스**입니다(약 573줄). 파스 트리의 각 노드에 대해 `enter*`/`exit*` 콜백 훅을 정의하는 스켈레톤으로, 트리 워킹(walk) 시 이벤트 기반 처리를 가능하게 합니다.

## 역할
- 각 문법 규칙(`Relation`, `Expr`, `Frac`, `Func`, `Matrix` 등)마다 빈 `enterX`/`exitX` 메서드 제공.
- 하위 클래스가 필요한 훅만 오버라이드해 트리 순회 중 동작 수행.

## 블록 다이어그램
```mermaid
flowchart LR
    Tree[파스 트리] --> W[ParseTreeWalker]
    W --> L[PSListener<br/>enter*/exit* 훅]
    L --> H[하위 클래스 오버라이드]
```

## 주의
- ANTLR 런타임에 의존. GPU 무관.
- latex2sympy2는 주로 리스너보다 직접 트리 순회(`convert_*`)를 사용하지만, 리스너는 ANTLR 표준 생성물로 함께 포함됩니다. 자동 생성이라 직접 수정하지 않습니다.
