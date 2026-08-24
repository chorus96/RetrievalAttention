# `benchmark/reasoning/latex2sympy/gen/PSLexer.py` 분석

## 개요
**ANTLR이 자동 생성한 렉서(lexer)** 입니다(약 1,446줄). LaTeX 문법 정의(`PS.g4`)로부터 생성되며, 입력 LaTeX 문자열을 토큰 스트림으로 분해합니다. 사람이 직접 편집하지 않는 생성 산출물입니다.

## 역할
- LaTeX 원문 → 토큰(숫자, 명령어 `\frac`·`\sqrt`, 괄호, 연산자, 그리스 문자 등) 시퀀스로 변환.
- 직렬화된 ATN(Augmented Transition Network) 테이블을 내장.

## 블록 다이어그램
```mermaid
flowchart LR
    L[LaTeX 문자열] --> LX[PSLexer<br/>ATN 상태 기계]
    LX --> Tok[토큰 스트림]
    Tok --> P[PSParser로 전달]
```

## 주의
- ANTLR 런타임(4.11.1)에 의존. GPU 무관.
- 문법을 바꾸려면 `PS.g4`를 수정하고 ANTLR로 재생성해야 합니다(이 파일 직접 수정 금지).
