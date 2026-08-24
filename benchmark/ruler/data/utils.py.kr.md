# `benchmark/ruler/data/utils.py` 분석

## 개요
RULER 데이터 생성기가 결과를 저장할 때 쓰는 **jsonl 기록 유틸리티**입니다. 매우 작은 헬퍼입니다.

## 함수
| 함수 | 역할 |
|---|---|
| `dump_jsonl(fname, data)` | 리스트의 각 원소를 한 줄 JSON으로 파일에 기록(`ensure_ascii=False`) |

## 블록 다이어그램
```mermaid
flowchart LR
    D[샘플 리스트] --> J[dump_jsonl]
    J --> F[.jsonl 파일]
```

## 주의
GPU/외부 의존성 없음. `niah.py` 등 생성기가 `save_file` 저장에 사용합니다.
