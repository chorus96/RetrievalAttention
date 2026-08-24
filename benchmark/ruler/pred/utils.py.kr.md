# `benchmark/ruler/pred/utils.py` 분석

## 개요
RULER 예측 단계에서 쓰는 **jsonl 로딩 유틸리티**입니다. 매우 작은 헬퍼 모듈입니다.

## 함수
| 함수 | 역할 |
|---|---|
| `iter_jsonl(fname, cnt)` | jsonl을 한 줄씩 파싱해 제너레이터로 yield(옵션으로 최대 `cnt`행) |
| `load_data(fname)` | `iter_jsonl` 결과를 리스트로 반환 |

## 블록 다이어그램
```mermaid
flowchart LR
    F[.jsonl 파일] --> I[iter_jsonl<br/>줄 단위 파싱]
    I --> L[load_data<br/>리스트 반환]
    L --> U[call_api.py에서 사용]
```

## 주의
GPU/외부 의존성 없음. `call_api.py`의 데이터 로딩에 사용됩니다.
