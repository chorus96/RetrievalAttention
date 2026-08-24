# `benchmark/reasoning/rm_maj_eval.py` 분석

## 개요
다수결(majority voting) 및 보상 모델(reward model) 기반 **집계 채점 스크립트**입니다. 한 문제에 대한 여러 샘플 예측을 그룹화해 다수결 답을 뽑고 정확도를 계산합니다(self-consistency 평가).

## 주요 함수
| 함수 | 역할 |
|---|---|
| `math_equal_timeout(pred, gt)` | 타임아웃 보호된 동치 판정 |
| `group_pred(preds, ...)` | 예측들을 정규화·그룹화해 다수결(majority) 답 선정 |
| (집계 로직) | 다수결/보상 가중 정확도 산출 |

## 블록 다이어그램
```mermaid
flowchart TD
    P[문제당 다중 예측] --> N[strip_string 정규화]
    N --> G[group_pred<br/>Counter 다수결]
    G --> Maj[majority 답]
    Maj --> E[math_equal로 정답 대조]
    E --> Acc[maj@k 정확도]
```

## 의존성 · 주의
- `grader.math_equal`, `parser.strip_string`, `utils.load_jsonl`에 의존. GPU 불필요.
- `n_sampling>1`로 생성된 결과에 대한 사후 집계 분석용입니다.
