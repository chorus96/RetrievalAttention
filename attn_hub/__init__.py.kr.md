# `attn_hub/__init__.py` 분석

## 개요
`attn_hub` 패키지의 **공개 API 관문(facade)** 입니다. 어텐션 연산 함수들을 한곳에서 export 하며, 선택적(optional) prefill 방법은 **지연 import + 실패 시 대체 스텁**으로 처리해 미설치 의존성 때문에 패키지 전체가 깨지지 않도록 합니다.

## export 대상
| 심볼 | 출처 | 필수 여부 |
|---|---|---|
| `full_decode_attn`, `full_prefill_attn` | `full_attn.py` | 필수 (flash-attn) |
| `retroinfer_decode_attn` | `retroinfer_attn.py` | 필수 |
| `prefill_minfer` | `minfer.py` | 선택 (MInference 미설치 시 스텁) |
| `prefill_xattn` | `xattn.py` | 선택 (Block-Sparse-Attention 미설치 시 스텁) |

## 핵심 패턴
`try/except ImportError`로 감싸, 의존성이 없으면 호출 시점에 명확한 `ImportError` 메시지를 던지는 **더미 함수**를 대신 바인딩합니다. 즉, `minfer`/`xattn`을 실제로 쓰지 않는 한 설치가 없어도 import 단계는 통과합니다.

## 블록 다이어그램
```mermaid
flowchart TD
    subgraph attn_hub 패키지
      I[__init__.py]
    end
    I -->|필수| FA[full_attn: full_decode/prefill]
    I -->|필수| RA[retroinfer_attn: retroinfer_decode]
    I -.선택.-> MI{minfer import 성공?}
    I -.선택.-> XA{xattn import 성공?}
    MI -->|예| MIok[prefill_minfer 실제 함수]
    MI -->|아니오| MIstub[ImportError 스텁]
    XA -->|예| XAok[prefill_xattn 실제 함수]
    XA -->|아니오| XAstub[ImportError 스텁]
```

## 주의
소비자(`model_hub/llama.py`, `qwen.py`)는 이 관문을 통해 어텐션 함수를 가져옵니다. 스텁 덕분에 `--prefill_method full` 경로는 minfer/xattn 미설치와 무관하게 동작합니다.
