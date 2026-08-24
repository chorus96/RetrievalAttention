# RetroInfer 세미나 자료 (Seminar Materials)

RetroInfer 시스템에 대한 **기술 심화 세미나 발표 자료**입니다. 연구자·개발자를 대상으로, 저장소 코드 분석을 바탕으로 구성했습니다.

## 파일

| 파일 | 설명 |
|---|---|
| `RetroInfer_세미나.pptx` | PowerPoint 발표 자료 (12슬라이드, 한국어) |
| `build_deck.js` | 위 PPTX를 생성하는 `pptxgenjs` 스크립트 (재현·수정용) |

## 슬라이드 구성 (12장)

1. 타이틀 — KV 캐시를 벡터 저장 시스템으로
2. 문제 정의 — 긴 컨텍스트 디코딩의 병목
3. 핵심 통찰 — 어텐션 희소성 → KV 캐시를 ANN 벡터 DB로
4. 시스템 아키텍처 — Wave Index + Wave Buffer
5. 정확도 보장 3-Zone 어텐션 (steady / retrieval / estimation)
6. Wave Index — 어텐션 인식 벡터 인덱스 (segmented k-means)
7. Wave Buffer — GPU–CPU 협력 실행 (LRU, overlap)
8. 디코딩 한 스텝의 실행 흐름 (`sparse_attention`)
9. 고성능 CUDA / C++ 커널
10. 성능 — 정확도 유지, 처리량 4.5–10.5×
11. 하드웨어 요구사항과 한계 (Ampere sm_80+)
12. 요약 & 참고문헌

## 재생성 방법

```bash
cd seminar
npm install pptxgenjs      # 최초 1회
node build_deck.js         # RetroInfer_세미나.pptx 생성
```

## 참고

- 슬라이드 10의 처리량 막대(4.5–10.5×)는 논문 보고 범위를 **도식화**한 값입니다. 실제 측정치가 있으면 `build_deck.js`의 `chartData`를 교체하세요.
- 한글 렌더링을 위해 `Malgun Gothic` 폰트를 사용합니다(미설치 환경에서는 시스템 한글 폰트로 대체됨).
- 참고문헌: RetroInfer (VLDB 2026, [arXiv:2505.02922](https://arxiv.org/abs/2505.02922)) · RetrievalAttention ([arXiv:2409.10516](https://arxiv.org/abs/2409.10516))
