# RetroInfer: 책임 있는 AI(Responsible AI) FAQ

## RetroInfer란 무엇인가요?
- RetroInfer는 긴 컨텍스트 대규모 언어 모델(LLM) 추론을 가속하기 위해, GPU–CPU 협력 실행(co-execution) 구조 안에서 KV 캐시를 벡터 저장소(vector storage)로 재해석하는 시스템입니다. 어텐션 메커니즘 고유의 희소성(sparsity)을 활용하며, KV 캐시에서 중요한 토큰을 효율적이고 정확하게 검색할 수 있게 해주는 **A**ttention-a**W**are **VE**ctor index(*wave index*, 어텐션 인식 벡터 인덱스)를 도입합니다. 이를 보완하는 것이 *wave buffer*로, KV 캐시 배치를 조율하고 GPU와 CPU 사이의 연산과 데이터 전송을 겹쳐(overlap) 높은 처리량을 유지합니다. RetroInfer는 정확도 손실 없이 FlashAttention 대비 디코딩 처리량을 4.5배~10.5배 향상시킵니다.

## RetroInfer는 무엇을 할 수 있나요?
- RetroInfer는 긴 컨텍스트 시나리오에서 생성형 LLM의 디코딩 처리량을 모델 정확도에 미치는 영향을 최소화하면서 효과적으로 향상시킬 수 있습니다.

## RetroInfer의 의도된 용도는 무엇인가요?
- RetroInfer는 긴 컨텍스트 시나리오를 효율적으로 관리해야 하는 LLM 배포자와 사용자를 위한 것입니다.

## RetroInfer는 어떻게 평가되었나요? 성능을 측정하는 데 어떤 지표가 사용되었나요?
- RULER, LongBench, Needle in a Haystack를 포함한 최신 긴 컨텍스트 벤치마크와 각각의 평가 지표를 사용하여 RetroInfer를 평가했습니다.
- multi-needle, multi-hop tracing, 다중 문서 QA, 단일 문서 QA, 코드 완성, few-shot 학습 등 다양한 시나리오에 걸쳐 광범위한 테스트를 수행했습니다. 그 결과 정확도에 거의 변화가 없음을 확인했습니다.

## RetroInfer의 한계는 무엇인가요? 사용자는 시스템을 사용할 때 RetroInfer의 한계로 인한 영향을 어떻게 최소화할 수 있나요?
- LLM이 생성하는 잠재적으로 유해하거나, 거짓이거나, 편향된 응답은 RetroInfer를 사용해도 대체로 달라지지 않습니다. 따라서 RetroInfer를 사용하는 것 자체가 이러한 책임 있는 AI 관련 우려를 본질적으로 완화하거나 악화시키지는 않습니다.
- RetroInfer는 연구 및 실험 목적으로 개발되었습니다. 실제 환경에 적용하는 것을 고려하기 전에 추가적인 테스트와 검증이 필요합니다.

## 어떤 운영 요소와 설정이 RetroInfer의 효과적이고 책임 있는 사용을 가능하게 하나요?
- 사용자는 RetroInfer를 사용할 때 검색 예산 비율(retrieval budget ratio)과 어텐션 추정 비율(attention estimate ratio) 같은 매개변수를 조정할 수 있습니다. 한 번 설정되면, RetroInfer는 긴 컨텍스트 시나리오에서 LLM의 응답 생성을 효과적으로 향상시킬 수 있습니다.
