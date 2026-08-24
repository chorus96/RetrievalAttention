# `library/retroinfer/retroinfer_kernels/src/batch_gemm_softmax.h` 분석

## 개요
`batch_gemm_softmax.cu`가 사용하는 **CUTLASS 기반 배치 GEMM+Softmax 커널 타입 정의 헤더**입니다(NVIDIA CUTLASS 예제에서 파생). GEMM(Q·Cᵀ) 결과에 대해 **부분 축약(partial reduction) → 최종 Softmax**를 GPU에서 융합 실행하도록, 커널 조립과 실행 로직을 캡슐화한 `BatchGemmSoftmax` 클래스를 제공합니다.

## 주요 구성 요소
| 구성 | 역할 |
|---|---|
| `cutlass::kernel::ApplySoftmax` | GEMM 출력 D에 대해 `exp(D - N)` 및 부분 합(Sum) 계산 커널 (Softmax 2단계 중 적용 단계) |
| `cutlass::BatchGemmSoftmax` | 전체 오케스트레이터 클래스: DefaultGemm + EpilogueVisitor + 최종 축약을 조합 |
| `DefaultGemmKernel` | CUTLASS 기본 GEMM 커널 구성 |
| `BatchGemmWithEpilogueVisitor` | GEMM에 softmax 부분 축약 epilogue를 끼워넣은 커널(별도 헤더) |
| `ApplySoftmaxFinalReduction` | 블록 간 부분합을 합쳐 최종 Softmax 정규화 |

## 연산 파이프라인 (수식)
1. **GEMM + epilogue**: `D[m,n] = alpha·(A·Bᵀ)`, 동시에 행별 max `N[m,0]` 추적.
2. **부분 축약**: `Sum[m,n'] = Σ_n exp(D[m,n] − N[m,0])` (블록 단위 부분합).
3. **최종 축약(FinalReduction)**: 블록 부분합을 합산해 정규화 상수 확정.
4. **ApplySoftmax**: `Softmax[m,n] = exp(D[m,n] − N[m,0]) / Sum[m]`.

## 블록 다이어그램
```mermaid
flowchart TD
    A["A(query)"] --> G[GEMM + EpilogueVisitor<br/>D = alpha·A·Bᵀ, max N 추적]
    B["B(centroids)"] --> G
    G --> PR["부분 축약<br/>Sum = Σ exp(D−N)"]
    PR --> FR[ApplySoftmaxFinalReduction<br/>블록 부분합 병합]
    FR --> AS[ApplySoftmax<br/>exp(D−N)/Sum]
    AS --> S[Softmax 출력]
```

## 의존성 · 주의
- CUTLASS 헤더(`cutlass/epilogue/...`, `cutlass/reduction/...`)와 `batch_gemm_with_epilogue_visitor.h`에 의존.
- `batch_gemm_softmax.cu`에서 `ArchTag=Sm80`, `InstructionShape<16,8,16>`, `OpClassTensorOp`로 인스턴스화 → **Ampere Tensor Core 전용**.
- 2단계 온라인 소프트맥스(부분합 → 최종 축약)로 수치 안정성과 큰 n(클러스터 수) 처리를 확보합니다.
