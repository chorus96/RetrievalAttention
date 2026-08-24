# RetroInfer

[RetroInfer](https://arxiv.org/pdf/2505.02922)는 긴 컨텍스트 LLM 추론을 가속하기 위해, GPU–CPU 협력 실행(co-execution) 구조 안에서 KV 캐시를 벡터 저장소(vector storage)로 재해석하는 새로운 시스템입니다. 어텐션 메커니즘 고유의 희소성(sparsity)을 활용하며, KV 캐시에서 중요한 토큰을 효율적이고 정확하게 검색할 수 있게 해주는 **A**ttention-a**W**are **VE**ctor index(*wave index*, 어텐션 인식 벡터 인덱스)를 도입합니다. 이를 보완하는 것이 *wave buffer*로, KV 캐시 배치를 조율하고 GPU와 CPU에 걸친 연산과 데이터 전송을 겹쳐(overlap) 높은 처리량을 유지합니다. RetroInfer의 핵심 아이디어는 다음과 같습니다:
- 어텐션을 `steady`, `retrieval`, `estimation`의 세 가지 존(zone)으로 나누어, *정확도가 보장되는 어텐션 추정(accuracy-bounded attention estimation)* 으로 동적 희소성을 처리합니다.
- 어텐션의 거친(coarse-grained) 공간적 지역성(spatial locality)을 활용하여, 낮은 오버헤드로 인덱스를 구축하고 갱신하는 경량 *분할 클러스터링(segmented clustering)* 알고리즘을 설계했습니다.
- 빠른 GPU–CPU 데이터 이동을 지원하고 높은 처리량을 유지하기 위한 고도로 최적화된 CUDA 커널을 제공합니다.

<div align="center">
  <img src="asserts/RetroInfer.png" width="500"/>
  <p><em>RetroInfer 아키텍처.</em></p>
</div>

## :zap: 시작하기

### 환경 설정
필요한 의존성 패키지는 `CUDA 12.4`에 의존합니다. 시스템에 `CUDA 12.4`가 설치되어 있지 않다면 도커 이미지 `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`를 사용할 수 있습니다.

코드는 `Python 3.10.16`에서 테스트되었으며, Python 환경 관리를 위해 `conda` 사용을 권장합니다:
```bash
# miniconda가 없다면 먼저 설치한 뒤, 새로운 conda 환경을 생성합니다:
conda create -n retroinfer python=3.10 -y
conda activate retroinfer 

# conda 패키지 설치
conda install -y mkl
conda install -c conda-forge libstdcxx-ng -y

# `pip install .`로 커널을 설치할 때 `DEPRECATION warning`을 해결하려면 pip를 <=25.0으로 다운그레이드해야 할 수 있습니다
python -m pip install pip==25.0

# python 패키지 설치
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
pip install flashinfer-python==0.2.4 -i https://flashinfer.ai/whl/cu124/torch2.5/
pip install git+https://github.com/Starmys/flash-attention.git@weighted
```

### 커널 설치
```bash
cd library/
git clone https://github.com/NVIDIA/cutlass.git
cd retroinfer && pip install . && cd ..

# MInference를 사용하려면 다음 패키지를 설치하세요:
pip install minference==0.1.6.0

# XAttention을 사용하려면 다음 커널을 설치하세요:
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git 
cd Block-Sparse-Attention && git checkout 0e2478b0a4d9858cf0910f78a8aaf4fba751de69 && export MAX_JOBS=8 && python setup.py install && cd ..

# 루트 디렉터리로 돌아갑니다
cd ..
```

### 간단한 테스트
환경이 올바르게 설정되었는지 확인할 수 있는 간단한 데모를 제공합니다. 이 데모는 [RULER](https://github.com/NVIDIA/RULER)의 네 가지 서로 다른 컨텍스트에서 실행되며, 각각 약 120,000개의 토큰을 포함합니다. 다음 명령으로 데모를 실행할 수 있습니다:
```bash
python -u simple_test.py --batch_size 4
```
이 명령은 [Llama-3-8B-1048K](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k) 모델에서 RetroInfer를 실행합니다. 이 데모를 실행하려면 약 35GB의 GPU 메모리와 70GB의 CPU 메모리가 필요합니다. out-of-memory 오류가 발생하면 batch size를 줄이는 것을 고려하세요.

다음 형식의 `json` 파일을 제공하여 입력 컨텍스트를 직접 지정할 수도 있습니다:
```
[
    {"input": str, "outputs": str}, 
    {"input": str, "outputs": str},
    ...
]
``` 
그런 다음 `--data_path` 인자로 파일 경로를 전달합니다:
```bash
python -u simple_test.py --data_path <your_json_file_path>
```

실행을 사용자화할 수 있는 여러 옵션이 있습니다:
- `--gpu_only`를 설정하면 모든 KV 캐시를 GPU 메모리에 유지하는 GPU 전용 버전의 RetroInfer를 실행합니다.
- `--use_cuda_graph`를 설정하면 CUDA 그래프를 활성화하여 커널 실행 오버헤드를 줄이고 처리량을 향상시킬 수 있습니다.
- `--do_sample`을 설정하면 생성 시 샘플링을 활성화합니다.
- `--prefill_method`로 prefill 방법을 지정합니다. 현재 `full`(Full attention, 기본값), `xattn`([XAttention](https://arxiv.org/pdf/2503.16428)), `minfer`([MInference](https://arxiv.org/pdf/2407.02490))를 지원합니다.

### API
RetroInfer를 사용할 수 있는 간단한 API를 제공합니다. 사용 예시는 다음과 같습니다:
```python
from model_hub import load_model, load_tokenizer
from config import generate_config

# 토크나이저와 모델 로드
tokenizer = load_tokenizer(model_name)
llm = load_model(model_name, max_seq_len, dtype, device, tokenizer)

# RetroInfer 설정 로드
attn_config = generate_config(
    model_name, input_seq_len, "RetroInfer", 
    retrieval_budget, estimation_budget, cache_ratio,
    use_cuda_graph=False, gpu_only=False
)

# 출력 생성
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
input_ids = inputs.input_ids
attention_masks = inputs.attention_mask
out = llm.generate(
    attention_type=attn_type,
    inputs_ids=input_ids.to(llm.layers[0].device),
    attention_masks=attention_masks,
    max_new_length=gen_len, 
    attn_config=attn_config
)
result = tokenizer.batch_decode(out, skip_special_tokens=True)
```

## :dart: 벤치마크 실행

> [!IMPORTANT]
> 벤치마크를 실행하기 전에 `CUDA_VISIBLE_DEVICES`를 설정해야 할 수 있습니다. 저희 코드는 모델을 사용 가능한 모든 GPU에 자동으로 분할하기 때문입니다. 예를 들어 A100 80GB로 평가할 때, 7B/8B 모델은 GPU 카드 한 장만 필요하지만 72B 모델은 최소 3장의 GPU 카드가 필요합니다.

> [!NOTE]
> 벤치마크 결과는 하드웨어 관련 무작위성으로 인해 논문에 보고된 수치와 약간 다를 수 있습니다.

### [RULER](https://github.com/NVIDIA/RULER)
RULER 벤치마크에서 모델 정확도를 평가하려면 먼저 벤치마크 데이터셋을 내려받아야 합니다:
```bash
cd benchmark/ruler
cd data/synthetic/json/ && python -u download_paulgraham_essay.py && bash download_qa_dataset.sh && cd ../../../
```
그런 다음 [ruler_run.sh](benchmark/ruler/ruler_run.sh)를 실행하여 평가할 수 있습니다. 예를 들어, 128K 컨텍스트 길이에서 RULER의 variable tracing 작업(`vt`)으로 RetroInfer를 평가하려면 다음 명령을 사용합니다:
```bash
bash ruler_run.sh llama-3-8b-1048k full RetroInfer 131072 vt bf16 0.018 0.232
```
평가 스크립트의 입력 매개변수는 순서대로 다음과 같습니다:
- `model name`: 지원되는 모델에는 `llama-3.1-8b`, `llama-3-8b-1048k`, `qwen2.5-7b`, `qwen2.5-72b`가 있습니다;
- `prefill method`: 지원되는 prefill 방법에는 `full`, `xattn`, `minfer`가 있습니다;
- `attention type`: `RetroInfer` 또는 `Full_Flash_Attn`;
- `input context length`: 입력 컨텍스트 길이;
- `evaluate task name`: 지원되는 작업에는 `niah_single_1`, `niah_single_2`, `niah_single_3`, `niah_multikey_1`, `niah_multikey_2`, `niah_multikey_3`, `niah_multivalue`, `niah_multiquery`, `vt`, `cwe`, `fwe`, `qa_1`, `qa_2`가 있습니다;
- `model data type`: 지원되는 데이터 타입에는 `bf16`과 `fp16`이 있습니다;
- `retrieval budget ratio`: 입력 컨텍스트의 전체 토큰 수 대비 KV 캐시에서 검색할 토큰 수의 비율;
- `attention estimate ratio`: 전체 클러스터 수 대비 어텐션 메커니즘에서 추정할 클러스터 수의 비율.

### [LongBench](https://github.com/THUDM/LongBench)
LongBench 벤치마크에서 RetroInfer의 모델 정확도를 평가하려면 다음 명령을 사용할 수 있습니다:
```bash
cd benchmark/longbench
bash longbench_run.sh llama-3-8b-1048k RetroInfer 0.018 0.232 bf16 SQA
```
평가 스크립트의 입력 매개변수는 순서대로 다음과 같습니다:
- `model name`: 지원되는 모델에는 `llama-3.1-8b`, `llama-3-8b-1048k`, `qwen2.5-7b`, `qwen2.5-72b`가 있습니다;
- `attention type`: `RetroInfer` 또는 `Full_Flash_Attn`;
- `retrieval budget ratio`: 입력 컨텍스트의 전체 토큰 수 대비 KV 캐시에서 검색할 토큰 수의 비율;
- `attention estimate ratio`: 전체 클러스터 수 대비 어텐션 메커니즘에서 추정할 클러스터 수의 비율;
- `model data type`: 지원되는 데이터 타입에는 `bf16`과 `fp16`이 있습니다;
- `sub categories`: 지원되는 범주에는 `SQA`(단일 문서 QA), `MQA`(다중 문서 QA), `SUM`(요약), `FSL`(few-shot 학습), `ST`(합성 작업), `CC`(코드 완성)가 있습니다.

### [추론 벤치마크(Reasoning Benchmark)](https://github.com/QwenLM/Qwen2.5-Math)
긴 추론 작업에서 모델 정확도를 평가하려면 먼저 의존성을 설치해야 합니다:
```bash
cd benchmark/reasoning/latex2sympy && pip install -e . && cd ..
pip install -r requirements.txt 
```

그런 다음 다음 명령을 사용하여 기본 설정으로 추론 작업에서 RetroInfer의 모델 정확도를 평가할 수 있습니다:
```bash
bash eval.sh deepseek-ai/DeepSeek-R1-Distill-Llama-8B RetroInfer aime24 0 -1
```
평가 스크립트의 입력 매개변수는 순서대로 다음과 같습니다:
- `model_name_or_path`: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`와 `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`를 지원합니다;
- `data_names`: `aime24`와 `gpqa`를 지원합니다;
- `attention type`: `RetroInfer` 또는 `Full_Flash_Attn`;
- `eval start index`: 평가 샘플의 시작 인덱스;
- `eval number`: 평가할 샘플 수. `-1`로 설정하면 모든 샘플을 평가한다는 의미입니다.

해당 python 파일 [`math_eval.py`](./benchmark/reasoning/math_eval.py)의 다른 주요 매개변수는 다음과 같습니다:
- 샘플링 매개변수: `temperature`, `top_p`, `top_k`, `do_sample`. 기본값은 각각 `0.6`, `0.95`, `20`, `True`입니다;
- 샘플링 횟수: `n_sampling`. 이 매개변수를 k로 설정하면 시스템은 k번의 독립적인 샘플링 실행을 수행하고 그에 해당하는 pass@k 결과를 평가합니다.

## :bar_chart: 처리량 결과 재현
논문에 보고된 처리량 결과를 재현할 수 있는 스크립트를 제공합니다. 이 실험들은 4개의 NUMA 노드를 갖춘 [Azure 가상 머신](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ndma100v4-series?tabs=sizebasic)에서 수행되었습니다. 각 NUMA 노드는 24개의 CPU 코어, 475GB의 CPU 메모리, 그리고 두 개의 80GB A100 GPU를 갖추고 있습니다.
```bash
# 먼저 numactl 패키지를 설치합니다
sudo apt install numactl -y

# 스크립트 실행
cd throughput_eval
bash run.sh
```

## :clipboard: 새로운 희소성(Sparsity) 방법 추가
이 저장소는 사용자가 새로운 희소성 기반 어텐션 방법을 쉽게 통합할 수 있는 유연한 추론 프레임워크를 제공합니다. 새로운 희소성 방법을 추가하려면 다음 단계를 따르세요:
1. `cache_hub/` 디렉터리에 KV 캐시 관리 로직을 추가합니다.
2. `attn_hub/` 디렉터리에 어텐션 연산 로직을 추가합니다.
3. [config.py](./config/config.py)를 갱신하여 새로운 방법에 대한 설정 옵션을 포함합니다.
4. [llama.py](./model_hub/llama.py)와 [qwen.py](./model_hub/qwen.py)의 `init_kv_cache()`, `decode_attention()`, `parameter_move()` 함수를 갱신하여 새로운 방법을 통합합니다.
5. 이제 스크립트 실행 시 `--attn_type` 인자에 지정하여 새로운 희소성 방법을 시도할 수 있습니다.

## :bulb: 참고문헌(Reference)
이 프로젝트가 도움이 되었다면, 저희 논문을 인용해 주세요:
```bibtex
@article{chen2026retroinfer,
  title={RetroInfer: A Vector Storage Engine for Scalable Long-Context LLM Inference},
  author={Chen, Yaoqi and Zhang, Jinkai and Lu, Baotong and Zhang, Qianxi and Zhang, Chengruidong and Liu, Jing and Luo, Jingjia and Liu, Di and Jiang, Huiqiang and Chen, Qi and others},
  journal={Proceedings of the VLDB Endowment},
  volume={19},
  number={5},
  pages={1016--1031},
  year={2026},
  publisher={VLDB Endowment}
}

@misc{liu2024retrievalattentionacceleratinglongcontextllm,
      title={RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval}, 
      author={Di Liu and Meng Chen and Baotong Lu and Huiqiang Jiang and Zhenhua Han and Qianxi Zhang and Qi Chen and Chengruidong Zhang and Bailu Ding and Kai Zhang and Chen Chen and Fan Yang and Yuqing Yang and Lili Qiu},
      year={2024},
      eprint={2409.10516},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.10516}, 
}
```

## 기여(Contributing)
이 프로젝트는 기여와 제안을 환영합니다. 대부분의 기여는 여러분이 기여물을 사용할 권리를 저희에게
부여할 권리가 있고 실제로 부여한다는 것을 선언하는 기여자 라이선스 계약(CLA, Contributor License
Agreement)에 동의할 것을 요구합니다. 자세한 내용은 https://cla.opensource.microsoft.com 을 방문하세요.

풀 리퀘스트를 제출하면, CLA 봇이 여러분이 CLA를 제공해야 하는지 자동으로 판단하고 PR에 적절히
표시(예: 상태 확인, 코멘트)합니다. 봇이 제공하는 지침을 따르기만 하면 됩니다. 저희 CLA를 사용하는
모든 저장소에 걸쳐 이 작업은 한 번만 하면 됩니다.

이 프로젝트는 [Microsoft 오픈 소스 행동 강령(Microsoft Open Source Code of Conduct)](https://opensource.microsoft.com/codeofconduct/)을 채택했습니다.
자세한 내용은 [행동 강령 FAQ](https://opensource.microsoft.com/codeofconduct/faq/)를 참고하거나
추가 질문이나 의견이 있으면 [opencode@microsoft.com](mailto:opencode@microsoft.com)으로 문의하세요.

## 상표(Trademarks)
이 프로젝트는 프로젝트, 제품, 서비스에 대한 상표나 로고를 포함할 수 있습니다. Microsoft
상표나 로고의 승인된 사용은
[Microsoft의 상표 및 브랜드 가이드라인](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general)을 따라야 합니다.
이 프로젝트의 수정된 버전에서 Microsoft 상표나 로고를 사용할 때 혼동을 일으키거나 Microsoft의 후원을 암시해서는 안 됩니다.
제3자 상표나 로고의 사용은 해당 제3자의 정책을 따릅니다.
