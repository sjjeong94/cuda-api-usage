# 00. 기준과 읽는 법

이 가이드는 **LLM 서빙 프레임워크를 만들 때 어떤 CUDA Driver·Runtime API가 필요한지**를 기능 단위로 정리합니다. 모든 결론은 "기능 X를 만들려면 API Y가 필요하다" 형식이고, 근거는 vLLM, SGLang, TensorRT-LLM, Ollama(llama.cpp·MLX), ExecuTorch의 소스 분석([`docs/evidence/`](../evidence/))입니다. 이 중 세 프레임워크가 의존하는 **PyTorch**도 함께 분석했습니다. PyTorch는 서빙 프레임워크가 아니라 의존 라이브러리이므로, 등급 판단에 쓰는 "프레임워크 수"에는 넣지 않았습니다.

---

## 1. 등급 (Tier)

등급은 **기능**에 매깁니다. API의 등급은 그 API가 쓰이는 기능 중 가장 낮은 등급입니다.

| 등급 | 의미 | 판단 기준 |
|---|---|---|
| **T0 필수** | 없으면 서빙 엔진이 동작하지 않음 | 기능상 필수. 5개 프레임워크가 모두 직접 쓰거나 PyTorch를 통해 씀 |
| **T1 표준 최적화** | 성능 좋은 서빙 엔진이라면 대부분 갖춤 | 3개 이상 프레임워크가 직접 구현했고, 구현 비용이 낮거나 폴백이 단순함 |
| **T2 고급 최적화** | 특정 병목을 크게 줄이지만 설계 비용이 큼 | 특정 시나리오(RL, PD 분리, JIT 등)에서 효과가 분명함. 채택한 프레임워크 수는 참고만 함 |
| **T3 시스템 특화** | 특정 하드웨어나 토폴로지에서만 의미 있음 | NVSwitch, GB200 NVL72, Grace(NVLink-C2C), NVLink 없는 PCIe 구성 등에 의존 |
| AUX | 기능이 아니라 개발·운영 보조 | 프로파일링, 디버깅 |

> 사용한 프레임워크 수는 **근거**일 뿐 기준 자체는 아닙니다. 예를 들어 VMM은 4개 프레임워크가 쓰지만, 메모리 설계 전체를 바꿔야 하므로 T2로 분류했습니다.

## 2. 프로파일

같은 기능이라도 프레임워크가 PyTorch 위에 있는지에 따라 직접 호출해야 하는 API가 달라집니다.

| 프로파일 | 대표 | 프레임워크 대신 맡아 주는 것 | 직접 다루는 범위 |
|---|---|---|---|
| **[PT] PyTorch 기반** | vLLM, SGLang, TensorRT-LLM | 캐싱 할당기, pinned 할당기, 스트림·이벤트 객체, CUDA Graph 캡처 (`torch.cuda.*`) | 자기 커널 launch + PyTorch에 없는 T1~T3 |
| **[SA] 독립형** | llama.cpp, MLX, ExecuTorch | 없음 | T0 전체 + 스트림·이벤트·그래프·할당기 |

각 기능 카드에는 프로파일별로 무엇을 직접 해야 하는지 적었습니다. PyTorch가 기능별로 정확히 무엇을 제공하고 무엇을 남기는지는 [09-pytorch-dependency.md](09-pytorch-dependency.md)에 따로 정리했습니다.

## 3. 기능 카드 형식

`02`~`08`의 각 기능은 아래 형식으로 씁니다.

| 항목 | 내용 |
|---|---|
| **목적** | 서빙에서 어떤 문제를 푸는지 |
| **필수 API** | 기능을 만들기 위한 최소 API 집합 |
| **선택·최적화 API** | 성능을 더 끌어올리는 API |
| **폴백** | 지원하지 않는 환경에서 대신 쓰는 경로 |
| **요구 사항** | GPU 아키텍처, CUDA·드라이버 버전, 시스템 조건 |
| **프로파일** | [PT]·[SA]에서 달라지는 점 |
| **근거** | 실제로 구현한 프레임워크와 위치 |
| **결론** | 이 기능에 대해 내릴 수 있는 판단 |

## 4. 데이터와 생성물

| 파일 | 내용 |
|---|---|
| [`reference/data/api_usage.csv`](../reference/data/api_usage.csv) | 사실 데이터: 프레임워크·구성 요소·API·범위(prod/test/conditional/hip-only). evidence 문서의 API 목록을 그대로 옮김 |
| [`reference/data/api_meta.csv`](../reference/data/api_meta.csv) | 해석 데이터: API별 기능 태그, 역할(primary/alt), 최소 CUDA, 하드웨어, 비고 |
| [`reference/data/features.csv`](../reference/data/features.csv) | 기능 정의: 등급, 가이드 위치, 직접 구현한 프레임워크, 위임한 프레임워크 |
| [`reference/api-catalog.md`](../reference/api-catalog.md) | 위 세 파일로 **생성**한 카탈로그 (`python3 scripts/gen_api_catalog.py`) |

생성 스크립트는 `features.csv`가 "직접 구현했다"고 적은 프레임워크가 실제로 그 기능의 API를 본체 코드에서 쓰는지 검사합니다.

## 5. 분석 범위와 한계

- **범위**: 각 저장소(또는 그 저장소가 빌드해 실행하는 코드)가 **직접** 호출하는 API만 다룹니다. PyTorch, NCCL, cuBLAS, cuDNN, CUTLASS, FlashInfer, Triton 내부 호출은 제외합니다. 따라서 "API를 직접 쓰지 않는다"가 "그 최적화가 없다"는 뜻은 아닙니다. 위임한 경우는 카드의 **프로파일** 항목과 카탈로그의 `위임` 열에 적었습니다.
- **기준 버전**:

  | 프로젝트 | 기준 커밋 |
  |---|---|
  | vLLM | `main` `68088ed` (2026-10-06) |
  | SGLang | `main` `1291f31` (2026-10-06) |
  | TensorRT-LLM | `main` `5cf80d6` (2026-10-06) |
  | Ollama | `main` `8a971df` (2026-10-05), llama.cpp `b11351`, MLX `264c14f` |
  | ExecuTorch | `main` `91b2e90` (2026-10-06), PyTorch `v2.14.0` AOTInductor |
  | PyTorch (의존 라이브러리) | `v2.14.0` `2b3ec34` (2026-08-26) |

- **방법 차이**: TensorRT-LLM과 PyTorch는 cuda-python `8c66b43` 바인딩 선언과 교차 검증했고, 나머지는 grep 결과를 직접 검토했습니다. PyTorch는 이번에 직접 스캔해서 CSV를 만들었고, 다른 다섯 프로젝트는 evidence 문서의 목록을 옮겼습니다. 다섯 프로젝트에 대한 방법론 통일 재집계는 아직 하지 않았습니다.
- **집계 불일치**: vLLM evidence 문서는 Driver 18개, Runtime 38개라고 적었지만, 문서에 나열된 API를 세면 Driver 16개, Runtime 36개(+ 테스트 전용 프로파일러 2개)입니다. CSV는 나열된 목록을 따랐습니다.
- **최소 CUDA 버전**: `api_meta.csv`의 `min_cuda` 값 중 12.8(배치 복사)과 13.4(Logical Endpoint)는 evidence 문서에 나온 값이고, 나머지는 CUDA Toolkit 문서 기준입니다. 빈칸은 확인하지 않은 항목입니다. 실제 대상 툴킷의 헤더로 다시 확인하세요.
- **TensorRT-LLM의 소스 없는 정적 라이브러리**(Git LFS)는 분석하지 못했습니다.
