# LLM 서빙 프레임워크를 위한 CUDA Driver·Runtime API 가이드

LLM 서빙 프레임워크를 만들 때 **어떤 CUDA Driver·Runtime API가 필수이고, 어떤 기능이나 최적화를 원할 때 어떤 API가 추가로 필요한지** 정리한 문서입니다. 결론은 모두 "기능 X를 만들려면 API Y가 필요하다" 형식이고, 근거는 vLLM, SGLang, TensorRT-LLM, Ollama(llama.cpp·MLX), ExecuTorch 다섯 프로젝트와, 그중 세 프로젝트가 의존하는 PyTorch가 실제로 호출하는 API 266개를 분석한 결과입니다.

---

## 핵심 결론

1. **최소 서빙 엔진은 Runtime API만으로 만들 수 있습니다.** 필수(T0) API는 디바이스·메모리·전송·스트림·이벤트·커널 실행·에러 처리의 Runtime API이고, ExecuTorch 런타임은 Driver API를 하나도 쓰지 않습니다. → [01](docs/guide/01-t0-essential.md)
2. **표준 최적화(T1)도 대부분 Runtime API로 됩니다.** CUDA Graph, PDL, occupancy 기반 launch, 스트림 순서 메모리 풀, Custom AllReduce, KV 배치 복사가 여기에 속합니다. T1에 속하는 Driver API는 5개입니다. 그중 Runtime으로 대체할 수 없는 것은 Custom AllReduce의 `cuPointerGetAttribute` 하나입니다. `cuMemcpyBatchAsync`와 `cuMemHostGetDevicePointer`는 Runtime 버전이 있고, `cuLaunchKernelEx`와 `cuOccupancyMaxPotentialBlockSize`는 JIT 커널을 쓸 때만 필요합니다.
3. **Driver API가 꼭 필요한 경우는 세 가지로 모입니다.**
   - **메모리 주소 제어**: VMM으로 주소를 유지한 채 확장·반납·공유 (VMM 풀, Sleep/Wake, 프로세스·노드 간 공유, NVLS, MNNVL)
   - **커널 바이너리 직접 관리**: 미리 빌드한 CUBIN, NVRTC JIT, AOT 생성 코드 로드와 TMA descriptor 생성
   - **하드웨어 자원 분할·연결**: green context(SM 분할), 멀티캐스트, Logical Endpoint, GPU 측 스트림 신호
   → [08](docs/guide/08-driver-vs-runtime.md)
4. **PyTorch 위에 만드는지에 따라 직접 다룰 범위가 달라집니다.** PyTorch 기반(vLLM, SGLang, TRT-LLM)은 할당기·스트림·CUDA Graph 캡처를 PyTorch에 맡기고 T1~T3만 직접 구현합니다. 독립형(llama.cpp, MLX, ExecuTorch)은 T0 전체와 그래프·할당기를 직접 다룹니다. → [00](docs/guide/00-criteria.md#2-프로파일), [09](docs/guide/09-pytorch-dependency.md)
   - PyTorch 자체도 Driver API 50개, Runtime API 99개를 씁니다. 하지만 **PDL, KV 배치 복사, Custom AllReduce, Sleep/Wake는 제공하지 않으므로** PyTorch 기반 엔진도 이 기능들은 직접 구현해야 합니다.
   - PyTorch가 제공하는데도 서빙 프레임워크가 직접 구현한 기능도 있습니다. VMM(expandable segments), green context, NVLS 멀티캐스트(symmetric memory)는 서빙 전용 요구(sleep/wake, PD multiplexing 등)에 맞추려고 자체 구현을 썼습니다.
5. **새 API는 실행 중에 찾아서 쓰세요.** 배치 복사, green context, Logical Endpoint, TMA는 `cuGetProcAddress`나 `cudaGetDriverEntryPoint(ByVersion)`로 심볼을 해석하고, 없으면 폴백합니다. 세 프로젝트가 모두 이 패턴을 씁니다. → [08](docs/guide/08-driver-vs-runtime.md#2-t2-vergate-버전-의존-심볼-해석)

6. **지금 API는 같은 일을 하는 이름이 너무 많습니다.** 266개 중 여섯 코드베이스가 모두 쓰는 것은 13개뿐이고, 113개는 한 곳에서만 씁니다. Driver/Runtime 중복, `Ex`·`WithFlags`·`Async` 변형, 객체별 getter, context·last error 같은 암묵적 상태가 원인입니다. 같은 기능 41개를 **56개 함수**로 표현하는 재설계안을 사고 실험으로 정리했습니다. 이 형태는 서빙 프레임워크의 내부 추상화 계층으로 바로 쓸 수 있습니다. → [재설계안](docs/redesign/cuda-api-redesign.md)

---

## 등급별 기능 지도

| 등급 | 기능 | Driver 필요 | 가이드 |
|---|---|:---:|---|
| **T0 필수** | 디바이스 열거·선택, 하드웨어·버전 조회, 메모리 용량 산정, 디바이스 메모리, Host↔Device 전송·pinned 버퍼, 스트림·이벤트, 커널 실행·smem 설정, 에러 처리 | | [01](docs/guide/01-t0-essential.md) |
| **T1 표준 최적화** | CUDA Graph, 캡처 인지 동작 | | [04](docs/guide/04-execution.md) |
| | PDL, Occupancy 기반 launch 구성 | | [04](docs/guide/04-execution.md) |
| | 스트림 순서 메모리 풀, mapped host (zero-copy) | | [02](docs/guide/02-memory.md) |
| | KV 배치 복사 (CUDA 12.8+) | 선택 | [03](docs/guide/03-kv-cache.md) |
| | Custom AllReduce (IPC) | ● (1개) | [06](docs/guide/06-multi-gpu.md) |
| | Peer access·P2P 복사 | | [06](docs/guide/06-multi-gpu.md) |
| **T2 고급 최적화** | VMM 확장 풀·arena, Sleep/Wake, VMM 공유 핸들 | ● | [02](docs/guide/02-memory.md) |
| | Unified memory | | [02](docs/guide/02-memory.md) |
| | 외부 CUBIN·JIT 커널 로딩 | ● | [05](docs/guide/05-kernel-loading.md) |
| | TMA | ● | [04](docs/guide/04-execution.md) |
| | Thread Block Cluster, cooperative launch, 호스트 콜백 | | [04](docs/guide/04-execution.md) |
| | 그래프 직접 조립·분석 | 선택 | [04](docs/guide/04-execution.md) |
| | 그래프 조건 노드 | | [04](docs/guide/04-execution.md) |
| | Green Context (SM 분할), 스트림 메모리 연산 | ● | [07](docs/guide/07-scheduling-isolation.md) |
| | 스트림 우선순위 | | [07](docs/guide/07-scheduling-isolation.md) |
| | 드라이버만으로 GPU 탐지, 버전 의존 심볼 해석, 컨텍스트 관리 | ● | [08](docs/guide/08-driver-vs-runtime.md) |
| **T3 시스템 특화** | NVLS 멀티캐스트, MNNVL fabric 메모리, Logical Endpoint | ● | [06](docs/guide/06-multi-gpu.md) |
| | PCIe AllReduce | | [06](docs/guide/06-multi-gpu.md) |
| | 링크 인지 복사 전략 (NVLink-C2C vs PCIe) | ● | [03](docs/guide/03-kv-cache.md) |

등급 기준과 기능 카드 형식은 [00-criteria.md](docs/guide/00-criteria.md)에 있습니다.

---

## 서빙 기능에서 API로: 빠른 참조

| 만들고 싶은 것 | 필요한 API | 등급 |
|---|---|:---:|
| KV cache 크기를 남은 메모리에 맞추기 | `cudaMemGetInfo` | T0 |
| 디코딩 launch 오버헤드 줄이기 | [SA] `cudaStreamBeginCapture`/`EndCapture`, `cudaGraphInstantiate`, `cudaGraphExecUpdate`, `cudaGraphLaunch` · [PT] `torch.cuda.CUDAGraph` + `cudaStreamIsCapturing` | T1 |
| 커널 사이의 빈틈 줄이기 (Hopper+) | `cudaLaunchKernelEx` + `cudaGridDependencySynchronize` / `cudaTriggerProgrammaticLaunchCompletion` | T1 |
| 요청별 할당을 동기화 없이 | `cudaMallocAsync`, `cudaMemPoolCreate`, `cudaMemPoolSetAttribute(ReleaseThreshold)` | T1 |
| KV를 host로 오프로드 | `cudaHostRegister`, `cudaMemcpyAsync`, 이벤트 + `cuMemcpyBatchAsync`(12.8+, `cuGetProcAddress`) | T0+T1 |
| 같은 노드 TP AllReduce를 NCCL보다 빠르게 | `cudaIpcGetMemHandle`/`OpenMemHandle`, `cuPointerGetAttribute`, `cudaThreadExchangeStreamCaptureMode` | T1 |
| RL 학습과 GPU 번갈아 쓰기 (Sleep/Wake) | `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, `cuMemUnmap`, `cuMemRelease` + 백업 복사 | T2 |
| KV 영역을 포인터 변화 없이 키우기 | VMM (위와 같음) + `cuMemGetAllocationGranularity` | T2 |
| 프로세스 간 GPU 버퍼 무복사 전달 | `cuMemExportToShareableHandle`, `cuMemImportFromShareableHandle`, (`cuStreamWaitValue32`/`WriteValue32`로 순서) | T2 |
| 모델 설정에 맞춘 JIT 커널 | NVRTC + `cuLibraryLoadData`, `cuLibraryGetKernel`, `cuKernelSetAttribute`, `cuLaunchKernelEx` | T2 |
| Hopper TMA 커널 직접 작성 | `cuTensorMapEncodeTiled` (`cudaGetDriverEntryPointByVersion`으로) | T2 |
| 한 GPU에서 prefill·decode 간섭 줄이기 | `cuDeviceGetDevResource`, `cuDevSmResourceSplitByCount`, `cuDevResourceGenerateDesc`, `cuGreenCtxCreate`, `cuGreenCtxStreamCreate` | T2 |
| NVSwitch에서 AllReduce 가속 | `cuMulticastCreate`, `cuMulticastAddDevice`, `cuMulticastBindMem` + VMM 공유 | T3 |
| GB200 NVL72에서 노드 간 GPU 메모리 매핑 | VMM + `CU_MEM_HANDLE_TYPE_FABRIC` | T3 |

전체 목록은 [API 카탈로그](docs/reference/api-catalog.md)의 3절(기능별 API)에 있습니다.

---

## 근거: 프레임워크별 API 사용량

본체 코드 기준, 중복 제외 ([카탈로그](docs/reference/api-catalog.md)에서 생성).

| 프레임워크 | 프로파일 | Driver | Runtime (호스트) | Runtime (디바이스 측) | Driver를 쓰는 곳 |
|---|---|---:|---:|---:|---|
| vLLM | PT | 16 | 36 | 2 | Sleep mode(VMM), KV 배치 복사, IPC base 주소 |
| SGLang | PT | 46 | 44 | 2 | VMM arena·공유, green context, 그래프 분석, TMA, CuTe DSL 로딩 |
| TensorRT-LLM | PT | 84 | 72 | 2 | CUBIN·JIT 로딩, KV v2, Sleep/Wake, NVLS, MNNVL, Logical Endpoint, green context |
| Ollama — 본체 | — | 8 | 0 | 0 | 드라이버만으로 GPU 탐지 |
| Ollama — llama.cpp | SA | 11 | 48 | 2 | VMM 풀 |
| Ollama — MLX | SA | 10 | 55 | 0 | JIT 커널 로딩·실행, TMA, 그래프 노드 |
| ExecuTorch — 런타임 | SA | 0 | 37 | 0 | 없음 |
| ExecuTorch — AOTI 생성 코드 | SA | 7 (+ 조건부 2) | 19 (+ HIP 전용 1) | 0 | 생성 커널 로딩·실행, TMA |
| **PyTorch** (의존 라이브러리) | — | 50 | 99 | 0 | expandable segments(VMM), symmetric memory·멀티캐스트, jiterator·Inductor 커널 로딩, green context, 컨텍스트 안전성 |

---

## 문서 구성

```
README.md                          ← 이 문서: 결론, 기능 지도, 빠른 참조
docs/
├── guide/                         ← 본문: 기능 카드 ("무엇을 만들려면 무엇이 필요한가")
│   ├── 00-criteria.md             ← 등급 기준, 프로파일, 카드 형식, 범위와 한계
│   ├── 01-t0-essential.md         ← 필수 API (최소 서빙 엔진)
│   ├── 02-memory.md               ← 메모리 풀, zero-copy, VMM, Sleep/Wake, 공유, UVM
│   ├── 03-kv-cache.md             ← KV 오프로드, 배치 복사, 전송 커널, 링크 인지 복사
│   ├── 04-execution.md            ← CUDA Graph, PDL, occupancy, TMA, cluster
│   ├── 05-kernel-loading.md       ← CUBIN, JIT, AOT 커널 로딩
│   ├── 06-multi-gpu.md            ← Custom AllReduce, P2P, NVLS, MNNVL, Logical Endpoint
│   ├── 07-scheduling-isolation.md ← green context, 스트림 우선순위, GPU 측 신호
│   ├── 08-driver-vs-runtime.md    ← Driver가 필요한 경우, 버전 분기, 컨텍스트, 호출 방식
│   └── 09-pytorch-dependency.md   ← PyTorch가 제공하는 것과 [PT] 엔진이 직접 할 일
├── redesign/
│   ├── cuda-api-redesign.md       ← 재설계 사고 실험: 266개 → 56개
│   └── data/api_redesign_map.csv  ←   현재 API → 재설계 API 매핑
├── reference/
│   ├── api-catalog.md             ← API 266개 카탈로그 (생성물)
│   ├── hw-requirements.md         ← 기능별 최소 CUDA·하드웨어
│   └── data/                      ← 원천 데이터 (CSV)
│       ├── api_usage.csv          ←   사실: 프레임워크 × API × 범위
│       ├── api_meta.csv           ←   해석: API별 기능·역할·버전
│       └── features.csv           ←   기능 정의: 등급, 직접 구현·위임 프레임워크
└── evidence/                      ← 프레임워크별 원본 분석 (근거 자료)
    ├── vllm_cuda_api_usage.md
    ├── sglang_cuda_api_usage.md
    ├── tensorrt_llm_cuda_api_usage.md
    ├── ollama_cuda_api_usage.md
    ├── executorch_cuda_api_usage.md
    └── pytorch_cuda_api_usage.md  ← 의존 라이브러리 PyTorch (직접 스캔)
scripts/
├── gen_api_catalog.py             ← CSV → api-catalog.md 생성·검증
├── pytorch/                       ← PyTorch 소스 스캔 → api_usage.csv (재현용)
└── redesign/build_map.py          ← 재설계 매핑 생성·검증
```

**데이터를 고칠 때**: `docs/reference/data/`의 CSV를 고친 뒤 `python3 scripts/gen_api_catalog.py`를 실행하세요. 스크립트는 모든 API에 메타데이터가 있는지, `features.csv`에 적은 프레임워크가 실제로 그 기능의 API를 쓰는지 검사합니다.

## 분석 범위

각 저장소가 **직접** 호출하는 API만 다룹니다. 서빙 프레임워크 분석에서는 PyTorch, NCCL, cuBLAS, CUTLASS, FlashInfer, Triton 내부 호출을 제외했고, PyTorch는 따로 분석했습니다([evidence](docs/evidence/pytorch_cuda_api_usage.md)). 그러므로 API가 목록에 없다고 해서 그 최적화가 없다는 뜻은 아닙니다. 기준 커밋, 방법론 차이, 알려진 집계 불일치는 [00-criteria.md](docs/guide/00-criteria.md#5-분석-범위와-한계)에 있습니다.
