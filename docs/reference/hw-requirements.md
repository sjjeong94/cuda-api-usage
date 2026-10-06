# 기능별 하드웨어·버전 요구 사항

서빙 엔진이 어떤 환경에서 어떤 기능을 켤 수 있는지 정리한 표입니다.

> **출처 구분**: ◆ 표시는 evidence 문서(프레임워크 소스·주석)에서 확인한 값이고, 나머지는 CUDA Toolkit 문서 기준입니다. 대상 툴킷의 헤더(`cuda.h`, `cuda_runtime_api.h`)에서 다시 확인하세요. 원천 데이터는 [`data/api_meta.csv`](data/api_meta.csv)의 `min_cuda`, `arch` 열입니다.

## 1. 요구 사항 표

| 기능 | 등급 | 최소 CUDA | GPU·시스템 조건 | 런타임 확인 방법 | 미지원 시 폴백 |
|---|:---:|---|---|---|---|
| 스트림 순서 메모리 풀 | T1 | 11.2 | `cudaDevAttrMemoryPoolsSupported` | `cudaDeviceGetAttribute` | `cudaMalloc` (MLX), 기본 풀 (ExecuTorch) |
| `cudaGraphInstantiateWithFlags` | T1 | 11.4 | — | — | `cudaGraphInstantiate` |
| `cudaLaunchKernelEx` (PDL·cluster 속성) | T1 | 11.8 | — | — | `<<<>>>` |
| PDL 디바이스 측 동작 | T1 | ◆ llama.cpp는 12.3+(Linux 11.8+)로 빌드 시에만 컴파일 | ◆ SM90+ | compute capability | 일반 실행 |
| 배치 복사 `cu(da)MemcpyBatchAsync` | T1 | ◆ 12.8 | ◆ legacy default stream 불가 (vLLM) | `cuGetProcAddress(…, 12080)`, `dlsym`, 버전 확인 | ◆ `cudaMemcpyAsync` 반복 |
| VMM | T2 | 10.2 | ◆ `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` | `cuDeviceGetAttribute` | ◆ `cudaMalloc` legacy 풀 (llama.cpp) |
| `cuMemRetainAllocationHandle` | T2 | 11.0 | — | — | — |
| Library API (`cuLibraryLoadData`) | T2 | 12.0 | — | — | Module API (`cuModuleLoadData`). Runtime `cudaLibraryLoadData`도 있음 (cuda-python 바인딩은 v12.8.0부터) |
| TMA `cuTensorMapEncodeTiled` | T2 | 12.0 | SM90+ | compute capability | TMA를 쓰지 않는 커널 |
| Thread Block Cluster | T2 | 11.8 | SM90+ | compute capability | cluster 없는 커널 |
| 그래프 조건 노드 | T2 | 12.3 (IF/ELSE·SWITCH는 12.8) | — | — | 그래프를 나누고 호스트에서 분기 |
| Green Context | T2 | 12.4 (`cuGreenCtxStreamCreate`는 12.5, work queue 설정은 드라이버 13.1, Runtime `cudaGreenCtxCreate`는 13.x) | ◆ 드라이버 버전 의존 | ◆ `cuDriverGetVersion`, `cuGetProcAddress` | ◆ `cuCtxFromGreenCtx` + `cuStreamCreate` (SGLang) |
| `cudaGetDriverEntryPointByVersion` | T2 | 12.5 | — | — | `cudaGetDriverEntryPoint` |
| NVLS 멀티캐스트 `cuMulticast*` | T3 | 12.1 | 3세대 이상 NVSwitch (NVLink SHARP, Hopper 세대 HGX부터. HGX A100은 미지원) | `cuDeviceGetAttribute(MULTICAST_SUPPORTED)`, ◆ `cuMulticastGetGranularity` 사전 점검 (SGLang) | NCCL / Custom AllReduce |
| FABRIC 핸들 (MNNVL) | T3 | — | ◆ 멀티노드 NVLink, IMEX 데몬 | `cuDeviceGetAttribute` | POSIX fd 핸들 (노드 내) |
| Logical Endpoint | T3 | ◆ 13.4 | ◆ LE 지원 드라이버, IMEX, NVSwitch fabric | ◆ `cuGetProcAddress` | — |
| 링크 인지 복사 | T3 | — | ◆ NVLink-C2C (Grace) vs PCIe | ◆ `cudaDeviceGetPCIBusId` + NVML | copy engine 배치 복사 |
| Unified memory | T2 | — | integrated GPU, concurrent managed access | `cudaDeviceGetAttribute` | ◆ pinned host 폴백 (MLX) |

## 2. 하드웨어 세대별로 켤 수 있는 기능

| 대상 | 켤 수 있는 기능 |
|---|---|
| **Ampere (SM80) 이하** | T0 전체, CUDA Graph, Occupancy, 메모리 풀, IPC AllReduce, P2P, 배치 복사(드라이버 12.8+), VMM, green context(드라이버 12.4+, 지원 아키텍처는 대상 툴킷 문서에서 확인) |
| **Hopper (SM90) 이상** | 위 + **PDL, TMA, Thread Block Cluster** |
| **NVSwitch 시스템** (HGX H100 이상) | 위 + **NVLS 멀티캐스트**, UserBuffers |
| **GB200 NVL72** | 위 + **MNNVL fabric 메모리**, Logical Endpoint(CUDA 13.4) |
| **Grace 계열** (GH200, GB200) | 위 + **NVLink-C2C 링크 인지 복사** |
| **소비자 PCIe 멀티 GPU** | P2P(지원 시), **PCIe AllReduce** |
| **Integrated GPU** (Jetson 등) | **Unified memory**를 기본 할당 경로로 |

## 3. 확인하지 않은 항목

`api_meta.csv`에서 `min_cuda`가 비어 있는 API는 버전을 확인하지 않았습니다. 266개 중 200개가 비어 있습니다. 대부분은 CUDA 11 이전부터 있던 기본 API이지만, 최근 추가된 것도 섞여 있습니다. 대표적으로 `cuDevSmResourceSplit`(TRT-LLM, cuda-python 바인딩은 v13.2.0부터), `cudaGetKernel`, `cudaLibraryLoadData`(바인딩은 v12.8.0부터), `cudaLaunchHostFunc_v2`입니다. 이 표의 기능별 최소 버전은 비어 있지 않은 값에서 뽑았습니다.
