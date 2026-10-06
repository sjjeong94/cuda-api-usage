# LLM 추론 프레임워크의 CUDA API 사용 현황

vLLM, Ollama, SGLang, TensorRT-LLM, ExecuTorch 다섯 프로젝트가 **CUDA Driver API와 Runtime API를 직접 어떻게 호출하는지** 조사하고, 각 API가 어떤 최적화 기법에 쓰이는지 정리한 문서입니다. 프로젝트별 상세 분석은 [`docs/`](docs/)에 있고, 이 README는 그 내용을 프로젝트 간 비교 관점으로 묶은 요약입니다.

---

## 목차

1. [분석 대상과 방법](#1-분석-대상과-방법)
2. [한눈에 보는 비교](#2-한눈에-보는-비교)
3. [최적화 기법별 교차 비교](#3-최적화-기법별-교차-비교)
4. [프로젝트별 요약](#4-프로젝트별-요약)
5. [주요 관찰](#5-주요-관찰)
6. [상세 문서](#6-상세-문서)

---

## 1. 분석 대상과 방법

### 1.1 기준 버전

| 프로젝트 | 기준 커밋 | 함께 분석한 코드 |
|---|---|---|
| vLLM | `main` `68088ed` (2026-10-06) | |
| Ollama | `main` `8a971df` (2026-10-05) | llama.cpp `b11351` (`631109b`), MLX `264c14f` (Ollama carry patch 적용) |
| SGLang | `main` `1291f31` (2026-10-06) | |
| TensorRT-LLM | `main` `5cf80d6` (2026-10-06) | 함수 목록 검증용 cuda-python `8c66b43` |
| ExecuTorch | `main` `91b2e90` (2026-10-06) | PyTorch `v2.14.0` (`2b3ec34`)의 AOTInductor 코드 생성 템플릿·`aoti_runtime` 헤더 |

### 1.2 방법과 범위

- **방법**: 소스 트리에서 `cuXxx(` / `cudaXxx(` 호출을 grep하고, 주석·타입·enum·로그 문자열·`#if 0` 코드를 제외했습니다. `dlsym`, `cuGetProcAddress`, `cudaGetDriverEntryPoint(ByVersion)`처럼 이름으로 찾아 호출하는 API와 템플릿 인자로 넘기는 함수는 따로 확인해 포함했습니다.
- **범위**: 각 저장소(또는 그 저장소가 빌드해 실행하는 코드)가 **직접** 호출하는 API만 셉니다. PyTorch, NCCL, cuBLAS, cuDNN, CUTLASS, FlashInfer, Triton 등 의존 라이브러리 내부 호출은 제외했습니다.
- **집계 구분**: 본체(런타임) 코드와 테스트·CI·벤치마크 전용 사용을 나눠 셌습니다.
- **주의**: TensorRT-LLM은 `cudaD2Dcpy`처럼 `cuda`로 시작하는 자체 헬퍼가 많아서 cuda-python 바인딩 선언과 교차 검증했습니다. 다른 프로젝트는 grep 결과를 직접 검토했습니다. 방법이 조금씩 달라서 개수는 대략적인 비교로 보는 것이 좋습니다.
- **한계**: TensorRT-LLM의 Git LFS 정적 라이브러리(`internal_cutlass_kernels`, `libTrtLlmGen*.a`)는 소스가 없어 분석하지 못했습니다.

---

## 2. 한눈에 보는 비교

### 2.1 API 개수

| 항목 | vLLM | Ollama | SGLang | TensorRT-LLM | ExecuTorch |
|---|---:|---:|---:|---:|---:|
| Driver API | 18 | 26 (본체 8 + llama.cpp 11 + MLX 10, 중복 제외) | 46 | **84** | 0 (+ AOTI 생성 코드 9) |
| Runtime API (호스트) | 38 | 48 (llama.cpp) / 55 (MLX) | 44 | **72** | 37 (+ AOTI 생성 코드 20) |
| Runtime API (디바이스 측) | 2 | 2 (llama.cpp) | 2 | 2 | 0 |

디바이스 측 API 2개는 모두 PDL용 `cudaGridDependencySynchronize`, `cudaTriggerProgrammaticLaunchCompletion`입니다.

### 2.2 구조와 성격

| 항목 | vLLM | Ollama | SGLang | TensorRT-LLM | ExecuTorch |
|---|---|---|---|---|---|
| CUDA 코드 위치 | `csrc/` + Python ctypes | 본체에는 `.cu` 없음. llama.cpp(ggml-cuda), MLX를 받아 빌드 | AOT 커널 + JIT 커널 + SRT 런타임(cuda-python) | C++ 런타임 + 커널 + Python 백엔드 | 런타임 + AOTI가 생성한 `.so` |
| 커널 공급 방식 | 손으로 작성 + 외부 라이브러리 | 손으로 작성 (llama.cpp) + JIT (MLX) | 손으로 작성 (AOT + JIT) | **미리 빌드한 CUBIN** + 손으로 작성 + JIT (XQA, DeepGEMM) | **AOT 생성** (Inductor → Triton) |
| Driver API 주 용도 | VMM(sleep mode), 배치 복사 | GPU 탐지, VMM 풀, JIT 실행 | VMM 공유, green context, 그래프 분석 | CUBIN 로드, KV v2, NVLS, MNNVL, Logical Endpoint | 생성 커널 로드·실행 |
| VMM 활용 | sleep mode | 메모리 풀 | KV arena, 가중치·feature 공유 | KV v2, sleep, NVLS, MNNVL, DWDP | 없음 |
| 메모리 관리 | PyTorch 캐싱 할당기 + VMM | VMM 풀 / `cudaMallocAsync` | PyTorch + VMM arena | 전용 메모리 풀 + VMM | 전용 풀 (`cudaMallocFromPoolAsync`) |
| 멀티 GPU 메모리 | IPC | P2P, PCIe AllReduce | IPC, VMM 공유 핸들 | IPC, **NVLS 멀티캐스트, MNNVL fabric, Logical Endpoint** | 없음 |
| SM 분할 | 없음 | 없음 | green context (PD multiplexing) | green context (locality domain) | 호출자가 green context 스트림 전달 가능 |
| PDL | 있음 (약 70곳) | 있음 (llama.cpp, 기본 켜짐) | 있음 | **있음 (약 300곳)** | 없음 |
| CUDA Graph | PyTorch에 위임 | 직접 (llama.cpp 캡처 / MLX 노드 조립) | PyTorch 캡처 + Driver API로 분석·dedup | PyTorch에 위임 | 직접 (캡처, method별 opt-in) |
| 런타임 커널 튜닝 | 많음 | 많음 | 많음 | 많음 | 거의 없음 (export 때 완료) |
| KV 오프로드 배치 복사 | `cuMemcpyBatchAsync` (Driver) | 없음 | `cudaMemcpyBatchAsync` (Runtime) | `cuMemcpyBatchAsync` (Driver, KV v2) | 없음 |

---

## 3. 최적화 기법별 교차 비교

### 3.1 기법 × 프로젝트 매트릭스

● 핵심적으로 사용, ○ 부분적으로 사용, 빈칸은 사용하지 않음 (직접 호출 기준)

| 최적화 기법 | vLLM | Ollama | SGLang | TRT-LLM | ExecuTorch |
|---|:---:|:---:|:---:|:---:|:---:|
| VMM 기반 메모리 (예약 + 지연 매핑) | ● | ● (llama.cpp) | ● | ● | |
| Sleep / Wake (주소 유지하며 물리 메모리 해제) | ● | | | ● | |
| VMM 공유 핸들 (프로세스·노드 간) | | | ● | ● | |
| CUDA Graph 직접 관리 | | ● | ○ (분석·dedup) | ○ (보조) | ● |
| PDL | ● | ● (llama.cpp) | ● | ● | |
| TMA (`cuTensorMapEncodeTiled`) | | ○ (MLX) | ● | ● | ○ (생성 커널) |
| Thread Block Cluster | | ○ (MLX) | ○ | ● | |
| CUBIN / JIT 커널 로딩 (Driver) | | ● (MLX) | ○ (CuTe DSL) | ● | ● |
| Green Context (SM 분할) | | | ● | ● | ○ (호출자 제공) |
| KV 계층 오프로드 / 배치 복사 | ● | | ● (HiCache) | ● (KV v2) | |
| Custom AllReduce (IPC) | ● | | ● | ● | |
| NVLS 멀티캐스트 / MNNVL | | | ○ (사전 점검) | ● | |
| 스트림 순서 메모리 풀 | | ● (MLX) | | ● | ● |
| Pinned / mapped host 메모리 | ● | ● | ● | ● | ○ |
| Unified (managed) memory | | ○ | | ○ (EPLB) | |
| 커널 튜닝 (smem, occupancy) | ● | ● | ● | ● | ○ |

### 3.2 VMM (가상 메모리 관리)

가상 주소를 크게 예약해 두고 물리 메모리를 필요할 때 매핑하는 방식입니다. 공통 API는 `cuMemAddressReserve` → `cuMemCreate` → `cuMemMap` → `cuMemSetAccess`, 해제는 `cuMemUnmap` → `cuMemRelease` → `cuMemAddressFree`, 준비 단계에서 `cuMemGetAllocationGranularity`입니다.

| 프로젝트 | 용도 | 추가 API |
|---|---|---|
| vLLM | **Sleep mode** (`CuMemAllocator`). RLHF처럼 학습과 추론을 번갈아 할 때 가상 주소는 두고 물리 메모리만 해제·재할당해서 CUDA Graph와 텐서 포인터를 다시 만들 필요가 없음 | `cuCtxGetCurrent`, `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent`, `cuDeviceGetAttribute` (RDMA·fabric 지원), 백업·복원에 `cudaMemcpy` |
| Ollama (llama.cpp) | **메모리 풀**. 예약 영역 끝에 물리 메모리를 이어 붙여 단편화 없이 풀을 키움. 기본 켜짐(`GGML_CUDA_NO_VMM`으로 끔), 미지원 GPU는 `cudaMalloc` legacy 풀 | `cuDeviceGetAttribute` (`VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`), multi-GPU에서는 `cuMemSetAccess`로 여러 디바이스에 권한 부여 |
| SGLang | **공통 기반** (`cuda_vmm_utils.py`, cuda-python). KV cache VMM arena, DWDP(MoE 전문가 가중치를 NVLink로 가져와 한 주소 공간에 이어 붙임), 멀티모달 feature 무복사 전송, Custom AllReduce v2, graph capture 입력 포인터 매핑 | `cuMemExportToShareableHandle` / `cuMemImportFromShareableHandle` (FABRIC, POSIX fd), `cuMemRetainAllocationHandle`, `cuMemGetAllocationPropertiesFromHandle`, `cuMemGetAddressRange`, `cuMemcpyDtoD` |
| TensorRT-LLM | **KV Cache Manager v2**의 GPU 계층 풀, **Sleep / Wake** (`CUDAVirtualMemoryChunk`), NVLS 멀티캐스트, MNNVL fabric 메모리, DWDP, disaggregated serving 전송 버퍼 | 공유 핸들 API, `cuMemGetAddressRange`, `cuCtxSynchronize`, 백업·복원에 `cuMemcpyDtoH` / `cuMemcpyHtoD(Async)`, 초기화에 `cuMemsetD8Async` |
| ExecuTorch | 사용하지 않음 | |

### 3.3 CUDA Graph

| 프로젝트 | 방식 | 주요 API |
|---|---|---|
| vLLM | PyTorch(`torch.cuda.CUDAGraph`)에 위임. Custom AllReduce만 캡처를 인지 | `cudaStreamIsCapturing`, `cudaThreadExchangeStreamCaptureMode` (캡처 중 버퍼 할당 허용) |
| Ollama (llama.cpp) | **스트림 캡처**. 이전 그래프와 구조가 같으면 exec를 업데이트해 재사용. 기본 켜짐 (`GGML_CUDA_DISABLE_GRAPHS`) | `cudaStreamBeginCapture/EndCapture`, `cudaGraphInstantiate`, `cudaGraphExecUpdate`, `cudaGraphLaunch`, 캡처 비호환 CUB 정렬 회피에 `cudaStreamIsCapturing` |
| Ollama (MLX) | **노드를 직접 조립**. 커널은 노드로 바로 추가하고 cuBLAS/cuDNN 호출만 캡처해 child graph로 삽입. 그래프 캐시 400개 | `cudaGraphCreate`, `cudaGraphAddKernelNode`, `cuGraphAddKernelNode`, `cudaGraphAddChildGraphNode`, `cudaGraphAddDependencies`, `cuGraphKernelNodeSetAttribute` (cluster dim) |
| SGLang | PyTorch로 캡처한 뒤 **Driver API로 그래프를 열어 분석**하고, 구조가 같은 그래프끼리 executable 하나를 공유 (graph exec dedup). Breakable CUDA Graph로 캡처 구간을 나눔 | `cuGraphGetNodes`, `cuGraphGetEdges`, `cuGraphKernelNodeGetParams`, `cuGraphMemcpyNodeGetParams` 등 + `cudaGraphInstantiateWithFlags`, `cudaGraphExecUpdate`, `cudaStreamGetCaptureInfo` |
| TensorRT-LLM | PyTorch에 위임. 본체의 `cudaGraph*`는 테스트에만 있음. 보조 기능만 직접 구현 | `cudaStreamGetCaptureInfo` (breakable graph), `cudaLaunchHostFunc(_v2)` (스트림에서 Python 콜백), `cudaStreamIsCapturing` |
| ExecuTorch | **직접 캡처**. method별 opt-in(기본 꺼짐), warmup 3회 후 1회 캡처, 입력을 static buffer에 복사하고 replay | `cudaStreamBeginCapture/EndCapture`, `cudaGraphInstantiate` (`AutoFreeOnLaunch`), `cudaGraphGetNodes` + `cudaGraphMemAllocNodeGetParams` (남는 할당 추적), `cudaDeviceGraphMemTrim` |

### 3.4 PDL (Programmatic Dependent Launch)

앞 커널이 끝나기 전에 다음 커널을 미리 띄워 launch 지연을 겹치게 만드는 기법입니다(SM90 이상). 호스트는 `cudaLaunchKernelEx`에 `cudaLaunchAttributeProgrammaticStreamSerialization`을 붙여 실행하고, 디바이스에서는 `cudaGridDependencySynchronize()`로 기다리고 `cudaTriggerProgrammaticLaunchCompletion()`으로 신호를 보냅니다.

| 프로젝트 | 규모 | 비고 |
|---|---|---|
| vLLM | 호스트 27곳, 디바이스 33곳 / 37곳 | fused QK-norm+RoPE, NVFP4·FP8 양자화, DSv3 fused A-GEMM 등 |
| Ollama (llama.cpp) | 공통 launcher `ggml_cuda_kernel_launch` | CUDA 12.3 이상 빌드 시 컴파일, 실행 시 기본 켜짐 (`GGML_CUDA_PDL=0`으로 끔) |
| SGLang | AOT·JIT 공통 launcher, TRT-LLM MoE overlay | Custom AllReduce에도 적용 |
| TensorRT-LLM | 호스트 117곳 이상, 디바이스 154곳 / 143곳 | 다섯 프로젝트 중 가장 광범위. `cuLaunchKernelEx`로도 실행 |
| ExecuTorch | 없음 | |

### 3.5 커널 로딩과 실행 (Driver API)

| 프로젝트 | 경로 | API |
|---|---|---|
| Ollama (MLX) | NVRTC로 JIT 컴파일한 fused 커널 | `cuModuleLoadDataEx`, `cuModuleGetFunction`, `cuFuncSetAttribute`, `cuOccupancyMaxPotentialBlockSize`, `cuLaunchKernelEx` |
| SGLang | CuTe DSL로 컴파일한 CUBIN | `cuLibraryLoadData`, `cuLibraryEnumerateKernels`, `cuKernelGetFunction`, `cuFuncGetAttribute`, `cudaLibraryLoadData` |
| TensorRT-LLM | 미리 빌드한 CUBIN (FMHA v2, trtllmGen, 9,322개) | `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel(Ex)`, `cuCtxGetId` (컨텍스트별 캐시) |
| TensorRT-LLM | XQA·DeepGEMM JIT | `cuLibraryLoadData`, `cuLibraryGetKernel`, `cuKernelSetAttribute`, `cuLibraryEnumerateKernels`, `cuKernelGetName` |
| ExecuTorch | AOTI가 `.so`에 내장한 Triton CUBIN | `cuModuleLoadData`, `cuModuleGetFunction`, `cuFuncSetAttribute`, `cuLaunchKernel` (모든 Triton 커널) |

vLLM은 커널을 일반 CUDA 확장으로 빌드하므로 이 경로가 없습니다.

### 3.6 TMA와 Thread Block Cluster (Hopper 이상)

| 프로젝트 | TMA (`cuTensorMapEncodeTiled`) | Cluster |
|---|---|---|
| Ollama (MLX) | FP 양자화 (`fp_quantize.cu`) | `cudaLaunchKernelExC`, `cuLaunchKernelEx`의 cluster dimension 속성 |
| SGLang | KDA prefill, DeepSeek-V4 wo_a fused, HiCache TMA 전송 (`cudaGetDriverEntryPointByVersion`으로 해석) | `cudaOccupancyMaxActiveClusters` |
| TensorRT-LLM | XQA, trtllmGen, Mamba selective scan, mHC, tinygemm2, triattention, DeepGEMM (entry point 경유) | `cuLaunchKernelEx`, `cuOccupancyMaxPotentialClusterSize` |
| ExecuTorch | Triton TMA 커널이 생성될 때만 (AOTI) | |

### 3.7 SM 분할 (Green Context)

| 프로젝트 | 용도 | API |
|---|---|---|
| SGLang | **PD Multiplexing**: 한 GPU의 SM을 prefill용과 decode용으로 나눠 동시에 실행 | `cuDeviceGetDevResource` → `cuDevSmResourceSplitByCount` → `cuDevResourceGenerateDesc` → `cuGreenCtxCreate` → `cuGreenCtxStreamCreate`. 구버전 드라이버는 `cuCtxFromGreenCtx` + `cuCtxPushCurrent` + `cuStreamCreate` 폴백 |
| TensorRT-LLM | **Locality Domain**: GPU를 domain 두 개로 나눠 domain마다 SM 파티션·스트림·할당기를 둠 | `cuDevSmResourceSplit`, `cuGreenCtxCreate`, `cuGreenCtxStreamCreate`, `cuMemAlloc` (모두 `cuGetProcAddress`로 해석) |
| ExecuTorch | `CallerStreamGuard`로 호출자가 green context 스트림을 넘길 수 있음. 자체 호출은 없음 | (스트림 순서는 이벤트 API로 맞춤) |

### 3.8 KV 캐시 계층화와 배치 복사

| 프로젝트 | 기능 | API |
|---|---|---|
| vLLM | KV 블록을 pinned CPU 메모리로 오프로드 | `cuGetProcAddress` → `cuMemcpyBatchAsync` (CUDA 12.8 이상), 폴백 `cudaMemcpyAsync`, `cudaHostRegister`, pinned 여부 판별에 `cudaPointerGetAttributes` |
| SGLang | **HiCache**: GPU KV를 host 메모리와 그 아래 저장소로 확장 | `cudaMemcpyBatchAsync` (`dlsym`), 폴백 `cudaMemcpyAsync`, host pool을 청크 단위로 `cudaHostRegister`, zero-copy 커널 전송에 `cudaHostGetDevicePointer`, TMA 전송 커널 |
| TensorRT-LLM | **KV Cache Manager v2**: prefix 재사용, eviction, GPU/host/disk 계층 이동. 스트림·이벤트까지 Driver API로 작성 | 링크별 전략: NVLink-C2C(Grace)는 SM 복사 커널(`cudaGetKernel` → `cuLaunchKernel`), PCIe는 copy engine 배치 복사(`cuMemcpyBatchAsync`). 링크 종류는 `cudaDeviceGetPCIBusId`로 NVML과 매칭. host 계층 pin은 `cuMemHostRegister` |
| ExecuTorch | **Off-graph KV cache**: KV 저장소를 그래프 밖에서 관리하고 필요할 때 늘림 | 확장 시 `cudaMemcpyAsync`, 다른 스트림과 순서 보장에 `cudaEventRecord` + `cudaStreamWaitEvent` |
| Ollama | 해당 없음 (부분 offload는 레이어 단위, 3.11 참고) | |

### 3.9 멀티 GPU 통신

| 프로젝트 | 기법 | API |
|---|---|---|
| vLLM | **Custom AllReduce**: 작은 텐서는 NCCL 대신 NVLink P2P로 직접 처리 | `cudaIpcGetMemHandle/OpenMemHandle/CloseMemHandle`, base 주소에 `cuPointerGetAttribute` (`RANGE_START_ADDR`), P2P 사전 테스트(ctypes 래퍼), MiniMax Lamport fused allreduce+RMSNorm |
| Ollama (llama.cpp) | Layer split P2P 복사, NVLink 없는 2-GPU용 **PCIe AllReduce** (pinned host 경유) | `cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess`, `cudaMemcpyPeerAsync`, `cudaHostAlloc` (`Mapped \| Portable`), `cudaHostGetDevicePointer` |
| SGLang | vLLM과 같은 구조의 Custom AllReduce + VMM 기반 v2, FlashInfer AllReduce fusion 사전 점검, 멀티모달 IPC all-to-all | `cudaIpc*`, `cuPointerGetAttribute`, `cuMulticastGetGranularity`, `cudaDeviceEnablePeerAccess` |
| TensorRT-LLM | IPC AllReduce, **NVLS 멀티캐스트**(NVLink SHARP), UserBuffers, **MNNVL fabric 메모리**(GB200 NVL72), **Logical Endpoint 기반 MoE AlltoAll**(CUDA 13.4), 비동기 Ulysses all-to-all | `cudaIpc*`, `cuMulticastCreate/AddDevice/BindMem/Unbind`, 공유 핸들 API, `cuLogicalEndpoint*` 9종, `cudaMemcpyBatchAsync` (Ulysses) |
| ExecuTorch | 없음 | |

### 3.10 메모리 할당기

| 프로젝트 | 방식 | API |
|---|---|---|
| vLLM | PyTorch 캐싱 할당기 + VMM (sleep mode) | |
| Ollama (llama.cpp) | VMM 풀 / legacy `cudaMalloc` 풀, opt-in unified memory | `cudaMallocManaged` (`GGML_CUDA_ENABLE_UNIFIED_MEMORY`) |
| Ollama (MLX) | 디바이스 기본 풀에서 스트림 순서 할당, integrated GPU는 unified memory | `cudaDeviceGetDefaultMemPool`, `cudaMallocAsync/FreeAsync`, `cudaMemPoolTrimTo`, `cudaMallocManaged`, `cudaMemAdvise` |
| SGLang | PyTorch + VMM arena를 `torch.cuda.MemPool`로 노출, prefill·decode 그래프가 메모리 풀 하나를 공유 | (VMM API) |
| TensorRT-LLM | 전용 메모리 풀 + 스트림 순서 할당 | `cudaMemPoolCreate`, `cudaMemPoolSetAttribute`, `cudaMallocAsync/FreeAsync`, `cudaMemPoolTrimTo` |
| ExecuTorch | 디바이스별 전용 풀 (`ReleaseThreshold` 설정), 실패 시 기본 풀 | `cudaMemPoolCreate`, `cudaMallocFromPoolAsync`, `cudaFreeAsync`, `cudaMemPoolTrimTo(pool, 0)` |

### 3.11 Host 메모리와 CPU Offload

| 프로젝트 | 기능 | API |
|---|---|---|
| vLLM | **UVA CPU 가중치 오프로드**: CPU 메모리를 GPU 주소로 매핑해 복사 없이 접근. Engram/EC 공유 메모리 pin | `cudaHostAlloc` (`Mapped`), `cudaHostGetDevicePointer`, `cudaHostRegister` |
| Ollama (llama.cpp) | VRAM이 부족할 때 CPU에 남는 레이어의 버퍼를 pinned로 | `cudaMallocHost` (기본 켜짐, `GGML_CUDA_NO_PINNED`), mmap 모델 pin은 `cudaHostRegister` (opt-in, `GGML_CUDA_REGISTER_HOST`) |
| Ollama (MLX) | 가중치 업로드 후 복사가 끝나는 시점에 호스트 버퍼 해제 | `cudaMemcpyAsync` + `cudaLaunchHostFunc` |
| SGLang | 확산 모델 layerwise offload, host 공유 메모리·Engram 테이블 pin | `cudaHostRegister`, `cudaHostUnregister` |
| TensorRT-LLM | KV v2 host 계층, NIXL bounce 버퍼, EPLB용 CPU·GPU 공용 메모리 | `cuMemHostRegister`, `cudaHostAlloc`, `cudaHostGetDevicePointer`, `cudaMallocManaged`, `cudaMemAdvise` |
| ExecuTorch (AOTI) | pageable 가중치를 pinned로 스테이징해 전용 스트림으로 비동기 복사 | `cudaPointerGetAttributes`, `cudaStreamCreateWithFlags` (`NonBlocking`), `cudaMemcpyAsync` |

### 3.12 커널 튜닝 / 하드웨어 적응형 실행

| 목적 | API | 사용 프로젝트 |
|---|---|---|
| SM 수, smem 한도, compute capability 조회 | `cudaGetDeviceProperties`, `cudaDeviceGetAttribute` | 전부 |
| 대용량 dynamic shared memory | `cudaFuncSetAttribute` (`MaxDynamicSharedMemorySize`) | vLLM, Ollama, SGLang (AOT 11곳, JIT 34곳), TRT-LLM (80곳) |
| Persistent 커널 그리드 크기 | `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaOccupancyAvailableDynamicSMemPerBlock` | vLLM, Ollama, SGLang, TRT-LLM |
| Grid-wide 동기화 커널 | `cudaLaunchCooperativeKernel` | Ollama (softmax), TRT-LLM (EPLB) |
| CUDA 버전에 따른 경로 선택 | `cudaRuntimeGetVersion`, `cudaDriverGetVersion` | vLLM, SGLang, TRT-LLM |
| SM별 AOTI 변형 선택 | `cudaGetDeviceProperties` | ExecuTorch |

ExecuTorch는 커널 최적화(fusion, autotune, 타일 선택)가 export 단계의 Inductor에서 끝나기 때문에 런타임 튜닝 API를 거의 쓰지 않습니다.

### 3.13 그 밖의 고유 기법

| 프로젝트 | 기법 | API |
|---|---|---|
| Ollama | CUDA 툴킷 없이 드라이버만 `dlopen`해서 GPU 탐지 (스케줄러 입력) | `cuInit`, `cuDriverGetVersion`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetAttribute`, `cuDeviceGetName`, `cuDeviceTotalMem_v2`, `cuDeviceGetPCIBusId` |
| Ollama (llama.cpp) | Concurrent streams (fork/join, 기본 꺼짐), 호스트 동기화 지연 감소 | `cudaEventRecord`, `cudaStreamWaitEvent`, `cudaSetDeviceFlags(cudaDeviceScheduleSpin)` |
| SGLang | 프로세스 간 스트림 순서 동기화 (호스트 동기화 없이 GPU 메모리 플래그로) | `cuStreamWriteValue32`, `cuStreamWaitValue32` |
| TensorRT-LLM | MoE Expert Load Balancer (EPLB) | `cudaMallocManaged`, `cudaMemAdvise`, `cudaMemcpy2DAsync`, `cudaLaunchCooperativeKernel` |
| TensorRT-LLM | Disaggregated serving 전송 (NIXL bounce) | `cudaDeviceGetStreamPriorityRange`, `cudaStreamCreateWithPriority`, `cudaStreamQuery` |
| ExecuTorch | Mutable state 세션 (한 프로그램을 mutable buffer만 따로 가진 여러 인스턴스로 실행), 가중치 공유 | `cudaMalloc`, `cudaMemcpy`, `cudaPointerGetAttributes` |
| ExecuTorch (AOTI) | 실행 완료 추적 (블로킹 없이 완료 확인) | `cudaEventRecord`, `cudaEventQuery` |

---

## 4. 프로젝트별 요약

### 4.1 vLLM — [상세](docs/vllm_cuda_api_usage.md)

- **Driver 18개, Runtime 호스트 38개 + 디바이스 2개.**
- Driver API는 거의 세 곳에서만 씁니다: **Sleep Mode(VMM)**, **KV 오프로드 배치 복사**(`cuMemcpyBatchAsync`), **Custom AllReduce 포인터 base 조회**(`cuPointerGetAttribute`).
- 나머지 최적화(PDL, 커널 튜닝, 양자화 보조, UVA 오프로드, IPC)는 Runtime API로 구현되어 있습니다.
- Python에서는 ctypes 래퍼(`CudaRTLibrary`)로 `libcudart`를 직접 로드합니다.
- CUDA Graph, 스트림, 이벤트는 PyTorch에 맡기므로 `cudaGraph*`, `cudaEvent*`가 없습니다.

### 4.2 Ollama — [상세](docs/ollama_cuda_api_usage.md)

- 저장소에 `.cu` 파일이 없고, 빌드할 때 **llama.cpp**(GGUF, 기본 엔진)와 **MLX**(safetensors)를 받아 함께 빌드합니다. carry patch는 CUDA 호출을 바꾸지 않습니다.
- **Ollama 본체(Go)는 Driver API 8개로 GPU 탐지만** 하고, Runtime API는 부르지 않습니다.
- **llama.cpp**: Driver API는 VMM 메모리 풀에만 쓰고, CUDA Graph(스트림 캡처), PDL, pinned 메모리, multi-GPU P2P·PCIe AllReduce는 Runtime API로 구현합니다.
- **MLX**: JIT 커널 실행, TMA, 그래프 노드 추가에 Driver API를 쓰고, 메모리는 `cudaMallocAsync` 기반 풀에 맡깁니다. 그래프를 노드 단위로 직접 조립합니다.

### 4.3 SGLang — [상세](docs/sglang_cuda_api_usage.md)

- **Driver 46개, Runtime 호스트 44개 + 디바이스 2개.** 구성: SRT 런타임(Python), AOT 커널, JIT 커널, Python 커널 ops, 멀티모달 생성.
- VMM, green context, 그래프 노드 분석, 스트림 메모리 연산처럼 Runtime API로는 할 수 없는 기능을 **Python(cuda-python)에서 직접** 다룹니다.
- VMM을 KV arena, DWDP, feature 전송, AllReduce v2의 **공통 기반**으로 씁니다.
- 커널 쪽은 vLLM과 비슷하게 Runtime API로 PDL, smem 설정, occupancy 계산을 하고, TMA만 entry point로 찾아 씁니다.

### 4.4 TensorRT-LLM — [상세](docs/tensorrt_llm_cuda_api_usage.md)

- **Driver 84개, Runtime 호스트 72개 + 디바이스 2개**로 가장 많습니다. 공통 래퍼 `cudaDriverWrapper`가 `libcuda`를 `dlopen`합니다(선언만 있는 7개는 집계 제외).
- 핵심 커널(FMHA, trtllmGen GEMM·MoE·FMHA)을 **미리 빌드한 CUBIN**으로 배포하고 Driver API로 로드합니다. XQA와 DeepGEMM은 NVRTC JIT입니다.
- NVLS 멀티캐스트, MNNVL fabric 메모리, Logical Endpoint처럼 **NVIDIA 시스템 전용 기능**을 직접 다룹니다.
- **KV Cache Manager v2**는 스트림·이벤트·메모리까지 Driver API로 작성되어 있고, 링크 종류에 따라 복사 전략을 바꿉니다.
- **PDL 사용량이 압도적**입니다(디바이스 측 약 300곳). CUDA Graph 캡처는 PyTorch에 맡깁니다.

### 4.5 ExecuTorch — [상세](docs/executorch_cuda_api_usage.md)

- CUDA 백엔드는 **AOTInductor 기반**입니다. export 때 Inductor가 Triton 커널과 C++ wrapper를 `.so`로 만들고, 런타임이 이를 `dlopen`해서 실행합니다 (`embed_kernel_binary=True`, `link_libtorch=False`).
- **ExecuTorch 자체 코드는 Driver API를 전혀 쓰지 않습니다.** Driver API 9개는 모두 AOTI 생성 코드에서 오며, 모든 Triton 커널이 `cuModuleLoadData` → `cuLaunchKernel`로 실행됩니다.
- 런타임은 Runtime API 37개로 전용 메모리 풀, CUDA Graph(기본 꺼짐), off-graph KV cache, AOTI C shim, 커스텀 양자화 GEMM을 구현합니다.
- 커널 튜닝이 export 단계에서 끝나서 런타임 튜닝 API를 거의 쓰지 않습니다.

---

## 5. 주요 관찰

1. **Driver API 사용량은 "하드웨어에 얼마나 가까이 붙는가"를 보여줍니다.** ExecuTorch(0) < vLLM(18) < Ollama(26) < SGLang(46) < TensorRT-LLM(84) 순입니다. Driver API가 필요한 곳은 대체로 VMM, 커널 바이너리 로딩, green context, 멀티캐스트·fabric 메모리처럼 Runtime API에 대응 기능이 없는 영역입니다.
2. **VMM은 가장 널리 쓰이는 Driver 기능입니다.** ExecuTorch를 제외한 네 프로젝트가 모두 쓰지만 목적은 다릅니다. vLLM은 sleep mode, llama.cpp는 단편화 없는 풀, SGLang은 프로세스 간 공유의 공통 기반, TensorRT-LLM은 KV 관리부터 멀티노드 fabric까지 씁니다.
3. **CUDA Graph를 다루는 방식이 세 갈래로 나뉩니다.** PyTorch에 위임(vLLM, TensorRT-LLM), PyTorch 캡처 위에 분석·dedup을 얹음(SGLang), 직접 캡처하거나 노드를 조립(llama.cpp, MLX, ExecuTorch)합니다. PyTorch에 의존하지 않는 엔진일수록 직접 관리합니다.
4. **PDL은 서버용 엔진에서 사실상 표준이 되었습니다.** vLLM, SGLang, TensorRT-LLM, llama.cpp가 모두 쓰며, TensorRT-LLM이 약 300곳으로 가장 광범위합니다. ExecuTorch와 MLX는 쓰지 않습니다.
5. **KV 오프로드의 배치 복사는 CUDA 12.8의 `cu(da)MemcpyBatchAsync`로 모이고 있습니다.** vLLM과 TensorRT-LLM은 Driver 버전을, SGLang은 Runtime 버전을 런타임에 심볼을 찾아 호출하고, 미지원 환경에서는 `cudaMemcpyAsync` 반복으로 폴백합니다. TensorRT-LLM은 링크 종류(C2C/PCIe)에 따라 SM 복사와 copy engine 복사를 고릅니다.
6. **SM 분할(green context)은 SGLang과 TensorRT-LLM만 직접 씁니다.** SGLang은 prefill·decode 동시 실행, TensorRT-LLM은 locality domain 분할이 목적입니다. 두 프로젝트 모두 `cuGetProcAddress`로 심볼을 찾아 드라이버 버전 차이에 대응합니다.
7. **커널 공급 방식이 런타임 API 사용 패턴을 결정합니다.** 손으로 작성한 커널(vLLM, SGLang, llama.cpp)은 `cudaFuncSetAttribute`와 occupancy API를 많이 쓰고, 미리 빌드하거나 생성한 커널(TensorRT-LLM CUBIN, ExecuTorch AOTI, MLX JIT)은 Driver의 module/library API로 로드합니다.

---

## 6. 상세 문서

| 문서 | 내용 |
|---|---|
| [docs/vllm_cuda_api_usage.md](docs/vllm_cuda_api_usage.md) | vLLM: 본체·테스트별 API 목록, 최적화 기법 10종 매핑 |
| [docs/ollama_cuda_api_usage.md](docs/ollama_cuda_api_usage.md) | Ollama: 본체 / llama.cpp / MLX 구성 요소별 API 목록, 기본값 포함 최적화 기법 12종 매핑 |
| [docs/sglang_cuda_api_usage.md](docs/sglang_cuda_api_usage.md) | SGLang: SRT / AOT / JIT / 커널 ops / 멀티모달별 API 목록, 최적화 기법 11종 매핑 |
| [docs/tensorrt_llm_cuda_api_usage.md](docs/tensorrt_llm_cuda_api_usage.md) | TensorRT-LLM: C++ / 커널 / Python별 API 목록, 최적화 기법 15종 매핑, 분석 한계 |
| [docs/executorch_cuda_api_usage.md](docs/executorch_cuda_api_usage.md) | ExecuTorch: 런타임 / AOTI 생성 코드별 API 목록, AOTI 컴파일 옵션, 최적화 기법 매핑 |
