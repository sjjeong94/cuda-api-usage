# Ollama CUDA API 사용 현황 및 최적화 기법 매핑

> 기준:
> - Ollama `main` 커밋 `8a971df` (2026-10-05)
> - llama.cpp `b11351` (`631109b`, Ollama `LLAMA_CPP_VERSION`) + Ollama carry patch 2종 적용
> - MLX `264c14f` (Ollama `MLX_VERSION`) + Ollama carry patch 적용
>
> 방법: 세 소스 트리에서 `cuXxx(` / `cudaXxx(` 호출을 grep하고 주석·타입·enum·로그 문자열·`#if 0` 코드는 제외. 템플릿 인자로 넘기는 함수(RAII 핸들의 destroy 함수 등)는 별도로 확인해 포함
> 범위: Ollama가 빌드해서 실행하는 코드가 **직접** 호출하는 API만 포함. cuBLAS, cuDNN, NCCL, NVRTC, CCCL 등 라이브러리 내부 호출은 제외

---

## 목차

0. [Ollama의 CUDA 코드 구성](#0-ollama의-cuda-코드-구성)
1. [CUDA Driver API](#1-cuda-driver-api)
2. [CUDA Runtime API](#2-cuda-runtime-api)
3. [최적화 기법별 API 매핑](#3-최적화-기법별-api-매핑)
4. [요약](#4-요약)

---

## 0. Ollama의 CUDA 코드 구성

Ollama 저장소에는 `.cu` 파일이 하나도 없습니다. CUDA 코드는 빌드할 때 아래 두 프로젝트를 받아서(`cmake/local.cmake`의 `ExternalProject_Add`) 패치한 뒤 함께 빌드합니다.

| 구성 요소 | 소스 | 역할 | CUDA 코드 위치 |
|---|---|---|---|
| **Ollama 본체 (Go)** | 이 저장소 | 서버, 스케줄러, GPU 탐지 | `discover/native_probe_{linux,windows}.go` (cgo + `dlopen`) |
| **llama.cpp runner** | llama.cpp `b11351` | GGUF 모델 추론 (기본 엔진) | `ggml/src/ggml-cuda/` |
| **MLX runner** | MLX `264c14f` | safetensors 모델 추론 (`mlxrunner/`), Linux/Windows에서 `MLX_BUILD_CUDA=ON` | `mlx/backend/cuda/` |

Ollama carry patch(`llama/compat/*.patch`, `mlx/compat/mlx/*.patch`)는 모델 메타데이터 변환, 텐서 스킵, Metal residency 같은 내용이고 **CUDA API 호출을 추가하거나 바꾸지 않습니다.**

빌드 옵션 중 CUDA 경로에 영향을 주는 것은 `GGML_CUDA_NO_PEER_COPY=ON` 하나인데, **Windows ROCm 빌드에만** 적용됩니다. NVIDIA CUDA 빌드는 llama.cpp 기본값을 그대로 씁니다.

---

## 1. CUDA Driver API

세 구성 요소를 합쳐 **26개**입니다 (중복 제외).

### 1.1 Ollama 본체: GPU 탐지

`discover/native_probe_linux.go`, `native_probe_windows.go`. `libcuda.so` / `nvcuda.dll`을 `dlopen`하고 `dlsym`으로 심볼을 찾아 cgo로 호출합니다. CUDA 툴킷 없이도 드라이버만 있으면 GPU를 탐지할 수 있게 하려는 구조입니다.

| API | 용도 |
|---|---|
| `cuInit` | 드라이버 초기화 |
| `cuDriverGetVersion` | 드라이버 버전 확인 (디바이스 정보의 `DriverMajor/Minor`로 기록) |
| `cuDeviceGetCount` | GPU 개수 |
| `cuDeviceGet` | 디바이스 핸들 획득 |
| `cuDeviceGetAttribute` | compute capability major/minor (75/76), integrated GPU 여부 (18) |
| `cuDeviceGetName` | GPU 이름 |
| `cuDeviceTotalMem_v2` (폴백 `cuDeviceTotalMem`) | 총 VRAM |
| `cuDeviceGetPCIBusId` | PCI 버스 ID (선택적, 없으면 생략) |

같은 파일에서 NVML(`nvmlInit_v2`, `nvmlSystemGetDriverVersion`, `nvmlShutdown`)도 쓰지만, 이것은 CUDA Driver API가 아닙니다.

### 1.2 llama.cpp (ggml-cuda)

`ggml/src/ggml-cuda/ggml-cuda.cu`, `common.cuh`

| API | 용도 |
|---|---|
| `cuDeviceGet` | 디바이스 핸들 획득 |
| `cuDeviceGetAttribute` | `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` 확인 |
| `cuMemGetAllocationGranularity` | VMM 할당 단위 조회 |
| `cuMemAddressReserve` | VMM 풀용 가상 주소 공간 예약 (`CUDA_POOL_VMM_MAX_SIZE`) |
| `cuMemCreate` | 물리 메모리 핸들 생성 |
| `cuMemMap` | 예약 영역 끝에 물리 메모리 매핑 (풀 확장) |
| `cuMemSetAccess` | 매핑 영역 접근 권한 설정 (multi-GPU면 여러 디바이스에 동시 부여) |
| `cuMemRelease` | 매핑 후 핸들 해제 (매핑은 유지) |
| `cuMemUnmap` | 풀 해제 시 매핑 해제 |
| `cuMemAddressFree` | 가상 주소 해제 |
| `cuGetErrorString` | `CU_CHECK` 매크로의 에러 메시지 |

### 1.3 MLX (CUDA 백엔드)

`mlx/backend/cuda/`

| API | 위치 | 용도 |
|---|---|---|
| `cuModuleLoadDataEx` | `jit_module.cpp` | NVRTC로 JIT 컴파일한 PTX/CUBIN 로드 |
| `cuModuleGetFunction` | `jit_module.cpp` | 로드한 모듈에서 커널 함수 획득 |
| `cuModuleUnload` | `jit_module.cpp` | 모듈 해제 |
| `cuLaunchKernelEx` | `device.cpp` | JIT 커널(`CUfunction`) 실행. cluster dimension 속성 지원 |
| `cuGraphAddKernelNode` | `device.cpp`, `jit_module.h` | JIT 커널을 CUDA Graph 노드로 직접 추가 |
| `cuGraphKernelNodeSetAttribute` | `device.cpp` | 그래프 노드에 cluster dimension 지정 |
| `cuFuncSetAttribute` | `custom_kernel.cpp`, `qmm_sm80.cu`, `qmm_sm90.cu` | JIT/커스텀 커널의 max dynamic smem 설정 |
| `cuOccupancyMaxPotentialBlockSize` | `utils.h` | JIT 커널의 최적 블록 크기 계산 |
| `cuTensorMapEncodeTiled` | `quantized/fp_quantize.cu` | TMA(Tensor Memory Accelerator) descriptor 생성 |
| `cuGetErrorString` | `utils.cpp` | 에러 메시지 |

---

## 2. CUDA Runtime API

| 구성 요소 | 호스트 API | 디바이스 측 API |
|---|---:|---:|
| Ollama 본체 | 0 | 0 |
| llama.cpp (ggml-cuda) | 48 (+ HIP 전용 1) | 2 |
| MLX | 55 | 0 |

Ollama 본체(Go)는 Runtime API를 직접 부르지 않습니다. `cudaRuntimeVersion`, `cudaFlashAttentionSupported`, `cudaJetpack` 같은 이름은 Ollama의 Go 헬퍼 함수입니다. `libcudart`는 runner 라이브러리를 찾을 때 경로 탐색 대상으로만 나옵니다.

### 2.1 llama.cpp (ggml-cuda)

#### 디바이스 / 속성

| API | 주요 위치 | 용도 |
|---|---|---|
| `cudaGetDeviceCount`, `cudaGetDevice`, `cudaSetDevice` | `ggml-cuda.cu` | 디바이스 열거·전환 |
| `cudaGetDeviceProperties` | `ggml-cuda.cu` | CC, SM 수, smem, integrated 여부 등 |
| `cudaDeviceGetAttribute` | `ggml-cuda.cu` | 개별 속성 조회 |
| `cudaDeviceGetPCIBusId` | `ggml-cuda.cu` | 디바이스 설명 (PCI ID) |
| `cudaMemGetInfo` | `ggml-cuda.cu` | 디바이스 free/total 메모리 (`ggml_backend_cuda_get_device_memory`) |
| `cudaSetDeviceFlags` | `ggml-cuda.cu` | `cudaDeviceScheduleSpin` 설정 |
| `cudaDeviceSynchronize` | `ggml-cuda.cu` | 동기화 |

#### Multi-GPU

| API | 주요 위치 |
|---|---|
| `cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess` | `ggml-cuda.cu` |
| `cudaMemcpyPeerAsync` | `ggml-cuda.cu` (backend 간 텐서 복사) |

#### 메모리 할당

| API | 주요 위치 | 용도 |
|---|---|---|
| `cudaMalloc` / `cudaFree` | `ggml-cuda.cu`, `common.cuh`, `allreduce.cu` | 디바이스 버퍼, legacy 풀 |
| `cudaMallocManaged` | `ggml-cuda.cu` | `GGML_CUDA_ENABLE_UNIFIED_MEMORY` 설정 시 |
| `cudaMemAdvise` | `ggml-cuda.cu` | HIP 빌드 전용 (`hipMemAdviseSetCoarseGrain`) |
| `cudaMallocHost` / `cudaFreeHost` | `ggml-cuda.cu` | pinned host 버퍼 (CPU에 남는 레이어용) |
| `cudaHostRegister` / `cudaHostUnregister` | `ggml-cuda.cu` | 기존 호스트 버퍼(mmap된 모델 등)를 pin (`GGML_CUDA_REGISTER_HOST`) |
| `cudaHostAlloc` / `cudaHostGetDevicePointer` | `allreduce.cu` | mapped pinned memory (PCIe AllReduce 스테이징) |

#### 메모리 복사 / 초기화

| API | 주요 위치 |
|---|---|
| `cudaMemcpyAsync` | `ggml-cuda.cu`, `cpy.cu`, `concat.cu`, `argsort.cu`, `solve_tri.cu`, `allreduce.cu` (16곳) |
| `cudaMemcpy2DAsync` | `ggml-cuda.cu`, `cpy.cu`, `top-k.cu` (strided 텐서, row split) |
| `cudaMemset` / `cudaMemsetAsync` | `ggml-cuda.cu`, `mmq.cu`, `mmvq.cu`, `count-equal.cu`, `allreduce.cu` |

#### 스트림 / 이벤트

| API | 주요 위치 |
|---|---|
| `cudaStreamCreateWithFlags` / `cudaStreamDestroy` | `common.cuh`, `ggml-cuda.cu`, `allreduce.cu` |
| `cudaStreamSynchronize` | `ggml-cuda.cu`, `allreduce.cu` (11곳) |
| `cudaStreamWaitEvent` | `ggml-cuda.cu`, `allreduce.cu` (9곳) |
| `cudaEventCreateWithFlags` / `cudaEventDestroy` / `cudaEventRecord` / `cudaEventSynchronize` | `common.cuh`, `ggml-cuda.cu`, `allreduce.cu` |

#### CUDA Graph

| API | 주요 위치 |
|---|---|
| `cudaStreamBeginCapture` / `cudaStreamEndCapture` | `ggml-cuda.cu` |
| `cudaGraphInstantiate` / `cudaGraphExecUpdate` / `cudaGraphLaunch` | `ggml-cuda.cu` |
| `cudaGraphDestroy` / `cudaGraphExecDestroy` | `common.cuh`, `ggml-cuda.cu` |
| `cudaStreamIsCapturing` | `argsort.cu`, `mean.cu` (캡처 중이면 다른 경로 선택) |

#### 커널 설정 / 실행

| API | 주요 위치 | 용도 |
|---|---|---|
| `cudaFuncSetAttribute` | `common.cuh`, `fattn-mma-f16.cuh` | 대용량 dynamic smem (FlashAttention 등) |
| `cudaFuncGetAttributes` | `common.cuh` | 커널 속성 조회 |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | `fattn-common.cuh` | FlashAttention 그리드 크기 |
| `cudaLaunchKernelEx` | `common.cuh` (`ggml_cuda_kernel_launch`) | PDL 속성을 붙여 실행 |
| `cudaLaunchCooperativeKernel` | `softmax.cu` | 열 방향 병렬 softmax (grid-wide sync) |

#### 에러 처리

| API | 용도 |
|---|---|
| `cudaGetLastError` (34곳) / `cudaGetErrorString` | 커널 실행 후 체크, `CUDA_CHECK` |

#### 디바이스 측 API (PDL, Hopper 이상)

| API | 위치 |
|---|---|
| `cudaGridDependencySynchronize` | `common.cuh` |
| `cudaTriggerProgrammaticLaunchCompletion` | `common.cuh` |

**제외한 항목**
- `cudaLaunchHostFunc`: `ggml-cuda.cu`의 `#if 0` 블록 안에만 있습니다 (미사용).
- `cudaEventCreate`: 로그 문자열에만 나옵니다.
- `vendors/hip.h`, `vendors/musa.h`: CUDA 이름을 HIP/MUSA로 바꾸는 매핑 헤더라서 제외했습니다.

### 2.2 MLX (CUDA 백엔드)

#### 디바이스 / 속성

| API | 주요 위치 |
|---|---|
| `cudaGetDeviceCount`, `cudaGetDevice`, `cudaSetDevice` | `device_info.cpp`, `device.cpp`, `distributed/nccl/nccl.cpp` |
| `cudaGetDeviceProperties` | `device_info.cpp`, `wddm.cpp` |
| `cudaDeviceGetAttribute` | `device.cpp`, `rms_norm.cu` |
| `cudaMemGetInfo` | `allocator.cpp`, `device_info.cpp` |

#### 메모리 할당 (`allocator.cpp`)

| API | 용도 |
|---|---|
| `cudaDeviceGetDefaultMemPool` | 디바이스 기본 메모리 풀 획득 |
| `cudaMallocAsync` / `cudaFreeAsync` | 스트림 순서 기반 풀 할당 (메모리 풀을 지원하는 디바이스) |
| `cudaMemPoolGetAttribute` | 풀 사용량 조회 (`wddm.cpp` 포함) |
| `cudaMemPoolTrimTo` | 캐시 정리 시 풀 반환 |
| `cudaMalloc` / `cudaFree` | 메모리 풀 미지원 시 폴백 |
| `cudaMallocManaged` / `cudaMemAdvise` | unified memory (integrated GPU, concurrent managed access) |
| `cudaMallocHost` / `cudaFreeHost` | managed memory 미지원 시 pinned host 폴백 |
| `cudaMemcpy` / `cudaMemcpyAsync` | 버퍼 이동, 가중치 로드 (`load.cpp`) |
| `cudaMemset` | 이벤트 플래그 초기화 (`event.cu`) |

#### 스트림 / 이벤트 / 동기화 (`event.cu`, `utils.cpp`, `worker.cpp`)

| API | 용도 |
|---|---|
| `cudaStreamCreateWithFlags` / `cudaStreamDestroy` | 스트림 RAII 핸들 |
| `cudaStreamSynchronize` | 동기화 |
| `cudaStreamWaitEvent` | 스트림 간 의존성 |
| `cudaEventCreateWithFlags` / `cudaEventDestroy` / `cudaEventRecord` / `cudaEventSynchronize` / `cudaEventQuery` | 이벤트 RAII 핸들, 완료 확인 |
| `cudaLaunchHostFunc` | 가중치 로드 후 호스트 버퍼 해제 (`load.cpp`), 완료 시그널 (`worker.cpp`) |

#### CUDA Graph (`device.cpp`, `utils.cpp`)

MLX는 스트림 캡처만 쓰지 않고, **그래프를 노드 단위로 직접 조립**합니다.

| API | 용도 |
|---|---|
| `cudaGraphCreate` / `cudaGraphDestroy` | 빈 그래프 생성 / 해제 |
| `cudaGraphAddKernelNode` | 커널을 노드로 직접 추가 |
| `cudaGraphAddEmptyNode` / `cudaGraphAddDependencies` | 노드 간 의존성 구성 |
| `cudaStreamBeginCapture` / `cudaStreamEndCapture` / `cudaStreamIsCapturing` | cuBLAS·cuDNN 호출처럼 직접 노드로 만들 수 없는 작업은 캡처해서 child graph로 사용 |
| `cudaGraphAddChildGraphNode` / `cudaGraphChildGraphNodeGetGraph` | 캡처한 그래프를 하위 그래프로 삽입 |
| `cudaGraphGetNodes` / `cudaGraphNodeGetType` / `cudaGraphKernelNodeGetAttribute` / `cudaGraphKernelNodeSetAttribute` | 그래프 구조 비교와 노드 속성(cluster dim) 설정 |
| `cudaGraphInstantiate` / `cudaGraphExecUpdate` / `cudaGraphLaunch` / `cudaGraphExecDestroy` | 인스턴스화, 기존 exec 재사용 업데이트, 실행 |
| `cudaGraphDebugDotPrint` | `MLX_SAVE_CUDA_GRAPHS_DOT_FILE` 설정 시 그래프 덤프 |

#### 커널 설정 / 실행

| API | 주요 위치 | 용도 |
|---|---|---|
| `cudaLaunchKernelExC` | `device.cpp` | AOT 커널 실행 (cluster dimension 속성) |
| `cudaFuncSetAttribute` | `rms_norm.cu` | dynamic smem 설정 |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | `rms_norm.cu` | 그리드 크기 결정 |
| `cudaOccupancyMaxPotentialBlockSize` | `utils.h` | 블록 크기 결정 |

#### 에러 처리

| API | 용도 |
|---|---|
| `cudaGetLastError` / `cudaGetErrorString` | `CHECK_CUDA_ERROR` 매크로 |

---

## 3. 최적화 기법별 API 매핑

표기: **[O]** Ollama 본체, **[L]** llama.cpp (ggml-cuda), **[M]** MLX

### 3.1 드라이버만으로 하는 GPU 탐지와 runner 선택 [O]

CUDA 툴킷 없이 드라이버 라이브러리만 `dlopen`해서 GPU를 찾고, compute capability·드라이버 버전·VRAM 같은 디바이스 정보를 수집합니다. 이 정보는 스케줄러가 GPU를 고르고 offload 양을 정하는 데 쓰입니다.

| 목적 | API | 종류 |
|---|---|---|
| 초기화·열거 | `cuInit`, `cuDeviceGetCount`, `cuDeviceGet` | Driver |
| 호환성 정보 | `cuDriverGetVersion`, `cuDeviceGetAttribute` (CC, integrated) | Driver |
| 스케줄링 정보 | `cuDeviceTotalMem_v2`, `cuDeviceGetName`, `cuDeviceGetPCIBusId` | Driver |
| runner 쪽 실측 메모리 [L] | `cudaMemGetInfo`, `cudaDeviceGetPCIBusId` | Runtime |

### 3.2 VMM 기반 메모리 풀 [L]

가상 주소 공간을 크게 예약해 두고, 필요할 때만 물리 메모리를 끝에 이어 붙입니다. 주소가 연속으로 유지되므로 단편화 없이 풀을 키울 수 있습니다. VMM을 지원하는 GPU에서 기본으로 켜지고(`GGML_CUDA_NO_VMM`으로 끔), 지원하지 않으면 `cudaMalloc` 기반 legacy 풀을 씁니다.

| 단계 | API | 종류 |
|---|---|---|
| 지원 여부 확인 | `cuDeviceGet`, `cuDeviceGetAttribute` (`VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`) | Driver |
| 준비 | `cuMemGetAllocationGranularity`, `cuMemAddressReserve` | Driver |
| 확장 | `cuMemCreate` → `cuMemMap` → `cuMemRelease` → `cuMemSetAccess` | Driver |
| 해제 | `cuMemUnmap` → `cuMemAddressFree` | Driver |
| 폴백 (legacy 풀) | `cudaMalloc`, `cudaFree` | Runtime |

### 3.3 스트림 순서 메모리 풀 / Unified Memory [M]

| 목적 | API | 종류 |
|---|---|---|
| 스트림 순서 기반 풀 할당 (free 시 동기화 불필요) | `cudaDeviceGetDefaultMemPool`, `cudaMallocAsync`, `cudaFreeAsync` | Runtime |
| 풀 사용량 확인·반환 | `cudaMemPoolGetAttribute`, `cudaMemPoolTrimTo` | Runtime |
| Integrated GPU에서 CPU-GPU 메모리 공유 | `cudaMallocManaged`, `cudaMemAdvise` | Runtime |
| 폴백 | `cudaMalloc`, `cudaMallocHost` | Runtime |
| [L] 쪽 unified memory (opt-in) | `cudaMallocManaged` (`GGML_CUDA_ENABLE_UNIFIED_MEMORY`) | Runtime |

### 3.4 CUDA Graph

토큰마다 반복되는 커널 수백 개를 그래프 하나로 묶어 launch 오버헤드를 줄입니다. 두 엔진 모두 기본으로 켜져 있지만 방식이 다릅니다.

| 엔진 | 방식 | API |
|---|---|---|
| [L] | **스트림 캡처**. 이전 그래프와 구조가 같으면 exec를 업데이트해서 재사용. `GGML_CUDA_DISABLE_GRAPHS`로 끔 | `cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaGraphInstantiate`, `cudaGraphExecUpdate`, `cudaGraphLaunch`, `cudaGraphDestroy`, `cudaGraphExecDestroy` |
| [L] | 캡처와 호환되지 않는 CUB 정렬은 캡처 중이면 다른 알고리즘 사용 | `cudaStreamIsCapturing` (`argsort.cu`, `mean.cu`) |
| [M] | **노드를 직접 조립**. 커널은 노드로 바로 추가하고, cuBLAS/cuDNN 호출만 캡처해서 child graph로 삽입. 그래프 캐시 400개 (`MLX_CUDA_GRAPH_CACHE_SIZE`), `MLX_USE_CUDA_GRAPHS`로 끔 | `cudaGraphCreate`, `cudaGraphAddKernelNode`, `cuGraphAddKernelNode`, `cudaGraphAddEmptyNode`, `cudaGraphAddDependencies`, `cudaGraphAddChildGraphNode`, `cudaStreamBeginCapture/EndCapture`, `cudaGraphExecUpdate`, `cudaGraphLaunch` |
| [M] | 그래프 노드에 cluster dimension 지정 | `cudaGraphKernelNodeSetAttribute`, `cuGraphKernelNodeSetAttribute` |

### 3.5 PDL (Programmatic Dependent Launch) [L]

앞 커널이 끝나기 전에 다음 커널을 미리 띄워서 launch 지연을 겹치게 만듭니다. CUDA 12.3 이상(Linux는 11.8 이상)으로 빌드하면 컴파일되고(`GGML_CUDA_USE_PDL`), 실행 시 **기본으로 켜져 있습니다**(`GGML_CUDA_PDL=0`으로 끔). 디바이스 측 동작은 Hopper 이상에서만 합니다.

| 위치 | API |
|---|---|
| 호스트 | `cudaLaunchKernelEx` (`ggml_cuda_kernel_launch`에서 PDL 가능한 커널만) |
| 디바이스 | `cudaGridDependencySynchronize`, `cudaTriggerProgrammaticLaunchCompletion` |

### 3.6 JIT 커널 컴파일 [M]

연산을 실행 시점에 fuse한 커널(elementwise fusion, custom kernel)을 NVRTC로 컴파일해서 Driver API로 로드·실행합니다.

| 목적 | API | 종류 |
|---|---|---|
| 모듈 로드·함수 획득·해제 | `cuModuleLoadDataEx`, `cuModuleGetFunction`, `cuModuleUnload` | Driver |
| 실행 설정 | `cuFuncSetAttribute` (smem), `cuOccupancyMaxPotentialBlockSize` (블록 크기) | Driver |
| 실행 | `cuLaunchKernelEx` (단독), `cuGraphAddKernelNode` (그래프 노드) | Driver |

### 3.7 Hopper/Blackwell 전용 기능 [M]

| 기능 | API | 사용처 |
|---|---|---|
| TMA (Tensor Memory Accelerator) | `cuTensorMapEncodeTiled` | FP 양자화 (`fp_quantize.cu`) |
| Thread Block Cluster | `cudaLaunchKernelExC`, `cuLaunchKernelEx` (`cudaLaunchAttributeClusterDimension`) | `device.cpp` 공통 실행 경로 |
| 대용량 smem (SM80/SM90 양자화 GEMM) | `cuFuncSetAttribute` | `qmm_sm80.cu`, `qmm_sm90.cu` |

### 3.8 부분 GPU Offload와 Pinned 메모리 [L]

VRAM이 부족하면 일부 레이어를 CPU에 두는데(Ollama의 기본 동작), 이때 CPU 쪽 버퍼를 pinned memory로 만들어 전송 속도를 높입니다.

| 목적 | API | 기본값 |
|---|---|---|
| pinned host 버퍼 할당 | `cudaMallocHost`, `cudaFreeHost` | 켜짐 (`GGML_CUDA_NO_PINNED`로 끔) |
| 이미 있는 버퍼(mmap 모델 등)를 pin | `cudaHostRegister` (`Portable \| ReadOnly`), `cudaHostUnregister` | 꺼짐 (`GGML_CUDA_REGISTER_HOST`로 켬) |
| 호스트↔디바이스 전송 | `cudaMemcpyAsync`, `cudaMemcpy2DAsync` | |

[M]은 가중치를 로드할 때 `cudaMemcpyAsync`로 올린 뒤, `cudaLaunchHostFunc`로 복사가 끝나는 시점에 호스트 버퍼를 해제합니다. 호스트가 동기화를 기다리지 않아도 됩니다.

### 3.9 Multi-GPU (Layer Split / Tensor Split) [L]

| 기법 | API |
|---|---|
| GPU 간 직접 복사 (layer split, NVLink/PCIe P2P) | `cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess`, `cudaMemcpyPeerAsync` |
| 백엔드 간 동기화 | `cudaEventRecord`, `cudaStreamWaitEvent` |
| VMM 풀을 여러 GPU에서 접근 | `cuMemSetAccess` (access descriptor 여러 개) |
| Row split 텐서 분배 | `cudaMemcpy2DAsync` |
| **PCIe AllReduce** (`allreduce.cu`): NVLink 없는 2-GPU tensor parallel에서 pinned host 메모리를 거쳐 합산 | `cudaHostAlloc` (`Mapped \| Portable`), `cudaHostGetDevicePointer`, `cudaStreamCreateWithFlags`, `cudaEvent*`, `cudaStreamWaitEvent`, `cudaMemcpyAsync`, `cudaMemsetAsync` (`GGML_CUDA_ALLREDUCE`로 선택) |

[M]은 NCCL(`distributed/nccl/nccl.cpp`)로 분산 처리하며, Runtime API는 디바이스 설정(`cudaSetDevice`, `cudaGetDeviceCount`)에만 씁니다.

### 3.10 Concurrent Streams (그래프 최적화) [L]

독립적인 연산 가지를 여러 스트림에서 동시에 실행합니다 (fork/join). **기본은 꺼짐**이고 `GGML_CUDA_GRAPH_OPT=1`로 켭니다.

| API |
|---|
| `cudaStreamCreateWithFlags`, `cudaEventCreateWithFlags`, `cudaEventRecord`, `cudaStreamWaitEvent` (`fork_event` / `join_events`) |

### 3.11 커널 튜닝 / 하드웨어 적응형 실행

| 목적 | API | 사용처 |
|---|---|---|
| 아키텍처별 커널 선택 (MMQ, FlashAttention, tensor core 경로) | `cudaGetDeviceProperties`, `cudaDeviceGetAttribute` | [L] [M] |
| 대용량 dynamic smem | `cudaFuncSetAttribute` | [L] FlashAttention (`fattn-mma-f16.cuh`), [M] RMSNorm |
| 그리드 크기 결정 | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | [L] FlashAttention (stream-k 분할), [M] RMSNorm |
| Grid-wide 동기화 softmax | `cudaLaunchCooperativeKernel` | [L] `softmax.cu` |
| 호스트 동기화 지연 감소 | `cudaSetDeviceFlags(cudaDeviceScheduleSpin)` | [L] |

### 3.12 공통 / 부가

| 분류 | API |
|---|---|
| 에러 처리 | `cudaGetLastError`, `cudaGetErrorString`, `cuGetErrorString` |
| 디버깅 | `cudaGraphDebugDotPrint` [M] |
| Windows 메모리 계측 (WDDM) | `cudaGetDeviceProperties`, `cudaMemPoolGetAttribute` [M] `wddm.cpp` |

---

## 4. 요약

| 최적화 기법 | 엔진 | Driver API | Runtime API | 기본값 |
|---|:---:|:---:|:---:|:---:|
| 드라이버 기반 GPU 탐지 | O | ● | | 켜짐 |
| VMM 메모리 풀 | L | ● (핵심) | ○ (폴백) | 켜짐 |
| 스트림 순서 메모리 풀 / Unified Memory | M | | ● | 켜짐 |
| CUDA Graph (스트림 캡처) | L | | ● | 켜짐 |
| CUDA Graph (노드 직접 조립) | M | ○ | ● | 켜짐 |
| PDL | L | | ● | 켜짐 (CUDA 12.3+, Hopper+) |
| JIT 커널 | M | ● (핵심) | | 켜짐 |
| TMA / Thread Block Cluster | M | ● | ● | 하드웨어 의존 |
| Pinned 메모리 (부분 offload) | L | | ● | 켜짐 |
| Multi-GPU P2P / PCIe AllReduce | L | ○ | ● | 상황 의존 |
| Concurrent Streams | L | | ● | 꺼짐 |
| 커널 튜닝 | L, M | ○ | ● | 켜짐 |

- **Ollama 본체는 CUDA 계산을 하지 않습니다.** Driver API 8개로 GPU를 탐지하고 스케줄링할 뿐이고, 실제 CUDA 코드는 모두 llama.cpp와 MLX에서 옵니다.
- **llama.cpp**에서 Driver API는 **VMM 메모리 풀**에만 쓰이고, 나머지(CUDA Graph, PDL, pinned 메모리, multi-GPU)는 Runtime API로 구현되어 있습니다.
- **MLX**는 **JIT 커널 실행, TMA, 그래프 노드 추가**에 Driver API를 쓰고, 메모리 관리는 Runtime의 **스트림 순서 메모리 풀**(`cudaMallocAsync`)에 맡깁니다.
- vLLM과 달리 Ollama 쪽 엔진들은 **스트림, 이벤트, CUDA Graph를 직접 관리합니다.** vLLM은 이것을 PyTorch에 맡기기 때문에 `cudaGraph*`, `cudaEvent*`가 목록에 없었습니다.
