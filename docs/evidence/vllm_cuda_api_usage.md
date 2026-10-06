# vLLM CUDA API 사용 현황 및 최적화 기법 매핑

> 기준: vLLM `main` 커밋 `68088ed` (2026-10-06)
> 방법: 소스 전체(`csrc/`, `vllm/`, `tests/`, `benchmarks/`, `.buildkite/`)에서 `cuXxx(` / `cudaXxx(` 호출을 grep하고 주석·타입·enum은 제외
> 범위: vLLM 저장소가 **직접** 호출하는 API만 포함. PyTorch, NCCL, FlashInfer, Triton, CUTLASS/CuTe DSL 등 의존 라이브러리 내부 호출은 제외

---

## 목차

1. [CUDA Driver API](#1-cuda-driver-api)
2. [CUDA Runtime API](#2-cuda-runtime-api)
3. [최적화 기법별 API 매핑](#3-최적화-기법별-api-매핑)
4. [요약](#4-요약)

---

## 1. CUDA Driver API

vLLM 본체는 **18개**를 사용합니다.

### 1.1 vLLM 본체 (런타임 코드)

#### 가상 메모리 관리 (VMM): `csrc/cumem_allocator.cpp`

| API | 용도 |
|---|---|
| `cuMemAddressReserve` | 가상 주소 공간 예약 |
| `cuMemAddressFree` | 예약한 가상 주소 해제 |
| `cuMemCreate` | 물리 메모리 핸들 생성 (청크 단위 할당 포함) |
| `cuMemRelease` | 물리 메모리 핸들 해제 |
| `cuMemMap` | 물리 메모리를 가상 주소에 매핑 |
| `cuMemUnmap` | 매핑 해제 |
| `cuMemSetAccess` | 매핑 영역의 접근 권한 설정 |
| `cuMemGetAllocationGranularity` | 할당 단위(granularity) 조회 |

#### 컨텍스트 / 디바이스 / 에러: `csrc/cumem_allocator.cpp`

| API | 용도 |
|---|---|
| `cuCtxGetCurrent` | 현재 컨텍스트 확인 |
| `cuDevicePrimaryCtxRetain` | 컨텍스트가 없으면 primary context 확보 |
| `cuCtxSetCurrent` | 확보한 컨텍스트를 현재 스레드에 설정 |
| `cuDeviceGetAttribute` | RDMA 지원 여부와 Fabric handle 지원 여부 조회 |
| `cuGetErrorString` | `CUDA_CHECK` 매크로의 에러 메시지 변환 |

#### KV 캐시 오프로드 / 블록 스왑

| API | 위치 | 용도 |
|---|---|---|
| `cuGetProcAddress` | `csrc/libtorch_stable/cache_kernels.cu`, `vllm/v1/simple_kv_offload/cuda_mem_ops.py` | `cuMemcpyBatchAsync` 심볼을 런타임에 해석 (CUDA 12.8 이상, version 12080) |
| `cuMemcpyBatchAsync` | 위와 동일 | 여러 블록 복사를 한 번에 제출. legacy default stream에서는 사용할 수 없음 |

#### Custom AllReduce

| API | 위치 | 용도 |
|---|---|---|
| `cuPointerGetAttribute` | `csrc/custom_all_reduce.cuh` | `CU_POINTER_ATTRIBUTE_RANGE_START_ADDR`로 IPC 버퍼의 base 주소 조회 |

**참고 사항**
- `csrc/cumem_allocator_compat.h`는 ROCm 빌드용 shim으로, 위 VMM/컨텍스트 API를 HIP 함수로 매핑합니다.
- CuTe DSL 커널들(`cuda.bindings.driver`)은 `CUstream` 타입만 import하고, 드라이버 함수를 직접 호출하지는 않습니다.
- `cuFileDriverOpen`, `cuFileHandleRegister`(`weight_utils.py`)는 주석에만 나옵니다. 실제 호출은 외부 라이브러리가 하는 GDS(cuFile) API이고, Driver API가 아닙니다.

### 1.2 테스트 / CI / 벤치마크에서만 사용

| API | 위치 |
|---|---|
| `cuInit`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetUuid_v2` | `.buildkite/scripts/ci-otel/ci_gpu.py` (GPU UUID 수집, ctypes) |
| `cuCtxGetDevice` | `tests/cuda/test_cuda_context.py` (ctypes로 `libcuda.so` 로드) |
| `cuFuncGetAttribute` | `benchmarks/kernels/benchmark_k3_cutedsl_residual.py` |

---

## 2. CUDA Runtime API

vLLM 본체는 **호스트 API 38개와 디바이스 측 API 2개**를 사용합니다.

### 2.1 C++/CUDA 확장 (`csrc/`)

#### 디바이스 / 속성 조회

| API | 주요 위치 |
|---|---|
| `cudaGetDevice` | custom_all_reduce, cutlass MLA, router gemm 등 |
| `cudaGetDeviceCount` | `torch_utils.h` |
| `cudaGetDeviceProperties` | `torch_utils.h`, `cuda_compat.h`, minimax_reduce_rms |
| `cudaDeviceGetAttribute` | marlin, sampler, cuda_utils 등 25곳 (SM 수, smem 한도 등) |
| `cudaRuntimeGetVersion` | `nvfp4_scaled_mm_entry.cu` |
| `cudaDeviceSynchronize` | `w4a8_utils.cu` |

#### 커널 설정 / 실행

| API | 주요 위치 |
|---|---|
| `cudaFuncSetAttribute` | marlin, selective_scan, sm100 MLA, cooperative_topk 등 (dynamic smem 설정) |
| `cudaFuncGetAttributes` | `marlin_moe_wna16/ops.cu` |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | sm100 MLA, topk, minimax_reduce_rms |
| `cudaLaunchKernel` | `persistent_topk.cuh` |
| `cudaLaunchKernelEx` | fused_qknorm_rope, nvfp4 quant, per_token_group_quant 등 27곳 (PDL 속성 지정용) |

#### 메모리 할당 / 복사

| API | 주요 위치 |
|---|---|
| `cudaMalloc` / `cudaFree` | `custom_all_reduce.cu` |
| `cudaHostAlloc` / `cudaFreeHost` / `cudaHostGetDevicePointer` | `cuda_view.cu` (mapped pinned memory) |
| `cudaMemcpy` | custom_all_reduce, cuda_view |
| `cudaMemcpyAsync` | cache_kernels, custom_all_reduce, custom_all_gather_reduce_scatter |
| `cudaMemcpy2DAsync` | sm100 fp8 blockwise scale padding |
| `cudaMemcpyToSymbol` | `w4a8_utils.cu` (LUT를 constant memory에 올림) |
| `cudaMemsetAsync` | topk, fp8 common, custom_all_reduce |
| `cudaPointerGetAttributes` | cache_kernels, hisparse_kernels |

#### IPC (Custom AllReduce)

| API | 주요 위치 |
|---|---|
| `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` / `cudaIpcCloseMemHandle` | `custom_all_reduce.cu(h)` |

#### 스트림 / CUDA Graph

| API | 주요 위치 |
|---|---|
| `cudaStreamIsCapturing` | `custom_all_reduce.cuh` |
| `cudaThreadExchangeStreamCaptureMode` | `custom_all_reduce.cu` |
| `cudaStreamGetCaptureInfo` | `rocm/skinny_gemms.cu` (hipify 대상) |
| `cudaStreamSynchronize` | custom_all_reduce, rocm/skinny_gemms |

#### 에러 처리

| API | 용도 |
|---|---|
| `cudaGetLastError` / `cudaGetErrorString` | 커널 실행 후 체크, `CUDA_CHECK` 매크로 |

#### 디바이스 측 API (PDL, SM90 이상)

| API | 위치 |
|---|---|
| `cudaGridDependencySynchronize` | fused_qknorm_rope, nvfp4 quant, dsv3_fused_a_gemm 등 33곳 |
| `cudaTriggerProgrammaticLaunchCompletion` | 위와 같은 커널들 (37곳) |

### 2.2 Python 쪽 (`vllm/`)

Python에서 런타임 API를 부르는 경로는 두 가지입니다.

**ctypes 래퍼.** `vllm/distributed/device_communicators/cuda_wrapper.py`의 `CudaRTLibrary`가 `libcudart`를 직접 로드해서 아래 함수들을 노출합니다.

`cudaSetDevice`, `cudaDeviceSynchronize`, `cudaDeviceReset`, `cudaGetErrorString`, `cudaGetLastError`, `cudaMalloc`, `cudaFree`, `cudaMemset`, `cudaMemcpy`, `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle`, `cudaHostRegister`, `cudaHostUnregister`

| 사용처 | API |
|---|---|
| `device_allocator/cumem.py` (sleep mode 백업과 복원) | `cudaMemcpy`, `cudaDeviceReset` |
| `distributed/device_communicators/all_reduce_utils.py` (P2P 접근 테스트) | `cudaSetDevice`, `cudaMemset`, `cudaDeviceSynchronize`, `cudaDeviceReset`, `cudaHostUnregister` |
| `v1/kv_offload/cpu/*`, `v1/simple_kv_offload/cuda_mem_ops.py`, `ec_shared_region.py`, `v1/hisparse/runtime.py` (공유 메모리 영역 pin) | `cudaHostRegister`, `cudaHostUnregister`, `cudaGetLastError` |
| `models/deepseek_v41/common/engram.py` | `cudaHostRegister`, `cudaHostUnregister`, `cudaIpcCloseMemHandle` |

**cuda-python `cudart`.** `model_executor/layers/minimax_rms_norm/lamport_workspace.py`에서 `cudaMalloc`, `cudaMemset`, `cudaMemcpy`, `cudaFree`, `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle`을 사용합니다.

### 2.3 테스트 / 벤치마크에서만 사용

- `cudaProfilerStart` / `cudaProfilerStop`: `benchmarks/kernels/*`에서 `torch.cuda.cudart()`로 호출
- 그 외 테스트는 위에 나온 API를 재사용할 뿐이고, 새로 추가되는 API는 없습니다.

### 2.4 제외한 항목

- 주석에만 나오는 것: `cudaMemGetInfo`, `cudaStreamWaitEvent`, `cudaMemcpy2D/3DAsync` (Python 쪽)
- 이벤트·스트림 생성(`cudaEventRecord`, `cudaStreamCreate`)과 graph capture는 vLLM이 직접 하지 않고 PyTorch(`torch.cuda.*`)를 통해 이뤄집니다.

---

## 3. 최적화 기법별 API 매핑

### 3.1 Sleep Mode / 가상 메모리 기반 할당기 (`CuMemAllocator`)

RLHF처럼 학습과 추론을 번갈아 할 때 GPU 메모리를 내려놓았다가 같은 주소로 다시 올리는 기능입니다. 가상 주소는 그대로 두고 물리 메모리만 해제하고 다시 붙이기 때문에, CUDA Graph나 텐서 포인터를 다시 만들 필요가 없습니다.

| 단계 | API | 종류 |
|---|---|---|
| 준비 | `cuMemGetAllocationGranularity` | Driver |
| 할당 | `cuMemAddressReserve` → `cuMemCreate` → `cuMemMap` → `cuMemSetAccess` | Driver |
| 해제 | `cuMemUnmap` → `cuMemRelease` → `cuMemAddressFree` | Driver |
| 컨텍스트 | `cuCtxGetCurrent`, `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent` | Driver |
| RDMA·Fabric handle 지원 확인 | `cuDeviceGetAttribute` | Driver |
| 에러 | `cuGetErrorString` | Driver |
| offload할 때 CPU로 백업·복원 | `cudaMemcpy` (ctypes 래퍼) | Runtime |

### 3.2 KV Cache CPU 오프로드 / 블록 스왑

GPU KV 블록을 pinned CPU 메모리로 내보내고 다시 가져옵니다.

| 목적 | API | 종류 |
|---|---|---|
| 배치 복사 (CUDA 12.8 이상): 블록 수백 개의 복사를 한 번의 호출로 제출 | `cuGetProcAddress` → `cuMemcpyBatchAsync` | Driver |
| 폴백: 블록마다 반복 복사 | `cudaMemcpyAsync` | Runtime |
| 공유 메모리 오프로드 영역을 pinned 메모리로 등록 | `cudaHostRegister`, `cudaHostUnregister`, `cudaGetLastError` | Runtime |
| 호스트 버퍼가 pinned인지 확인해 경로 선택 (cache_kernels, hisparse) | `cudaPointerGetAttributes` | Runtime |

### 3.3 Custom AllReduce (TP 통신 최적화)

작은 텐서에서는 NCCL 대신 NVLink P2P로 직접 AllReduce를 해서 지연을 줄입니다.

| 목적 | API | 종류 |
|---|---|---|
| 피어 메모리 공유 (IPC) | `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle`, `cudaIpcCloseMemHandle` | Runtime |
| IPC 버퍼의 base 주소 계산. 캐싱 할당기가 내준 텐서는 큰 블록 안의 offset이라 base가 따로 필요함 | `cuPointerGetAttribute` (`RANGE_START_ADDR`) | Driver |
| 캡처 중인지 확인하고, 캡처 중에 사용한 버퍼는 나중에 일괄 등록 | `cudaStreamIsCapturing` | Runtime |
| 캡처 도중에도 버퍼를 할당할 수 있게 Relaxed 모드로 전환 | `cudaThreadExchangeStreamCaptureMode` | Runtime |
| 버퍼 관리 | `cudaMalloc`, `cudaFree`, `cudaMemsetAsync`, `cudaMemcpy(Async)`, `cudaStreamSynchronize` | Runtime |
| P2P 가능 여부 사전 테스트 (`all_reduce_utils.py`) | `cudaSetDevice`, `cudaMemset`, `cudaDeviceSynchronize`, `cudaDeviceReset` | Runtime |
| Lamport 방식 fused allreduce+RMSNorm (MiniMax, cuda-python) | `cudaMalloc`, `cudaMemset`, `cudaMemcpy`, `cudaFree`, `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle` | Runtime |

### 3.4 PDL (Programmatic Dependent Launch, SM90 이상)

앞 커널이 끝나기 전에 다음 커널을 미리 띄워서, 커널 사이의 launch 지연을 겹치게 만드는 기법입니다.

| 위치 | API | 역할 |
|---|---|---|
| 호스트 | `cudaLaunchKernelEx` | `cudaLaunchAttributeProgrammaticStreamSerialization` 속성을 붙여 실행 (27곳) |
| 디바이스 | `cudaGridDependencySynchronize()` | 앞 커널의 결과가 필요한 지점에서 대기 |
| 디바이스 | `cudaTriggerProgrammaticLaunchCompletion()` | 뒤 커널이 일찍 시작할 수 있게 신호 |

사용 커널: fused QK-norm+RoPE, NVFP4·FP8 per-token-group 양자화, DSv3 fused A-GEMM, Kimi-K3 MLA KV concat 등

### 3.5 커널 튜닝 / 하드웨어 적응형 실행

GPU 사양에 맞춰 타일 크기, 그리드, shared memory를 고릅니다.

| 목적 | API | 주요 사용처 |
|---|---|---|
| SM 수·smem 한도·compute capability 조회 | `cudaGetDevice`, `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cudaDeviceGetAttribute` | Marlin, sampler 등 (SM 수에 맞춰 그리드 결정) |
| 대용량 dynamic shared memory 사용 | `cudaFuncSetAttribute` (`MaxDynamicSharedMemorySize`) | Marlin, selective_scan(Mamba), SM100 MLA, cooperative topk |
| Persistent 커널의 그리드 크기 결정 | `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaFuncGetAttributes` | SM100 MLA, topk, Marlin MoE |
| Cooperative / persistent topk 실행 | `cudaLaunchKernel` | `persistent_topk.cuh` |
| CUDA 버전에 따른 경로 선택 | `cudaRuntimeGetVersion` | NVFP4 scaled_mm |

### 3.6 양자화 커널 보조

| 기법 | API | 용도 |
|---|---|---|
| W4A8 (CUTLASS) | `cudaMemcpyToSymbol`, `cudaDeviceSynchronize` | nibble 변환 LUT를 constant memory에 올림 |
| FP8 blockwise (SM100) | `cudaMemcpy2DAsync` | scale 텐서를 정렬에 맞게 패딩 |
| FP8 공통 | `cudaMemsetAsync` | scale 버퍼 초기화 |

### 3.7 UVA 기반 CPU 가중치 오프로드

CPU 메모리를 GPU 주소로 직접 매핑해서, 복사 없이 GPU 커널이 접근하게 합니다 (`get_cuda_view_from_cpu_tensor`, `csrc/libtorch_stable/cuda_view.cu`).

| API | 용도 |
|---|---|
| `cudaHostAlloc` (`cudaHostAllocMapped`) | mapped pinned memory 할당 |
| `cudaHostGetDevicePointer` | 호스트 포인터에 대응하는 디바이스 포인터 획득 |
| `cudaFreeHost` | 해제 |
| `cudaMemcpy` | 초기 데이터 적재 |

### 3.8 모델별 공유 메모리 (Engram, DeepSeek-V4.1) / EC 커넥터

여러 프로세스(DP rank)가 CPU 메모리 영역을 공유하면서 GPU에서 직접 읽습니다.

| API | 용도 |
|---|---|
| `cudaHostRegister`, `cudaHostUnregister` | 공유 영역 pin / unpin |
| `cudaIpcCloseMemHandle` | IPC 핸들 정리 |

### 3.9 ROCm skinny GEMM (CUDA Graph 대응)

| API | 용도 |
|---|---|
| `cudaStreamGetCaptureInfo`, `cudaStreamSynchronize` | hipify되어 HIP 함수로 바뀌고, 그래프 캡처 중인지에 따라 동작을 바꿈 |

### 3.10 공통 / 부가

| 분류 | API |
|---|---|
| 에러 처리 | `cudaGetLastError`, `cudaGetErrorString`, `cuGetErrorString` |
| CI: GPU UUID 수집 | `cuInit`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetUuid_v2` |
| 벤치마크 프로파일 구간 지정 | `cudaProfilerStart`, `cudaProfilerStop` |
| CuTe DSL 커널 속성 확인 (벤치마크) | `cuFuncGetAttribute` |

---

## 4. 요약

| 최적화 기법 | Driver API | Runtime API |
|---|:---:|:---:|
| Sleep Mode (VMM) | ● (핵심) | ○ (백업 복사) |
| KV Cache 오프로드 | ● (`cuMemcpyBatchAsync`) | ● (폴백, pinning) |
| Custom AllReduce | ○ (`cuPointerGetAttribute`) | ● (IPC, graph 대응) |
| PDL | | ● |
| 커널 튜닝 | | ● |
| 양자화 보조 | | ● |
| UVA CPU 가중치 오프로드 | | ● |
| Engram / EC 공유 메모리 | | ● |

- Driver API는 거의 **Sleep Mode(VMM)**, **KV 오프로드(배치 복사)**, **Custom AllReduce(포인터 base 조회)** 세 곳에서만 씁니다.
- 나머지 최적화(PDL, 커널 튜닝, 양자화, UVA, IPC)는 Runtime API로 구현되어 있습니다.
- CUDA Graph, 스트림, 이벤트 관리는 vLLM이 직접 하지 않고 PyTorch에 맡깁니다. 그래서 위 목록에는 `cudaGraph*`나 `cudaEvent*`가 없습니다.
