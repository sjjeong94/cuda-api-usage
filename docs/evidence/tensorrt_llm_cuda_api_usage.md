# TensorRT-LLM CUDA API 사용 현황 및 최적화 기법 매핑

> 기준: TensorRT-LLM `main` 커밋 `5cf80d6` (2026-10-06)
>
> 방법: 저장소 전체(`3rdparty/`, `docs/` 제외)에서 `cuXxx(` / `cudaXxx(` 호출을 grep하고, 이번에는 **CUDA 공식 함수 목록과 교차 검증**했습니다. TensorRT-LLM에는 `cudaD2Dcpy`, `cudaCoreGemm`처럼 `cuda`로 시작하는 자체 헬퍼가 많아서, NVIDIA `cuda-python`(`8c66b43`)의 Runtime/Driver 바인딩 선언(`cyruntime.pxd`, `cydriver.pxd`)에 있는 이름만 남겼습니다. 바인딩에 없는 C++ 템플릿 API(`cudaLaunchKernelEx` 등)와 디바이스 측 API는 직접 확인해 포함했습니다. `cuGetProcAddress`로 이름을 찾아 호출하는 API도 포함했습니다.
> 범위: TensorRT-LLM 저장소가 **직접** 호출하는 API만 포함. CUTLASS, DeepEP, DeepGEMM(외부), FlashMLA, NCCL, UCX, NIXL, PyTorch 등 외부 의존성 내부 호출은 제외

---

## 목차

0. [TensorRT-LLM의 CUDA 코드 구성](#0-tensorrt-llm의-cuda-코드-구성)
1. [CUDA Driver API](#1-cuda-driver-api)
2. [CUDA Runtime API](#2-cuda-runtime-api)
3. [최적화 기법별 API 매핑](#3-최적화-기법별-api-매핑)
4. [요약](#4-요약)

---

## 0. TensorRT-LLM의 CUDA 코드 구성

| 구성 요소 | 위치 | 역할 |
|---|---|---|
| **C++ 런타임** | `cpp/tensorrt_llm/{batch_manager,runtime,executor,common}` | KV cache manager(v1/v2), 메모리 관리, disaggregated serving 전송, NVLS·MNNVL 메모리 |
| **커널** | `cpp/tensorrt_llm/kernels/` | 손으로 작성한 커널 + 미리 빌드한 CUBIN 로더(FMHA v2, trtllmGen) + JIT(XQA, DeepGEMM) |
| **PyTorch 연동** | `cpp/tensorrt_llm/thop/`, `nanobind/` | PyTorch custom op, Python 바인딩 |
| **Python 백엔드** | `tensorrt_llm/_torch/` 등 | PyTorch 기반 실행기, DWDP, MNNVL, CuTe DSL 커널 |
| **커널 생성기·테스트** | `cpp/kernels/{fmha_v2,xqa}`, `cpp/tests` | FMHA 커널 생성기와 단독 테스트 (집계에서 테스트로 분류) |

**분석하지 못한 부분 (한계)**
- Git LFS로 배포되는 **소스 없는 정적 라이브러리**는 들여다보지 못했습니다. 익명 git 경로로는 LFS 객체를 받을 수 없습니다.
  - `kernels/internal_cutlass_kernels/*/tensorrt_llm_internal_cutlass_kernels_static.tar.xz` (64 MB)
  - `kernels/trtllmGenKernels/fmha/lib/*/libTrtLlmGen.a`, `libTrtLlmGenFmhaLib.a`
- 미리 빌드한 CUBIN 9,322개(`*.cubin.tar.zst`)는 커널 바이너리입니다. 이것들을 **로드하고 실행하는 코드**는 소스에 있으므로 분석에 포함했습니다.

**공통 Driver 래퍼.** `common/cudaDriverWrapper.{h,cpp}`가 `libcuda`를 `dlopen`해서 25개 함수를 노출합니다. 그중 7개는 **선언만 있고 어디서도 호출하지 않아** 집계에서 뺐습니다: `cuLinkCreate`, `cuLinkAddData`, `cuLinkAddFile`, `cuLinkComplete`, `cuLinkDestroy`, `cuLaunchCooperativeKernel`, `cuOccupancyMaxActiveClusters`.

---

## 1. CUDA Driver API

본체(C++ + Python)에서 **84개**를 사용합니다. `_v2` 접미사는 같은 API로 합쳤고, 테스트·생성기 전용 12개는 따로 적었습니다.

### 1.1 커널 로딩 / 실행

미리 빌드한 CUBIN과 JIT 컴파일한 CUBIN을 Driver API로 직접 로드합니다.

| API | 주요 위치 | 용도 |
|---|---|---|
| `cuModuleLoadData` / `cuModuleGetFunction` / `cuModuleUnload` | `contextFusedMultiHeadAttention/fused_multihead_attention_v2.cpp`, `trtllmGenKernels/{fmha,gemm,batchedGemm,gemmGatedAct}` | **FMHA v2·trtllmGen의 미리 빌드한 CUBIN** 로드 |
| `cuLibraryLoadData` / `cuLibraryGetKernel` / `cuLibraryUnload` / `cuLibraryGetGlobal` | `decoderXQAImplJIT/cubinObj.cpp`, `decoderXQARunnerUtils.h` | **XQA JIT 커널** (NVRTC로 컴파일한 CUBIN) 로드 |
| `cuLibraryLoadData` / `cuLibraryEnumerateKernels` / `cuLibraryGetKernelCount` / `cuKernelGetName` / `cuLibraryUnload` | `include/tensorrt_llm/deep_gemm/runtime.cuh` | **DeepGEMM JIT 커널** 로드 |
| `cuKernelSetAttribute` | `cubinObj.cpp` | XQA 커널 smem 설정 |
| `cuFuncSetAttribute` | FMHA v2, trtllmGen, CuTe DSL top-k (Python) | dynamic smem 설정 |
| `cuLaunchKernel` | FMHA v2, `kvCacheManagerV2Utils.cu` | 커널 실행 (KV v2는 `cudaGetKernel`로 얻은 런타임 커널을 Driver로 실행) |
| `cuLaunchKernelEx` | XQA, trtllmGen `CudaKernelLauncher.h` | cluster·PDL 속성을 붙여 실행 |
| `cuCtxGetId` | trtllmGen `GemmInterface.h` 등 | 컨텍스트별 모듈 캐시 키 |
| `cuOccupancyMaxPotentialClusterSize` | CuTe DSL top-k (Python) | cluster 크기 결정 |

### 1.2 TMA

| API | 주요 위치 |
|---|---|
| `cuTensorMapEncodeTiled` | XQA `tensorMapUtils.cpp`, trtllmGen `TmaDescriptor.h`·`KernelParams.h`, Mamba `selectiveScan/{bmmchunk,chunkscan,chunkstate}.h`, `mhcFusedHcKernel.cu`, `tinygemm2_cuda.cu`, triattention (Python) |

### 1.3 VMM (가상 메모리) / 공유 핸들

| API | 주요 위치 |
|---|---|
| `cuMemAddressReserve` / `cuMemAddressFree` | `runtime/virtualMemory.h`, `kv_cache_manager_v2/cudaVirtMem.cpp`, `cacheTransBuffer.cpp`, `userbuffers`, `moeAlltoAllCftManager.h`, `_mnnvl_utils.py`, `dwdp/vmm.py` |
| `cuMemCreate` / `cuMemRelease` | 위와 같음 + `cudaUtils.h` |
| `cuMemMap` / `cuMemUnmap` / `cuMemSetAccess` | 위와 같음 |
| `cuMemGetAllocationGranularity` | 위와 같음 |
| `cuMemExportToShareableHandle` / `cuMemImportFromShareableHandle` | `ipcNvlsMemory.cu`, `mcastDeviceMemory.cpp`, `userbuffers-host.cpp`, `_mnnvl_utils.py`, `dwdp/transport.py` |
| `cuMemGetAddressRange` | `executor/cache_transmission/transferAgent.cpp` (전송 등록할 메모리 범위 조회) |
| `cuCtxSynchronize` | `kv_cache_manager_v2/cudaVirtMem.cpp` |
| `cuDeviceTotalMem` | `kv_cache_manager_v2/storage/core.cpp` |

### 1.4 NVLink SHARP 멀티캐스트 (NVLS)

| API | 주요 위치 |
|---|---|
| `cuMulticastCreate` / `cuMulticastAddDevice` / `cuMulticastBindMem` / `cuMulticastUnbind` / `cuMulticastGetGranularity` | `runtime/mcastDeviceMemory.cpp`, `runtime/ipcNvlsMemory.cu`, `kernels/userbuffers/userbuffers-host.cpp`, `virtualMemory.h` |

### 1.5 Logical Endpoint (`cuGetProcAddress`로 해석)

`kernels/moe/communication/moeAlltoAllCftManager.h`. 코드 주석상 CUDA 13.4 헤더와 LE를 지원하는 드라이버, IMEX 데몬, NVSwitch fabric이 필요합니다.

| API | 용도 |
|---|---|
| `cuLogicalEndpointIdReserve` / `cuLogicalEndpointIdRelease` | LE ID 예약 / 반환 |
| `cuLogicalEndpointCreate` / `cuLogicalEndpointDestroy` | peer rank마다 unicast LE 생성 / 해제 |
| `cuLogicalEndpointBindMem` / `cuLogicalEndpointUnbind` | LE를 MNNVL workspace의 수신 버퍼에 연결 |
| `cuLogicalEndpointExport` / `cuLogicalEndpointImport` | rank 간 LE 교환 |
| `cuLogicalEndpointQuery` | LE 상태 조회 |
| `cuGetProcAddress`, `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent`, `cuDeviceGet` | API 해석과 컨텍스트 준비 |

### 1.6 Green Context / Locality Domain (`cuGetProcAddress`로 해석)

`runtime/locality_domain/locality_domain_utils.cpp`

| API | 용도 |
|---|---|
| `cuDeviceGetDevResource` / `cuDevSmResourceSplit` / `cuDevResourceGenerateDesc` | SM 리소스를 locality domain별로 분할 |
| `cuGreenCtxCreate` / `cuGreenCtxDestroy` / `cuGreenCtxStreamCreate` | domain별 green context와 스트림 |
| `cuInit`, `cuDeviceGet`, `cuDeviceGetAttribute`, `cuMemAlloc`, `cuMemFree`, `cuStreamDestroy` | 초기화, 지원 확인, domain 메모리 |

### 1.7 KV Cache Manager v2 (Driver API로 작성)

`batch_manager/kv_cache_manager_v2/`는 스트림·이벤트·호스트 메모리까지 Driver API로 다룹니다.

| API | 위치 | 용도 |
|---|---|---|
| `cuEventCreate` / `cuEventRecord` / `cuEventQuery` / `cuEventSynchronize` / `cuEventDestroy` | `utils/cudaEvent.cpp` | 이벤트 래퍼 |
| `cuStreamCreate` / `cuStreamWaitEvent` / `cuStreamSynchronize` / `cuStreamDestroy` | `utils/cudaEvent.cpp`, `stagingBuffer.cpp` | 스트림 래퍼 |
| `cuMemHostRegister` / `cuMemHostUnregister` | `utils/hostMem.cpp` | host 계층 KV 메모리 pin |
| `cuMemAlloc` / `cuMemFree` | `stagingBuffer.cpp` | 스테이징 버퍼 |
| `cuMemcpyBatchAsync` / `cuMemcpyAsync` | `batchedPageCopy.cu` | 페이지 일괄 복사 (copy engine 경로) |
| `cuLaunchHostFunc` | `kvCacheManagerV2Utils.cpp` | 스트림 순서대로 호스트 콜백 |

### 1.8 메모리 복사·초기화 / 디바이스·컨텍스트

| API | 주요 위치 |
|---|---|
| `cuMemcpyHtoD` / `cuMemcpyHtoDAsync` / `cuMemcpyDtoH` | `runtime/virtualMemory.cpp` (sleep 시 백업·복원) |
| `cuMemcpyDtoD` | `dwdp/vmm.py` |
| `cuMemsetD8` / `cuMemsetD8Async` / `cuMemsetD32` | `mcastDeviceMemory.cpp`, `virtualMemory.h`, `fusedMoeCommKernels.cu` |
| `cuMemGetInfo` | `pyexecutor/trace_log_utils.py` |
| `cuCtxGetCurrent` / `cuCtxGetDevice` | `opUtils.cpp`, trtllmGen, `cudaVirtMem.cpp`, `_mnnvl_utils.py`, fp4 MLA (Python) |
| `cuDeviceGetAttribute` | `cacheTransBuffer.cpp`, `ub_interface.cpp`, `ipcNvlsMemory.cu`, `mcastDeviceMemory.cpp`, `allreduceOp.cpp`, `autotuner.py` |
| `cuGetErrorString` / `cuGetErrorName` | `TLLM_CU_CHECK` 등 |

### 1.9 테스트·생성기에서만 사용

| API | 위치 |
|---|---|
| `cuCtxCreate`, `cuModuleLoad`, `cuModuleGetGlobal` | `cpp/kernels/fmha_v2`, `cpp/kernels/xqa` 테스트 |
| `cuEventElapsedTime`, `cuGraphKernelNodeGetParams`, `cuMemHostAlloc`, `cuMemFreeHost`, `cuMemHostGetDevicePointer`, `cuMemcpyDtoHAsync`, `cuMemRetainAllocationHandle`, `cuMemGetAllocationPropertiesFromHandle`, `cuPointerGetAttribute` | `cpp/tests`, `tests/` |

---

## 2. CUDA Runtime API

본체에서 **호스트 API 72개와 디바이스 측 API 2개**를 사용합니다. C++ 68개(템플릿 API 5개 포함)와 Python 전용 4개입니다.

### 2.1 C++ (`cpp/tensorrt_llm`, `cpp/include`)

#### 디바이스 / 속성

| API | 비고 |
|---|---|
| `cudaGetDevice` (81곳), `cudaSetDevice` (33곳), `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cudaDeviceGetAttribute` (52곳) | |
| `cudaDeviceGetPCIBusId` | `batchedPageCopy.cu` (NVML 장치와 매칭해 링크 종류 판별) |
| `cudaDeviceCanAccessPeer` | `ipcUtils.cpp`, `allreduceOp.cpp` |
| `cudaDeviceGetStreamPriorityRange` | `nixl_utils/bounce/ExecPool.cpp` |
| `cudaDriverGetVersion`, `cudaMemGetInfo`, `cudaDeviceSynchronize` | |

#### 커널 설정 / 실행

| API | 비고 |
|---|---|
| `cudaFuncSetAttribute` (80곳), `cudaFuncGetAttributes`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (15곳) | |
| `cudaLaunchKernelEx` (105곳), `cudaLaunchKernelExC` (12곳) | PDL·cluster 속성 |
| `cudaLaunchKernel` | `common/sageQuant.cu` |
| `cudaLaunchCooperativeKernel` | `moe/loadBalance/moeLoadBalanceKernels.cu` |
| `cudaGetKernel` | `kvCacheManagerV2Utils.cu` (Driver로 실행할 커널 핸들 획득) |
| `cudaGetDriverEntryPoint` / `cudaGetDriverEntryPointByVersion` | DeepGEMM `tma_utils.cuh`, `fp8_blockscale_tma_utils.cuh` (`cuTensorMapEncodeTiled` 해석) |

#### 메모리 할당

| API | 주요 위치 | 용도 |
|---|---|---|
| `cudaMalloc` / `cudaFree` | 다수 | |
| `cudaMallocAsync` / `cudaFreeAsync` | `runtime/tllmBuffers.h`, `moeAlignKernels.cu` | 스트림 순서 할당 |
| `cudaMemPoolCreate` / `cudaMemPoolSetAttribute` / `cudaMemPoolGetAttribute` / `cudaMemPoolTrimTo` / `cudaMemPoolDestroy` | `runtime/cudaMemPool.cpp` | 전용 메모리 풀 |
| `cudaMallocManaged` / `cudaMemAdvise` | `tllmBuffers.h`, `moeLoadBalancer/hostAccessibleDeviceAllocator.cpp` | CPU에서도 접근하는 메모리 |
| `cudaMallocHost` / `cudaHostAlloc` / `cudaFreeHost` | `tllmBuffers.h`, `ExecPool.cpp` | pinned host 메모리 |
| `cudaHostRegister` / `cudaHostUnregister` | `hostAccessibleDeviceAllocator.cpp`, `topologyDetector.cpp` | 기존 host 메모리 pin |
| `cudaHostGetDevicePointer` | `ExecPool.cpp` | mapped host 메모리 |
| `cudaPointerGetAttributes` | `cudaUtils.h`, `iBuffer.cpp`, `torchUtils.h` | 포인터 종류 판별 |

#### 메모리 복사 / 초기화

| API | 주요 위치 |
|---|---|
| `cudaMemcpy` (22곳), `cudaMemcpyAsync` (25곳), `cudaMemset` (11곳), `cudaMemsetAsync` (39곳) | 다수 |
| `cudaMemcpy2DAsync` | `kvCacheTransferManager.cpp`, `moeLoadBalancer.cpp`, `inplaceSliceCopyOp.cpp` |
| `cudaMemcpyBatchAsync` | `thop/asyncUlyssesOp.cpp` |
| `cudaMemcpyToSymbol` | DeepGEMM `nvrtc_cutlass.cuh` (CUTLASS synclog 디버그 버퍼) |

#### IPC

| API | 주요 위치 |
|---|---|
| `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` / `cudaIpcCloseMemHandle` | `runtime/ipcUtils.cpp` |

#### 스트림 / 이벤트 / 콜백

| API | 주요 위치 |
|---|---|
| `cudaStreamCreateWithFlags`, `cudaStreamCreateWithPriority`, `cudaStreamDestroy`, `cudaStreamSynchronize`, `cudaStreamQuery`, `cudaStreamWaitEvent` | `include/tensorrt_llm/runtime/cudaStream.h`, `ExecPool.cpp`, `BounceTransport.cpp` |
| `cudaStreamIsCapturing` | `ncclUtils.cpp`, `cudaUtils.h`, `fmhaKernels.h`, `asyncUlyssesOp.cpp` |
| `cudaEventCreate`, `cudaEventCreateWithFlags`, `cudaEventRecord`, `cudaEventQuery`, `cudaEventSynchronize`, `cudaEventDestroy` | |
| `cudaLaunchHostFunc` / `cudaLaunchHostFunc_v2` | `nanobind/runtime/hostfunc.cpp` (스트림에서 Python 함수 호출, GIL 획득) |
| `cudaStreamAddCallback` | `common/memoryUtils.cu` |

#### 에러 처리

| API | 비고 |
|---|---|
| `cudaGetLastError` (79곳), `cudaGetErrorString` (40곳), `cudaPeekAtLastError` | |

#### 디바이스 측 API (PDL)

| API | 사용 횟수 |
|---|---|
| `cudaGridDependencySynchronize` | 154곳 |
| `cudaTriggerProgrammaticLaunchCompletion` | 143곳 |

### 2.2 Python (`tensorrt_llm/`)

C++에도 나오는 API를 제외한 Python 전용 API는 4개입니다.

| API | 위치 |
|---|---|
| `cudaStreamGetCaptureInfo` | `pyexecutor/breakable_cuda_graph/breakable_cuda_graph.py` |
| `cudaStreamCreate` | `disaggregation/native/bounce/impl.py` |
| `cudaProfilerStart` / `cudaProfilerStop` | `pyexecutor/profiling.py`, `visual_gen/profiler.py` |

그 밖의 Python 사용처: `_ipc_utils.py`(`cudaIpc*`, `cudaDeviceCanAccessPeer`), `dwdp/weight_manager.py`(`cudaMemcpyBatchAsync`), `qwen4_exp/ple.py`(`cudaHostGetDevicePointer`)

### 2.3 테스트에서만 사용

`cudaGraphCreate`, `cudaGraphInstantiate`, `cudaGraphLaunch`, `cudaGraphDestroy`, `cudaGraphExecDestroy`, `cudaGraphGetNodes`, `cudaGraphNodeGetType`, `cudaGraphKernelNodeGetParams`, `cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaDeviceGetDefaultMemPool`, `cudaDeviceGetMemPool`, `cudaDeviceReset`, `cudaEventElapsedTime`, `cudaEventRecordWithFlags`, `cudaGetErrorName`, `cudaMemPrefetchAsync`

본체에는 `cudaGraph*`나 스트림 캡처 API가 없습니다. CUDA Graph는 PyTorch(`torch.cuda.CUDAGraph`)로 캡처합니다.

---

## 3. 최적화 기법별 API 매핑

표기: **[C]** C++ 런타임, **[K]** 커널, **[P]** Python 백엔드

### 3.1 미리 빌드한 CUBIN / JIT 커널 로딩 [K]

TensorRT-LLM의 핵심 커널(FMHA, trtllmGen GEMM·MoE·FMHA)은 NVIDIA가 미리 컴파일해 CUBIN으로 배포하고, XQA와 DeepGEMM은 실행 시점에 모델 설정에 맞춰 NVRTC로 컴파일합니다. 어느 쪽이든 Driver API로 로드합니다.

| 경로 | API |
|---|---|
| 미리 빌드한 CUBIN (FMHA v2, trtllmGen) | `cuModuleLoadData`, `cuModuleGetFunction`, `cuFuncSetAttribute`, `cuLaunchKernel` / `cuLaunchKernelEx`, `cuModuleUnload`, `cuCtxGetId` (컨텍스트별 캐시) |
| XQA JIT (NVRTC) | `cuLibraryLoadData`, `cuLibraryGetKernel`, `cuKernelSetAttribute`, `cuLibraryGetGlobal`, `cuLaunchKernelEx`, `cuLibraryUnload` |
| DeepGEMM JIT | `cuLibraryLoadData`, `cuLibraryEnumerateKernels`, `cuLibraryGetKernelCount`, `cuKernelGetName`, `cuLibraryUnload` |

### 3.2 PDL (Programmatic Dependent Launch) [K]

다섯 프로젝트 중 **가장 광범위하게** 씁니다.

| 위치 | API | 규모 |
|---|---|---|
| 호스트 | `cudaLaunchKernelEx`, `cudaLaunchKernelExC`, `cuLaunchKernelEx` | 117곳 이상 |
| 디바이스 | `cudaGridDependencySynchronize`, `cudaTriggerProgrammaticLaunchCompletion` | 154곳, 143곳 |

### 3.3 TMA [K]

| 사용처 | API |
|---|---|
| XQA, trtllmGen(GEMM·FMHA·gated act), Mamba selective scan, mHC, tinygemm2, triattention | `cuTensorMapEncodeTiled` |
| DeepGEMM, FP8 blockscale GEMM | `cudaGetDriverEntryPoint(ByVersion)` → `cuTensorMapEncodeTiled` |

### 3.4 KV Cache Manager v2 [C]

prefix 재사용(radix tree), eviction, GPU/host/disk 계층 이동을 담당하는 새 KV 관리자입니다. 스트림·이벤트·메모리를 모두 Driver API로 다룹니다.

| 목적 | API |
|---|---|
| GPU 계층 물리 메모리 풀 (VMM) | `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, `cuCtxSynchronize` |
| host 계층 pin | `cuMemHostRegister`, `cuMemHostUnregister` |
| **링크별 페이지 복사 전략** | NVLink-C2C(Grace)에서는 SM 기반 복사 커널(`cudaGetKernel` → `cuLaunchKernel`), PCIe에서는 copy engine 배치 복사(`cuMemcpyBatchAsync`, SM 사용 0). 링크 종류는 `cudaDeviceGetPCIBusId`로 NVML 장치와 매칭해 판별. CUDA 12.8 미만은 `cuMemcpyAsync` 반복 |
| 스테이징 | `cuMemAlloc`, `cuMemFree`, `cuStreamSynchronize` |
| 비동기 순서 관리 | `cuEvent*`, `cuStream*`, `cuLaunchHostFunc` |

### 3.5 Sleep / Wake (메모리 해제·복원) [C]

RL 등에서 GPU 메모리를 내려놓았다가 다시 올립니다(`release_memory` / `resume_memory`, `sleep(sleep_tags)`). `CUDAVirtualMemoryChunk`가 물리 메모리를 해제·재생성하면서 가상 주소를 유지합니다.

| 단계 | API |
|---|---|
| 해제·재생성 | `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, `cuMemUnmap`, `cuMemRelease` |
| 내용 백업·복원 | `cuMemcpyDtoH`, `cuMemcpyHtoD`, `cuMemcpyHtoDAsync` |
| 초기화 | `cuMemsetD8Async` |
| 멀티캐스트 객체 재바인딩 | `cuMulticastBindMem`, `cuMulticastUnbind` |

### 3.6 NVLS 멀티캐스트 AllReduce / UserBuffers [C] [K]

NVSwitch의 NVLink SHARP로 한 번 쓰면 모든 GPU에 브로드캐스트되는 멀티캐스트 메모리를 만들어 AllReduce를 가속합니다. UserBuffers는 GEMM과 통신을 겹치는 데 씁니다.

| 단계 | API |
|---|---|
| 멀티캐스트 객체 | `cuMulticastGetGranularity`, `cuMulticastCreate`, `cuMulticastAddDevice`, `cuMulticastBindMem` |
| 물리 메모리 공유 | `cuMemCreate`, `cuMemExportToShareableHandle`, `cuMemImportFromShareableHandle`, `cuMemMap`, `cuMemSetAccess` |
| 지원 확인 | `cuDeviceGetAttribute` (멀티캐스트 지원 여부) |
| 초기화 | `cuMemsetD8` |

### 3.7 MNNVL (Multi-Node NVLink) / Fabric 메모리 [C] [P]

GB200 NVL72처럼 여러 노드가 NVLink로 묶인 시스템에서, fabric 핸들로 노드 간 GPU 메모리를 직접 매핑합니다.

| 기능 | API |
|---|---|
| MNNVL 메모리 (`_mnnvl_utils.py`) | `cuMemCreate` (fabric 핸들), `cuMemExportToShareableHandle`, `cuMemImportFromShareableHandle`, `cuMemAddressReserve`, `cuMemMap`, `cuMemSetAccess` |
| Disaggregated serving KV 전송 버퍼 (`cacheTransBuffer.cpp`의 `FabricMemory`) | 위와 같은 VMM API + `cuDeviceGetAttribute` |
| 전송 에이전트 메모리 등록 | `cuMemGetAddressRange` |
| **MoE AlltoAll over CFT (Logical Endpoint)** | `cuLogicalEndpoint*` 9종. rank마다 peer별 unicast LE를 만들어 수신 버퍼에 연결하고, dispatch 커널이 fabric counted write로 상대 LE에 직접 씀 |

### 3.8 DWDP (Distributed Weight Data Parallelism) [P]

MNNVL 위에서 MoE 전문가 가중치를 rank 간에 나눠 두고, 필요한 가중치를 composite VA에 매핑합니다. SGLang의 DWDP와 같은 개념입니다.

| API |
|---|
| `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, `cuMemExportToShareableHandle`, `cuMemImportFromShareableHandle`, `cuMemcpyDtoD`, `cudaMemcpyBatchAsync` (`weight_manager.py`) |

### 3.9 Locality Domain 분할 [C] [P]

GPU를 locality domain(0, 1) 두 개로 나눠, domain마다 SM 파티션(green context), 스트림, 메모리 할당기를 둡니다.

| API |
|---|
| `cuDeviceGetDevResource`, `cuDevSmResourceSplit`, `cuDevResourceGenerateDesc`, `cuGreenCtxCreate`, `cuGreenCtxStreamCreate`, `cuGreenCtxDestroy`, `cuMemAlloc` |

### 3.10 MoE Expert Load Balancer (EPLB) [C] [K]

전문가 배치를 실행 중에 재조정합니다. CPU가 GPU 통계를 직접 읽을 수 있는 메모리를 쓰고, Grace에서는 NUMA 바인딩과 THP를 적용합니다.

| 목적 | API |
|---|---|
| CPU·GPU 공용 메모리 | `cudaMallocManaged`, `cudaMemAdvise`, `cudaHostRegister` |
| 가중치 재배치 복사 | `cudaMemcpy2DAsync` |
| grid-wide 동기화 커널 | `cudaLaunchCooperativeKernel` |

### 3.11 Custom AllReduce (IPC) / Ulysses 시퀀스 병렬 [C]

| 기법 | API |
|---|---|
| IPC 버퍼 공유 | `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle`, `cudaIpcCloseMemHandle`, `cudaDeviceCanAccessPeer` |
| 확산 모델용 비동기 Ulysses all-to-all (`asyncUlyssesOp.cpp`) | `cudaMemcpyBatchAsync` (copy engine으로 push, 별도 통신 스트림), `cudaStreamIsCapturing` |

### 3.12 Disaggregated Serving 전송 (NIXL bounce) [C] [P]

| 목적 | API |
|---|---|
| 높은 우선순위 전송 스트림 | `cudaDeviceGetStreamPriorityRange`, `cudaStreamCreateWithPriority` |
| bounce 버퍼 (mapped pinned) | `cudaHostAlloc`, `cudaHostGetDevicePointer` |
| 완료 확인 | `cudaStreamQuery` |

### 3.13 메모리 할당기 [C]

| 기법 | API |
|---|---|
| 전용 메모리 풀 + 스트림 순서 할당 | `cudaMemPoolCreate`, `cudaMemPoolSetAttribute`, `cudaMallocAsync`, `cudaFreeAsync`, `cudaMemPoolTrimTo` |
| pinned host 버퍼 | `cudaMallocHost`, `cudaHostAlloc` |

### 3.14 CUDA Graph 보조 [C] [P]

| 기법 | API |
|---|---|
| 스트림에서 Python 콜백 실행 (그래프 안의 host 작업) | `cudaLaunchHostFunc`, `cudaLaunchHostFunc_v2` |
| Breakable CUDA Graph | `cudaStreamGetCaptureInfo` |
| 캡처 중 동작 분기 | `cudaStreamIsCapturing` (NCCL, FMHA, Ulysses) |

### 3.15 공통 / 부가

| 분류 | API |
|---|---|
| 커널 튜닝 | `cudaFuncSetAttribute`, `cudaFuncGetAttributes`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaDeviceGetAttribute` |
| 에러 처리 | `cudaGetLastError`, `cudaGetErrorString`, `cudaPeekAtLastError`, `cuGetErrorString`, `cuGetErrorName` |
| 프로파일링 | `cudaProfilerStart`, `cudaProfilerStop` [P] |
| 디버그 | `cudaMemcpyToSymbol` (CUTLASS synclog) |

---

## 4. 요약

| 최적화 기법 | 위치 | Driver API | Runtime API |
|---|:---:|:---:|:---:|
| 미리 빌드한 CUBIN / JIT 커널 로딩 | K | ● (핵심) | |
| PDL | K | ○ | ● (최다) |
| TMA | K | ● | ○ (entry point) |
| KV Cache Manager v2 | C | ● (핵심) | ○ |
| Sleep / Wake | C | ● | |
| NVLS 멀티캐스트 / UserBuffers | C, K | ● | |
| MNNVL / Fabric / Logical Endpoint | C, P | ● | |
| DWDP | P | ● | ○ |
| Locality Domain (green context) | C, P | ● | |
| MoE EPLB | C, K | | ● |
| Custom AllReduce / Ulysses | C | | ● |
| Disagg NIXL bounce | C, P | | ● |
| 메모리 풀 | C | | ● |
| CUDA Graph 보조 | C, P | | ● |

- TensorRT-LLM은 Driver API **84개**로 다섯 프로젝트 중 가장 많이 씁니다. 하드웨어 기능에 가장 가까이 붙어 있습니다.
  - NVLS 멀티캐스트, MNNVL fabric 메모리, CUDA 13.4 Logical Endpoint처럼 **NVIDIA 시스템 전용 기능**을 직접 다룹니다.
  - 핵심 커널을 **미리 빌드한 CUBIN**으로 배포하고 Driver API로 로드합니다.
- Runtime API 쪽은 **PDL 사용량이 압도적**입니다(디바이스 측 API 약 300곳).
- CUDA Graph 캡처 자체는 PyTorch에 맡기고, Runtime의 graph API는 테스트에서만 씁니다.

### 다섯 프로젝트 비교

| 항목 | vLLM | Ollama | ExecuTorch | SGLang | TensorRT-LLM |
|---|---|---|---|---|---|
| Driver API 개수 | 16 | 26 (본체 8 + llama.cpp·MLX) | 0 (+ AOTI 생성 코드 7, 조건부 2) | 46 | **84** |
| Runtime API 개수 (호스트) | 35 | 48 (llama.cpp) / 55 (MLX) | 37 (+ AOTI 19, HIP 전용 1) | 44 | **72** |
| 커널 공급 방식 | 손으로 작성 + 외부 라이브러리 | 손으로 작성 + JIT (MLX) | AOT 생성 (Inductor) | 손으로 작성 (AOT + JIT) | **미리 빌드한 CUBIN** + 손으로 작성 + JIT (XQA, DeepGEMM) |
| VMM 활용 | sleep mode | 메모리 풀 | 없음 | KV arena, 가중치·feature 공유 | KV v2, sleep, NVLS, MNNVL, DWDP |
| 멀티 GPU 메모리 | IPC | P2P, PCIe AllReduce | 없음 | IPC, VMM 공유 핸들 | IPC, **NVLS 멀티캐스트, MNNVL fabric, Logical Endpoint** |
| SM 분할 | 없음 | 없음 | 호출자가 스트림 전달 | green context (PD mux) | green context (locality domain) |
| PDL | 있음 (약 70곳) | 있음 (llama.cpp, 기본 켜짐) | 없음 | 있음 | **있음 (약 300곳)** |
| CUDA Graph | PyTorch에 위임 | 직접 | 직접 | PyTorch + Driver로 분석·dedup | PyTorch에 위임 |
