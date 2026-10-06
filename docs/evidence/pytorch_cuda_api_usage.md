# PyTorch CUDA API 사용 현황 및 최적화 기법 매핑

> 기준: PyTorch `v2.14.0` (`2b3ec34`, 2026-08-26). ExecuTorch 분석에서 AOTInductor를 볼 때 쓴 커밋과 같습니다
>
> 방법: `aten/`, `c10/`, `torch/`에서 `cuXxx(` / `cudaXxx(` 호출을 찾고, 주석·문자열·`#if 0` 코드를 걸러 냈습니다. 함수 이름은 NVIDIA cuda-python `8c66b43`(CUDA 12.9~13.4)의 바인딩 선언(`cydriver.pxd`, `cyruntime.pxd`)과 대조했고, 바인딩에 없는 C++ 템플릿 API는 따로 포함했습니다. TensorRT-LLM 분석과 같은 방법입니다. 다음 호출 경로도 포함했습니다.
> - C++ Driver 테이블: `c10::cuda::DriverAPI::get()->cuXxx_(...)`
> - Lazy NVRTC 테이블: `at::globalContext().getNVRTC().cuXxx(...)`
> - Python `ctypes`로 연 `libcuda`
> - Python cuda-python 바인딩 (`_drv.cuXxx`, `_rt.cudaXxx`)
>
> 범위: PyTorch가 **직접** 호출하는 API만 다룹니다. cuBLAS, cuDNN, cuFFT, NCCL, NVSHMEM, CUTLASS, Triton 내부 호출은 제외했습니다. 경로에 `hip`이 들어간 파일은 제외했고, 파일 안의 `USE_ROCM` 분기는 구분하지 않았습니다.
>
> 원천 데이터: [`docs/reference/data/api_usage.csv`](../reference/data/api_usage.csv)의 `framework=pytorch` 행
>
> 재현 방법: 아래 명령은 PyTorch와 cuda-python을 sparse clone하고, 스캔해서 `framework=pytorch` 행을 다시 만듭니다. 제외 규칙(CUPTI 콜백 이름 문자열, CUTLASS mock 모듈, LazyNVRTC 스텁 정의)은 `scripts/pytorch/to_csv.py`에 있습니다.
> ```sh
> git clone --depth 1 --branch v2.14.0 --filter=blob:none --sparse https://github.com/pytorch/pytorch.git /tmp/src/pytorch
> git -C /tmp/src/pytorch sparse-checkout set aten c10 torch
> git clone --depth 1 --filter=blob:none --sparse https://github.com/NVIDIA/cuda-python.git /tmp/src/cuda-python
> git -C /tmp/src/cuda-python sparse-checkout set --no-cone '**/cydriver.pxd' '**/cyruntime.pxd'
> python3 -I scripts/pytorch/scan.py /tmp/src/pytorch /tmp/src/cuda-python /tmp/pt_scan.json
> python3 -I scripts/pytorch/to_csv.py /tmp/pt_scan.json docs/reference/data/api_usage.csv /tmp/pt_locs.json
> python3 scripts/gen_api_catalog.py
> ```

---

## 목차

0. [PyTorch의 CUDA 코드 구성](#0-pytorch의-cuda-코드-구성)
1. [CUDA Driver API](#1-cuda-driver-api)
2. [CUDA Runtime API](#2-cuda-runtime-api)
3. [최적화 기법별 API 매핑](#3-최적화-기법별-api-매핑)
4. [서빙 프레임워크가 PyTorch에 맡기는 것](#4-서빙-프레임워크가-pytorch에-맡기는-것)
5. [요약](#5-요약)

---

## 0. PyTorch의 CUDA 코드 구성

| 구성 요소 (CSV `component`) | 위치 | 역할 | Driver | Runtime |
|---|---|---|---:|---:|
| **c10** | `c10/cuda/` | 디바이스·스트림·이벤트, **CUDACachingAllocator**, cudaMallocAsync 백엔드, Driver API 테이블, 그래프 캡처 유틸 | 13 | 56 |
| **aten** | `aten/src/ATen/cuda/`, `aten/src/ATen/native/**/cuda/` | CUDA context, **CUDAGraph**, pinned host 할당기, jiterator(NVRTC), 커널 | 8 | 39 |
| **torch.cuda** | `torch/csrc/cuda/`, `torch/cuda/*.py`, `torch/csrc/CudaIPCTypes.cpp` 등 | Python API: `torch.cuda.cudart()`, pluggable allocator, green context, 그래프 분석, UVM, `_compile_kernel` | 22 | 35 |
| **c10d** | `torch/csrc/distributed/c10d/` | ProcessGroupNCCL, **Symmetric Memory** (VMM·멀티캐스트) | 19 | 11 |
| **inductor** | `torch/csrc/inductor/`, `torch/_inductor/codegen/aoti_runtime/*.h` | `torch.compile`의 **static launcher**(Triton 커널 실행), AOTInductor 런타임 | 14 | 19 |
| **inductor-gen** | `torch/_inductor/codegen/cuda/device_op_overrides.py`, `cpp_wrapper_gpu.py` | Inductor가 **생성하는** C++ wrapper 코드 (문자열 템플릿) | 6 (+ 조건부 2) | 3 (+ HIP 전용 1) |
| **jit** | `torch/csrc/jit/` | TorchScript fuser, TensorExpr (레거시 NVRTC 경로) | 5 | 0 |
| **nativert** | `torch/nativert/executor/triton/` | 네이티브 런타임의 Triton 커널 관리자 | 5 | 0 |
| **profiler** | `torch/csrc/profiler/stubs/`, `torch/profiler/_cupti/` | 이벤트 기반 타이밍, CUPTI 모니터 | 2 | 7 |
| **합계 (중복 제외)** | | | **50** | **99** |

디바이스 측 Runtime API(`cudaGridDependencySynchronize` 등)는 쓰지 않습니다. PDL은 `torch/_vendor/quack`(CuTe DSL GEMM)에만 있고, CUDA API 호출이 아니라 DSL로 표현되어 있습니다.

### PyTorch가 CUDA를 부르는 네 가지 경로

| 경로 | 방법 | 쓰는 곳 |
|---|---|---|
| **Runtime 직접 링크** | `libcudart`에 링크하고 `C10_CUDA_CHECK(cudaXxx(...))` | 대부분 (c10, aten, c10d) |
| **c10 Driver 테이블** | `c10/cuda/driver_api.{h,cpp}`: `cudaGetDriverEntryPointByVersion`(실패하면 `cudaGetDriverEntryPoint`)으로 심볼마다 **요청 버전**을 지정해 해석. NVML도 `dlopen` | VMM(expandable segments), symmetric memory, 멀티캐스트, 드라이버 로그 콜백 |
| **ATen lazy NVRTC 테이블** | `aten/src/ATen/cuda/nvrtc_stub/ATenNVRTC.h`, `detail/LazyNVRTC.cpp`: NVRTC와 일부 Driver 함수를 처음 쓸 때 로드 | jiterator, cuBLAS·cuFFT 전 context 확인, primary context 상태 |
| **Python** | `ctypes.CDLL("libcuda.so.1")` (`torch/cuda/_utils.py`), cuda-python (`_drv`, `_rt`) | `torch.cuda._compile_kernel`, green context, 그래프 분석, UVM |

c10 Driver 테이블의 요청 버전은 다음과 같습니다(`driver_api.h`). 주석에 따르면 새 드라이버에서 동작이 바뀐 새 버전 함수에 묶이지 않도록, **가능한 한 낮은 버전**을 요청합니다.

| 그룹 | 요청 버전 | 함수 |
|---|---|---|
| 필수 | 12000 | `cuDeviceGet`, `cuDeviceGetAttribute`, `cuMemGetAddressRange`, `cuMemAddressReserve`/`Free`, `cuMemCreate`/`Release`, `cuMemMap`/`Unmap`, `cuMemSetAccess`, `cuMemGetAllocationGranularity`, `cuMemExportToShareableHandle`/`ImportFromShareableHandle`, `cuMemRetainAllocationHandle`, `cuMemGetAllocationPropertiesFromHandle`, `cuMemsetD32Async`, `cuStreamWriteValue32`, `cuGetErrorString` |
| CUDA 12.3+ 빌드 | 12030 | `cuMulticastCreate`, `cuMulticastAddDevice`, `cuMulticastBindMem`, `cuMulticastUnbind` |
| CUDA 12.8+ 빌드 | 12080 | `cuGreenCtxCreate`/`Destroy`/`StreamCreate`, `cuCtxFromGreenCtx`, `cuCtxGetCurrent`/`SetCurrent`/`PushCurrent`/`PopCurrent`, `cuDevSmResourceSplitByCount`, `cuDeviceGetDevResource`, `cuDevResourceGenerateDesc` |
| CUDA 12.9+ 빌드 | 12090 | `cuLogsRegisterCallback`, `cuLogsUnregisterCallback` |

**선언만 있고 C++에서 호출하지 않는 테이블 항목**: `cuDeviceGet`, `cuMemGetAddressRange`, `cuMemRetainAllocationHandle`, `cuMemGetAllocationPropertiesFromHandle`, 12.8 그룹 전체. green context는 Python(`torch/cuda/green_contexts.py`)이 cuda-python 바인딩으로 따로 호출합니다.

---

## 1. CUDA Driver API

본체 코드에서 **50개**를 사용합니다.

### 1.1 메모리: VMM, 공유 핸들, 멀티캐스트

| API | 위치 | 용도 |
|---|---|---|
| `cuMemAddressReserve` / `cuMemAddressFree` | c10 `CUDACachingAllocator.cpp`, c10d `CUDASymmetricMemoryUtils.cpp` | **expandable segments**: GPU 전체 메모리의 1⅛배 가상 주소를 예약. symmetric memory 매핑 |
| `cuMemCreate` / `cuMemRelease` | 위와 같음 + `PeerToPeerAccess.cpp` | 세그먼트 단위 물리 메모리. `GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED`이면 RDMA 가능 플래그 설정 |
| `cuMemMap` / `cuMemUnmap` / `cuMemSetAccess` | 위와 같음 | 예약 영역에 매핑·해제·권한 |
| `cuMemGetAllocationGranularity` | `PeerToPeerAccess.cpp`, `CUDASymmetricMemory.cu` | 할당 단위 |
| `cuMemExportToShareableHandle` / `cuMemImportFromShareableHandle` | `CUDACachingAllocator.cpp`, `PeerToPeerAccess.cpp`, `CUDASymmetricMemory.cu` | expandable segment를 다른 프로세스와 공유(POSIX fd 또는 **FABRIC**), symmetric memory 핸들 교환 |
| `cuDeviceGetAttribute` | `CUDACachingAllocator.cpp`, c10d `cuda/utils.cpp`, `CUDASymmetricMemory.cu`, static launcher | RDMA·VMM 지원, **`MULTICAST_SUPPORTED`** 확인 |
| `cuMulticastCreate` / `cuMulticastAddDevice` / `cuMulticastBindMem` / `cuMulticastUnbind` | `CUDASymmetricMemory.cu` | NVLS 멀티캐스트 객체 |
| `cuMemsetD32Async` | `CUDASymmetricMemoryOps.cu` | signal pad 초기화 |
| `cuStreamWriteValue32` | `CUDASymmetricMemoryOps.cu` | 스트림 순서로 원격 signal 쓰기 |

`PeerToPeerAccess.cpp`의 `isFabricSupported()`는 FABRIC 핸들로 작은 메모리를 실제로 만들고 export·import해 보는 방식으로 지원 여부를 확인합니다.

### 1.2 커널 로딩·실행

| API | 위치 | 용도 |
|---|---|---|
| `cuModuleLoadData` / `cuModuleGetFunction` / `cuLaunchKernel` | aten `jit_utils.cpp` (jiterator), jit fuser·TensorExpr, `torch/cuda/_utils.py` | NVRTC로 컴파일한 커널 로드·실행. `torch.cuda._compile_kernel`은 ctypes로 libcuda를 직접 부름 |
| `cuModuleLoad` | inductor static launcher, nativert | **파일 경로**의 Triton CUBIN 로드 |
| `cuModuleUnload` | AOTI `model_base.h`, static launcher, jit fuser, nativert | 해제 |
| `cuFuncSetAttribute` / `cuFuncGetAttribute` / `cuFuncSetCacheConfig` | static launcher, `_utils.py` | dynamic smem, 속성, L1/smem 분할 |
| `cuOccupancyMaxActiveBlocksPerMultiprocessor` | jit fuser | 그리드 크기 |
| `cuPointerGetAttribute` | static launcher | 커널 인자 포인터 검증 |
| `cuTensorMapEncodeTiled` | inductor-gen (`device_op_overrides.py`) | Triton TMA 커널용 descriptor (생성 코드) |

### 1.3 컨텍스트

| API | 위치 | 용도 |
|---|---|---|
| `cuCtxGetCurrent` → `cuDevicePrimaryCtxRetain` → `cuCtxSetCurrent` | aten `CublasHandlePool.cpp`, `SpectralOps.cpp`, c10d `ProcessGroupUCC.cpp`, static launcher | cuBLAS·cuFFT·UCC·Triton 호출 전에 **현재 스레드에 context가 없으면 primary context를 설정**. cuBLAS 경로는 "context가 없어서 primary context를 설정한다"는 경고를 한 번 출력 |
| `cuDevicePrimaryCtxGetState` | aten `CUDAHooks.cpp` | `hasPrimaryContext()`: context를 만들지 않고 이미 있는지만 확인 (lazy init, fork 안전성) |
| `cuCtxPushCurrent` / `cuCtxPopCurrent` | `torch/cuda/green_contexts.py` | green context 진입·이탈 (deprecated API) |

### 1.4 Green Context (`torch/cuda/green_contexts.py`, cuda-python)

| API | 용도 |
|---|---|
| `cuDriverGetVersion` | 지원 드라이버 확인 |
| `cuDeviceGet`, `cuDeviceGetDevResource` | SM 리소스와 **work queue 설정 리소스** (`CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG`) |
| `cuDevSmResourceSplitByCount` | 요청한 SM 수로 분할 |
| `cuDevResourceGenerateDesc` → `cuGreenCtxCreate` | green context 생성 |
| `cuCtxFromGreenCtx` | context 핸들 |
| `cuGreenCtxStreamCreate`, `cuStreamDestroy` | green context 스트림 (풀당 32개) |
| `cuGreenCtxDestroy` | 해제 |

### 1.5 그래프 분석·기타

| API | 위치 | 용도 |
|---|---|---|
| `cuGraphKernelNodeGetParams`, `cuFuncGetName` | `torch/cuda/graphs.py` | 캡처한 그래프의 커널 노드 이름 |
| `cuGraphNodeGetType` | `torch/cuda/_graph_annotations.py` | 노드 종류 |
| `cuLogsRegisterCallback` | c10 `CUDAException.cpp` | **드라이버 에러 로그 콜백** (CUDA 12.9+). 에러 메시지에 드라이버 로그를 붙임 |
| `cuGetErrorString` | c10 `driver_api.h`(`C10_CUDA_DRIVER_CHECK`), aten `Exceptions.h`, AOTI, nativert, `_utils.py` | 에러 메시지 |
| `cuInit`, `cuCtxGetCurrent` | `torch/profiler/_cupti/monitor.py` | CUPTI 모니터 초기화 |

---

## 2. CUDA Runtime API

본체 코드에서 **99개**를 사용합니다.

### 2.1 디바이스·context (c10 `CUDAFunctions.cpp`, aten `CUDAContext.cpp`)

| API | 용도 |
|---|---|
| `cudaGetDeviceCount`, `cudaGetDevice`, `cudaSetDevice` | 디바이스 관리 (`c10::cuda::device_count()`, `CUDAGuard`) |
| `cudaGetDeviceProperties` | 디바이스별로 한 번 조회해 캐시 (`at::cuda::getDeviceProperties`) |
| `cudaDeviceGetAttribute` | 개별 속성 (`torch/csrc/cuda/Module.cpp`, nested tensor 커널, UVM 지원 확인) |
| `cudaDriverGetVersion`, `cudaRuntimeGetVersion` | `torch.cuda` 버전 정보, 기능 분기 |
| `cudaDeviceCanAccessPeer` | P2P 가능 여부 캐시 |
| `cudaPointerGetAttributes` | 포인터가 속한 디바이스 (`CUDADevice.h`), AOTI 상수 적재 |
| `cudaDeviceSynchronize` | `torch.cuda.synchronize()`, allocator `empty_cache` 등 |

### 2.2 스트림·이벤트 (c10 `CUDAStream`, `CUDAEvent`, `CUDAGuardImpl`)

| API | 용도 |
|---|---|
| `cudaDeviceGetStreamPriorityRange`, `cudaStreamCreateWithPriority`, `cudaStreamGetPriority` | **디바이스마다 우선순위별 스트림 풀**을 미리 만들어 돌려 씀 (`getStreamFromPool`) |
| `cudaStreamQuery`, `cudaStreamSynchronize`, `cudaStreamWaitEvent` | `torch.cuda.Stream.query/synchronize/wait_event` |
| `cudaEventCreateWithFlags`, `cudaEventRecord`, `cudaEventRecordWithFlags`, `cudaEventQuery`, `cudaEventSynchronize`, `cudaEventElapsedTime`, `cudaEventDestroy` | `torch.cuda.Event`. 타이밍 없는 이벤트가 기본 |
| `cudaIpcGetEventHandle`, `cudaIpcOpenEventHandle` | 프로세스 간 이벤트 공유 (`torch.multiprocessing`의 CUDA 텐서 공유, `StorageSharing.cpp`) |
| `cudaStreamCreate`, `cudaStreamDestroy` | `torch.cuda.cudart()` 바인딩 |
| `cudaLaunchHostFunc` | NVTX range를 **스트림 순서**로 시작·종료 (`torch/csrc/cuda/shared/nvtx.cpp`) |

### 2.3 메모리 할당기

**CUDACachingAllocator** (`c10/cuda/CUDACachingAllocator.cpp`, 기본 백엔드)

| API | 용도 |
|---|---|
| `cudaMalloc`, `cudaFree` | 세그먼트 할당·반환 (expandable segments가 꺼져 있을 때) |
| `cudaMemGetInfo` | OOM 메시지, `torch.cuda.mem_get_info()` |
| `cudaEventCreateWithFlags`, `cudaEventRecord`, `cudaEventQuery`, `cudaEventSynchronize` | **다른 스트림에서 쓰인 블록**(`record_stream`)이 끝났는지 확인한 뒤 재사용 |
| `cudaStreamGetCaptureInfo`, `cudaGraphNodeGetDependencies` | 그래프 캡처 중에 해제된 블록을 언제 재사용해도 되는지 판단 |
| `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle`, `cudaIpcCloseMemHandle` | `torch.multiprocessing` CUDA 텐서 공유 (legacy IPC) |
| `cudaDeviceEnablePeerAccess`, `cudaMemcpyPeerAsync`, `cudaMemcpyAsync`, `cudaMemcpy` | 디바이스 간 복사 |
| (Driver) VMM API | **expandable segments** (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) |

**cudaMallocAsync 백엔드** (`c10/cuda/CUDAMallocAsyncAllocator.cpp`, `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`)

| API | 용도 |
|---|---|
| `cudaDeviceGetDefaultMemPool`, `cudaMemPoolSetAttribute`, `cudaMemPoolGetAttribute` | 기본 풀 설정·통계 |
| `cudaMallocAsync`, `cudaFreeAsync` | 스트림 순서 할당 |
| `cudaMemPoolTrimTo` | `empty_cache` |
| `cudaMemPoolSetAccess` | 다른 GPU에서 풀 접근 |
| `cudaDeviceGetGraphMemAttribute`, `cudaDeviceSetGraphMemAttribute` | 그래프 메모리 통계·high watermark 초기화 |

**CachingHostAllocator** (`aten/src/ATen/cuda/CachingHostAllocator.cpp`, `pin_memory()`)

| API | 용도 |
|---|---|
| `cudaHostAlloc` / `cudaFreeHost` | 기본 pinned 할당 (주석: driver 전역 lock을 잡음) |
| `cudaHostRegister` / `cudaHostUnregister` | `pinned_use_cuda_host_register` 설정 시 `malloc` 후 등록 |
| (캡처 중이면) `cudaThreadExchangeStreamCaptureMode` | Relaxed 모드로 바꿔 캡처 중에도 할당 |

**Pluggable allocator** (`torch/csrc/cuda/CUDAPluggableAllocator.cpp`): 사용자 할당 함수를 꽂는 인터페이스. 자체적으로는 `cudaDeviceEnablePeerAccess`, `cudaMemcpy(Async)`만 부릅니다.

**UVM** (`torch/cuda/memory.py`, cuda-python): `cudaMallocManaged`, `cudaMemAdvise`, `cudaDeviceGetAttribute(ConcurrentManagedAccess)`, `cudaFree`. c10 `CUDADeviceAssertionHost.cpp`는 디바이스 측 assertion 버퍼에 managed memory를 씁니다.

### 2.4 CUDA Graph (`aten/src/ATen/cuda/CUDAGraph.cpp`, `c10/cuda/CUDAGraphsC10Utils.h`)

| API | 용도 |
|---|---|
| `cudaStreamBeginCapture` / `cudaStreamEndCapture` | `torch.cuda.CUDAGraph.capture_begin/end`. 기본 모드는 가장 보수적인 `cudaStreamCaptureModeGlobal` |
| `cudaGraphInstantiateWithFlags` | **`AutoFreeOnLaunch \| UseNodePriority`** (ROCm은 `AutoFreeOnLaunch`만) |
| `cudaGraphLaunch`, `cudaGraphDestroy`, `cudaGraphExecDestroy` | replay, 해제 |
| `cudaGraphGetNodes` | 빈 그래프 경고 (잘못된 디바이스·스트림에서 캡처했을 때) |
| `cudaStreamIsCapturing`, `cudaStreamGetCaptureInfo` | 캡처 상태 (`currentStreamCaptureStatus`, capture id) |
| `cudaThreadExchangeStreamCaptureMode` | `CUDAStreamCaptureModeGuard` |
| `cudaUserObjectCreate`, `cudaGraphRetainUserObject`, `cudaUserObjectRelease` | 리소스 수명을 그래프에 묶음 |
| **조건 노드**: `cudaGraphConditionalHandleCreate`, `cudaGraphAddNode`, `cudaStreamBeginCaptureToGraph`, `cudaStreamUpdateCaptureDependencies` | `begin_capture_to_conditional_node()`: 데이터에 따라 실행 여부가 갈리는 본문을 그래프 안에 캡처 |
| **분석** (`torch/cuda/graphs.py`, `_graph_annotations.py`, cuda-python): `cudaGraphGetNodes`, `cudaGraphGetEdges`, `cudaGraphGetRootNodes`, `cudaGraphNodeGetType`, `cudaGraphNodeGetDependentNodes`, `cudaGraphHostNodeGetParams`, `cudaGraphEventRecordNodeGetEvent`, `cudaGraphEventWaitNodeGetEvent`, `cudaGraphGetId`, `cudaGraphExecGetId`, `cudaGraphNodeGetToolsId`, `cudaGraphDebugDotPrint` | 캡처한 그래프 구조 덤프, 노드와 프로파일러 이벤트 연결 |

### 2.5 커널 지원 (`aten/src/ATen/native/**/cuda/`)

| API | 용도 |
|---|---|
| `cudaFuncSetAttribute` | memory-efficient attention 순전파·역전파, nested tensor 커널의 대용량 smem |
| `cudaFuncGetAttributes` | int4mm, attention 역전파 |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaOccupancyMaxPotentialBlockSize` | softmax, sort |
| `cudaMemcpyAsync`, `cudaMemsetAsync` | 복사, reduce·topk 버퍼 초기화 |
| `cudaMemcpyToSymbol`, `cudaGetSymbolAddress` | cuBLAS device pointer mode용 상수(`BLASConstants.cu`) |
| `cudaGetLastError` | `C10_CUDA_KERNEL_LAUNCH_CHECK` (aten 커널에서만 48곳) |

### 2.6 분산 (c10d)

| API | 위치 | 용도 |
|---|---|---|
| `cudaEventQuery` | `ProcessGroupNCCL.cpp` | watchdog이 collective 완료를 블로킹 없이 확인 |
| `cudaStreamSynchronize`, `cudaMemcpy(Async)`, `cudaMemset(Async)`, `cudaDeviceSynchronize` | symmetric memory (CUDA·NCCL·NVSHMEM 백엔드), `NCCLUtils.cpp` | 버퍼 초기화·동기화 |
| `cudaGetDeviceCount`, `cudaGetDeviceProperties` | `CudaDMAConnectivity.cpp` | NVLink 토폴로지 감지 (NVML과 함께) |

### 2.7 Inductor / AOTInductor

| 구성 | API |
|---|---|
| AOTI 런타임 (`model_base.h`, `model_container.h`, `streams.h`) | `cudaMalloc`, `cudaFree`, `cudaMemcpy(Async)`, `cudaPointerGetAttributes`, `cudaStreamCreateWithFlags`, `cudaStreamDestroy`, `cudaStreamSynchronize`, `cudaStreamIsCapturing`, `cudaEventCreate(WithFlags)`, `cudaEventRecord`, `cudaEventQuery`, `cudaEventSynchronize`, `cudaEventDestroy`, `cudaGetDevice`, `cudaSetDevice`, `cudaGetLastError`, `cudaGetErrorString` |
| 생성 코드 (`cpp_wrapper_gpu.py`) | `cudaEventRecord`, `cudaEventSynchronize`, `cudaStreamWaitEvent`, (ROCm) `cudaDeviceSynchronize` |

ExecuTorch 분석의 "AOTI 생성 코드" 항목은 이 코드를 가리킵니다.

### 2.8 Python 노출 (`torch.cuda.cudart()`, `torch/csrc/cuda/shared/cudart.cpp`)

`cudaGetErrorString`, `cudaProfilerStart`, `cudaProfilerStop`, `cudaHostRegister`, `cudaHostUnregister`, `cudaStreamCreate`, `cudaStreamDestroy`, `cudaMemGetInfo`. vLLM 벤치마크가 프로파일 구간을 정할 때 이 바인딩을 씁니다.

### 2.9 테스트에서만 사용

`cudaGraphCreate`, `cudaGraphAddMemAllocNode`, `cudaMallocManaged`(aten 테스트), `cudaStreamCreateWithPriority`(스트림 테스트) 등. 본체에도 있는 API를 제외하면 새로 추가되는 것은 `cudaGraphCreate`, `cudaGraphAddMemAllocNode` 두 개입니다.

---

## 3. 최적화 기법별 API 매핑

### 3.1 캐싱 할당기 [c10]

요청마다 `cudaMalloc`/`cudaFree`를 부르지 않도록 세그먼트를 캐시하고 블록으로 나눠 씁니다.

| 기법 | 하는 일 | API |
|---|---|---|
| 블록 캐시 | 세그먼트를 크기별 풀에 두고 split·merge | `cudaMalloc`, `cudaFree` |
| 스트림 간 재사용 | `record_stream`으로 표시된 블록은 그 스트림의 이벤트가 끝나야 재사용 | `cudaEventRecord`, `cudaEventQuery` |
| **Expandable segments** | 세그먼트를 VMM으로 만들어, 큰 할당이 와도 새 세그먼트를 만들지 않고 기존 세그먼트를 끝으로 늘림 (단편화 감소) | `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, `cuMemUnmap`, `cuMemRelease` |
| Expandable segment 공유 | 다른 프로세스에 POSIX fd 또는 FABRIC 핸들로 공유 | `cuMemExportToShareableHandle`, `cuMemImportFromShareableHandle` |
| 그래프 전용 풀 | 캡처 중 할당은 그래프 전용 private pool에서. 캡처 중 해제된 블록은 그래프 의존성을 보고 재사용 판단 | `cudaStreamGetCaptureInfo`, `cudaGraphNodeGetDependencies` |
| MemPool / pluggable allocator | 사용자가 만든 할당기나 풀을 특정 코드 구간에서 쓰게 함 (`torch.cuda.MemPool`, `use_mem_pool`, `CUDAPluggableAllocator`) | (사용자 할당기에 위임) |

### 3.2 cudaMallocAsync 백엔드 [c10]

| 기법 | API |
|---|---|
| 드라이버 메모리 풀 사용 | `cudaDeviceGetDefaultMemPool`, `cudaMallocAsync`, `cudaFreeAsync` |
| 풀 유지·반환 | `cudaMemPoolSetAttribute`, `cudaMemPoolTrimTo` |
| 멀티 GPU 접근 | `cudaMemPoolSetAccess` |
| 그래프 메모리 계측 | `cudaDeviceGetGraphMemAttribute`, `cudaDeviceSetGraphMemAttribute` |

### 3.3 Pinned host 메모리 [aten]

| 기법 | API |
|---|---|
| 캐싱 pinned 할당기 (`tensor.pin_memory()`, `pin_memory=True`) | `cudaHostAlloc` / `cudaFreeHost` |
| 대안: `malloc` + 등록 (driver 전역 lock 회피) | `cudaHostRegister` / `cudaHostUnregister` |
| 기존 메모리 pin (`torch.cuda.cudart().cudaHostRegister`, `_pin_memory_utils.py`) | `cudaHostRegister` |
| 블록 재사용 시점 판단 | 이벤트 API |

### 3.4 CUDA Graph [aten, c10, torch.cuda]

| 기법 | API |
|---|---|
| 캡처·인스턴스화·replay | `cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaGraphInstantiateWithFlags`, `cudaGraphLaunch` |
| 그래프 안 할당 자동 해제, 노드 우선순위 | `cudaGraphInstantiateFlagAutoFreeOnLaunch \| cudaGraphInstantiateFlagUseNodePriority` |
| 캡처 안전성 | `cudaThreadExchangeStreamCaptureMode`, `cudaStreamIsCapturing`, `cudaStreamGetCaptureInfo` |
| 그래프 수명에 묶인 리소스 | `cudaUserObjectCreate`, `cudaGraphRetainUserObject` |
| **조건 노드** (데이터 의존 제어 흐름) | `cudaGraphConditionalHandleCreate`, `cudaGraphAddNode`, `cudaStreamBeginCaptureToGraph`, `cudaStreamUpdateCaptureDependencies` |
| 그래프 분석·디버깅 | `cudaGraphGetNodes`/`GetEdges`/`GetRootNodes` …, `cuGraphKernelNodeGetParams`, `cuFuncGetName`, `cudaGraphDebugDotPrint` |

### 3.5 런타임 커널 컴파일 [aten, inductor, jit, torch.cuda]

| 경로 | 컴파일 | 로드·실행 |
|---|---|---|
| jiterator (elementwise 커널 JIT) | NVRTC | `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel` (lazy NVRTC 테이블) |
| `torch.cuda._compile_kernel` | NVRTC (ctypes) | `cuModuleLoadData`, `cuModuleGetFunction`, `cuFuncSetAttribute`, `cuLaunchKernel` (ctypes libcuda) |
| **Inductor static launcher** (`torch.compile`) | Triton → CUBIN 파일 | `cuModuleLoad`, `cuModuleGetFunction`, `cuFuncGetAttribute`, `cuFuncSetAttribute`, `cuFuncSetCacheConfig`, `cuLaunchKernel`. Triton의 Python launcher를 거치지 않고 C++에서 바로 실행 |
| AOTInductor 생성 코드 | Triton → `.so`에 내장 | `cuModuleLoadData`, `cuModuleGetFunction`, `cuFuncSetAttribute`, `cuLaunchKernel`, `cuTensorMapEncodeTiled` |
| nativert | Triton CUBIN | `cuModuleLoad`, `cuModuleGetFunction`, `cuLaunchKernel` |
| TorchScript fuser·TensorExpr (레거시) | NVRTC | `cuModuleLoadData`, `cuModuleGetFunction`, `cuOccupancyMaxActiveBlocksPerMultiprocessor`, `cuLaunchKernel` |

### 3.6 Symmetric Memory [c10d]

`torch.distributed._symmetric_memory`(`empty`, `rendezvous`, `set_backend("CUDA" | "NCCL" | "NVSHMEM")`). 모든 rank가 같은 크기의 버퍼를 갖고 서로의 버퍼를 직접 읽고 쓰는 통신 기반입니다.

| 기법 | API |
|---|---|
| VMM 버퍼와 핸들 교환 (CUDA 백엔드) | `cuMemCreate`, `cuMemGetAllocationGranularity`, `cuMemExportToShareableHandle`, `cuMemImportFromShareableHandle`, `cuMemAddressReserve`, `cuMemMap`, `cuMemSetAccess` |
| **NVLS 멀티캐스트** | `cuDeviceGetAttribute(MULTICAST_SUPPORTED)` + `cuMulticastCreate` 존재 여부(드라이버 535+)로 확인 → `cuMulticastCreate`, `cuMulticastAddDevice`, `cuMulticastBindMem` |
| signal pad | `cuMemsetD32Async`, `cuStreamWriteValue32` |
| NVLink 토폴로지 | `cudaGetDeviceCount`, `cudaGetDeviceProperties` + NVML |

### 3.7 SM 분할 (Green Context) [torch.cuda]

`torch.cuda.green_contexts.GreenContext.create(...)` → `GreenContext.Stream()`. SM 수뿐 아니라 **work queue 공유 범위**(`workqueue_scope`: `device_ctx` / `balanced`)와 동시성 한도도 설정합니다. API는 [1.4](#14-green-context-torchcudagreen_contextspy-cuda-python)를 참고하세요.

### 3.8 컨텍스트 안전성

| 문제 | 대응 | API |
|---|---|---|
| 새 스레드에 context가 없는 상태에서 cuBLAS·cuFFT·UCC·Triton 호출 | primary context를 확보해 현재로 설정 | `cuCtxGetCurrent`, `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent` |
| context를 만들지 않고 초기화 여부만 확인 | `torch.cuda` lazy init, fork 전 검사 | `cuDevicePrimaryCtxGetState` |

### 3.9 에러·관측

| 분류 | API |
|---|---|
| 에러 | `cudaGetLastError`, `cudaGetErrorString`, `cudaGetErrorName`, `cuGetErrorString`, **`cuLogsRegisterCallback`** (드라이버 로그를 에러 메시지에 첨부) |
| 타이밍 | `cudaEventElapsedTime` (`torch.cuda.Event(enable_timing=True)`, TunableOp `StreamTimer`) |
| 프로파일 구간 | `cudaProfilerStart`/`Stop`, `cudaLaunchHostFunc` (스트림 순서 NVTX) |

---

## 4. 서빙 프레임워크가 PyTorch에 맡기는 것

vLLM, SGLang, TensorRT-LLM의 evidence 문서에 "PyTorch에 위임"이라고 적힌 기능들이 PyTorch 안에서 실제로 쓰는 CUDA API입니다.

| 서빙 프레임워크의 기능 | 쓰는 PyTorch API | PyTorch 안의 CUDA API |
|---|---|---|
| 일반 텐서 할당 (활성값, workspace, KV 고정 풀) | 캐싱 할당기 | `cudaMalloc`, 이벤트 API (expandable segments면 VMM) |
| KV 용량 산정 | `torch.cuda.mem_get_info()` | `cudaMemGetInfo` |
| 스트림·이벤트 | `torch.cuda.Stream`, `torch.cuda.Event` | `cudaStreamCreateWithPriority`(풀), `cudaEventCreateWithFlags`, `cudaEventRecord`, `cudaStreamWaitEvent` |
| CUDA Graph 캡처 (vLLM, SGLang, TRT-LLM) | `torch.cuda.CUDAGraph`, `torch.cuda.graph` | `cudaStreamBeginCapture`, `cudaGraphInstantiateWithFlags(AutoFreeOnLaunch\|UseNodePriority)`, `cudaGraphLaunch` |
| 그래프 간 메모리 풀 공유 (SGLang `runner_utils/pool.py`) | `torch.cuda.graph_pool_handle()` | 캐싱 할당기의 그래프 private pool |
| VMM 할당기를 텐서에 연결 (vLLM `CuMemAllocator`, SGLang `kv_vmm_backing.py`) | `CUDAPluggableAllocator`, `torch.cuda.MemPool`, `use_mem_pool` | 할당 자체는 서빙 프레임워크의 VMM 코드가 함 |
| Pinned host 버퍼 | `pin_memory()` | `cudaHostAlloc` / `cudaHostRegister` |
| 벤치마크 프로파일 구간 (vLLM) | `torch.cuda.cudart().cudaProfilerStart()` | `cudaProfilerStart` |
| `torch.compile`로 만든 커널 실행 | Inductor | Triton CUBIN을 `cuModuleLoad` → `cuLaunchKernel` (static launcher) |
| NCCL 통신 | `torch.distributed` (ProcessGroupNCCL) | NCCL 내부 + watchdog `cudaEventQuery` |

**PyTorch가 제공하지만 세 서빙 프레임워크가 쓰지 않고 직접 구현한 것** (evidence 문서 기준):

| 기능 | PyTorch 제공 | 서빙 프레임워크의 선택 |
|---|---|---|
| VMM 기반 할당 | expandable segments | vLLM·SGLang·TRT-LLM은 sleep/wake, 공유, arena를 위해 **자체 VMM 코드** 작성 |
| Green context | `torch.cuda.green_contexts` | SGLang·TRT-LLM은 **자체 C++ 구현** |
| NVLS 멀티캐스트 | symmetric memory | TRT-LLM은 **자체 구현** (`mcastDeviceMemory.cpp`) |
| IPC 버퍼 공유 | `torch.multiprocessing` | vLLM·SGLang은 Custom AllReduce를 위해 **ctypes로 `cudaIpc*` 직접 호출** |

---

## 5. 요약

| 최적화 기법 | 구성 요소 | Driver API | Runtime API |
|---|:---:|:---:|:---:|
| 캐싱 할당기 | c10 | | ● |
| Expandable segments (VMM) | c10 | ● | |
| cudaMallocAsync 백엔드 | c10 | | ● |
| Pinned host 할당기 | aten | | ● |
| CUDA Graph (캡처·조건 노드·분석) | aten, c10, torch.cuda | ○ (분석) | ● |
| 스트림 풀·이벤트 | c10 | | ● |
| 런타임 커널 컴파일·실행 (jiterator, static launcher, AOTI) | aten, inductor | ● | |
| Symmetric memory·NVLS 멀티캐스트 | c10d | ● | ○ |
| Green context | torch.cuda | ● | |
| 컨텍스트 안전성 | aten, inductor, c10d | ● | |

- **PyTorch는 Driver API 50개, Runtime API 99개를 씁니다.** 서빙 프레임워크로 치면 TensorRT-LLM(84 / 72)과 SGLang(46 / 44) 사이에 해당하는 규모이고, **Runtime API는 다섯 서빙 프레임워크 중 가장 많은 TRT-LLM보다도 많습니다.** [PT] 프로파일의 서빙 프레임워크는 이 위에 자기 API를 더 얹습니다.
- **C++의 Driver 호출은 거의 모두 함수 테이블을 거칩니다.** c10은 `cudaGetDriverEntryPointByVersion`으로 심볼마다 최소 버전을 정해 찾고, ATen·Inductor static launcher·nativert는 lazy NVRTC 테이블을, Python은 ctypes나 cuda-python을 씁니다. 함수를 직접 부르는 곳은 AOTI 런타임 헤더(`model_base.h`의 `cuModuleUnload`, `cuGetErrorString`)뿐인데, 이 헤더는 PyTorch 본체가 아니라 AOTI가 만든 모델 `.so`에 컴파일됩니다. 그래서 드라이버 버전이 달라도 PyTorch 본체는 로드되고, 기능별로 지원 여부를 확인할 수 있습니다.
- **PyTorch는 Driver API를 메모리(VMM·공유·멀티캐스트), 커널 로딩, green context, 컨텍스트 안전성에만 씁니다.** 서빙 가이드의 결정 규칙([08](../guide/08-driver-vs-runtime.md))과 같은 패턴입니다.
- **CUDA Graph는 PyTorch가 가장 넓게 다룹니다.** 조건 노드, user object, 그래프 분석까지 포함해 그래프·캡처 관련 Runtime API를 30개 넘게 씁니다. vLLM·SGLang·TRT-LLM 본체에 `cudaGraph*`가 거의 없는 이유가 이것입니다.
