# CUDA API 카탈로그

> 이 파일은 `scripts/gen_api_catalog.py`가 `docs/reference/data/*.csv`에서 생성합니다. 직접 고치지 말고 CSV를 고친 뒤 다시 생성하세요.

표기: ● 본체 코드에서 사용, t 테스트·CI·벤치마크에서만 사용, c 설정에 따라 사용, h HIP(ROCm) 빌드 전용.
Ollama의 O/L/M은 Ollama 본체/llama.cpp/MLX, ExecuTorch의 E/A는 ExecuTorch 런타임/AOTI 생성 코드입니다.
등급은 그 API가 쓰이는 기능 중 가장 낮은 등급입니다. 역할 `alt`는 같은 기능을 하는 다른 계층(Driver↔Runtime)의 API가 기본이라는 뜻입니다.

## 1. 프레임워크별 집계 (본체 코드, 중복 제외)

| 프레임워크 | Driver | Runtime (호스트) | Runtime (디바이스 측) |
|---|---:|---:|---:|
| vLLM | 16 | 36 | 2 |
| SGLang | 46 | 44 | 2 |
| TRT-LLM | 84 | 72 | 2 |
| Ollama | 26 | 70 | 2 |
| └ llama.cpp | 11 | 48 | 2 |
| └ mlx | 10 | 55 | 0 |
| └ ollama | 8 | 0 | 0 |
| ExecuTorch | 7 | 42 | 0 |
| └ aoti | 7 | 19 | 0 |
| └ runtime | 0 | 37 | 0 |

## 2. 등급별 API 수

| 등급 | Driver | Runtime (호스트) | Runtime (디바이스 측) | 합계 |
|---|---:|---:|---:|---:|
| T0 | 28 | 46 | 0 | 74 |
| T1 | 5 | 36 | 2 | 43 |
| T2 | 73 | 27 | 0 | 100 |
| T3 | 14 | 0 | 0 | 14 |
| AUX | 1 | 4 | 0 | 5 |
| 합계 | 121 | 113 | 2 | 236 |

## 3. 기능별 API

`직접 구현`은 그 기능을 자기 코드에서 CUDA API로 구현한 프레임워크, `위임`은 PyTorch 등에 맡기거나 호출자에게 넘기는 프레임워크입니다.

| 기능 | 등급 | 기본 API | 다른 계층 대응 API | 직접 구현 | 위임 | 가이드 |
|---|:---:|---|---|---|---|---|
| **T0-DEV** 디바이스 열거·선택 | T0 | `cudaDeviceReset`, `cudaGetDevice`, `cudaGetDeviceCount`, `cudaSetDevice`, `cudaSetDeviceFlags` |  | vLLM, SGLang, TRT-LLM, Ollama, ExecuTorch |  | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **T0-CAP** 하드웨어·버전 조회 | T0 | `cudaDeviceGetAttribute`, `cudaDeviceGetPCIBusId`, `cudaDriverGetVersion`, `cudaGetDeviceProperties`, `cudaRuntimeGetVersion` |  | vLLM, SGLang, TRT-LLM, Ollama, ExecuTorch |  | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **T0-MEMINFO** 메모리 용량 산정 | T0 | `cudaMemGetInfo` | `cuMemGetInfo` | TRT-LLM, Ollama, ExecuTorch | vLLM, SGLang | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **T0-ALLOC** 디바이스 메모리 할당·초기화 | T0 | `cudaFree`, `cudaMalloc`, `cudaMemset`, `cudaMemsetAsync` | `cuMemAlloc`, `cuMemFree`, `cuMemsetD32`, `cuMemsetD8`, `cuMemsetD8Async` | vLLM, SGLang, TRT-LLM, Ollama, ExecuTorch |  | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **T0-XFER** Host↔Device 전송과 pinned 버퍼 | T0 | `cudaFreeHost`, `cudaHostAlloc`, `cudaHostRegister`, `cudaHostUnregister`, `cudaMallocHost`, `cudaMemcpy`, `cudaMemcpy2DAsync`, `cudaMemcpyAsync`, `cudaMemcpyToSymbol`, `cudaPointerGetAttributes` | `cuMemcpyAsync`, `cuMemcpyDtoD`, `cuMemcpyDtoH`, `cuMemcpyDtoHAsync`, `cuMemcpyHtoD`, `cuMemcpyHtoDAsync`, `cuMemFreeHost`, `cuMemHostAlloc`, `cuMemHostRegister`, `cuMemHostUnregister` | vLLM, SGLang, TRT-LLM, Ollama, ExecuTorch |  | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **T0-STREAM** 스트림·이벤트 동기화 | T0 | `cudaDeviceSynchronize`, `cudaEventCreate`, `cudaEventCreateWithFlags`, `cudaEventDestroy`, `cudaEventQuery`, `cudaEventRecord`, `cudaEventRecordWithFlags`, `cudaEventSynchronize`, `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamDestroy`, `cudaStreamQuery`, `cudaStreamSynchronize`, `cudaStreamWaitEvent` | `cuCtxSynchronize`, `cuEventCreate`, `cuEventDestroy`, `cuEventQuery`, `cuEventRecord`, `cuEventSynchronize`, `cuStreamCreate`, `cuStreamDestroy`, `cuStreamSynchronize`, `cuStreamWaitEvent` | vLLM, SGLang, TRT-LLM, Ollama, ExecuTorch | vLLM, SGLang | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **T0-LAUNCH** 커널 실행·설정 | T0 | `cudaFuncGetAttributes`, `cudaFuncSetAttribute`, `cudaLaunchKernel` |  | vLLM, SGLang, TRT-LLM, Ollama, ExecuTorch |  | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **T0-ERR** 에러 처리 | T0 | `cudaGetErrorName`, `cudaGetErrorString`, `cudaGetLastError`, `cudaPeekAtLastError` | `cuGetErrorName`, `cuGetErrorString` | vLLM, SGLang, TRT-LLM, Ollama, ExecuTorch |  | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **T1-GRAPH** CUDA Graph (캡처·재실행) | T1 | `cudaDeviceGetGraphMemAttribute`, `cudaDeviceGraphMemTrim`, `cudaGraphDestroy`, `cudaGraphExecDestroy`, `cudaGraphExecUpdate`, `cudaGraphInstantiate`, `cudaGraphInstantiateWithFlags`, `cudaGraphLaunch`, `cudaStreamBeginCapture`, `cudaStreamEndCapture` |  | Ollama, ExecuTorch | vLLM, SGLang, TRT-LLM | [04-execution.md](../guide/04-execution.md) |
| **T1-CAPAWARE** 캡처 인지 동작 | T1 | `cudaStreamGetCaptureInfo`, `cudaStreamIsCapturing`, `cudaThreadExchangeStreamCaptureMode` |  | vLLM, SGLang, TRT-LLM, Ollama |  | [04-execution.md](../guide/04-execution.md) |
| **T1-PDL** PDL (Programmatic Dependent Launch) | T1 | `cudaGridDependencySynchronize`, `cudaLaunchKernelEx`, `cudaLaunchKernelExC`, `cudaTriggerProgrammaticLaunchCompletion`, `cuLaunchKernelEx` |  | vLLM, SGLang, TRT-LLM, Ollama |  | [04-execution.md](../guide/04-execution.md) |
| **T1-OCC** Occupancy 기반 launch 구성 | T1 | `cudaOccupancyAvailableDynamicSMemPerBlock`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaOccupancyMaxPotentialBlockSize` | `cuOccupancyMaxPotentialBlockSize` | vLLM, SGLang, TRT-LLM, Ollama |  | [04-execution.md](../guide/04-execution.md) |
| **T1-POOL** 스트림 순서 메모리 풀 | T1 | `cudaDeviceGetDefaultMemPool`, `cudaDeviceGetMemPool`, `cudaFreeAsync`, `cudaMallocAsync`, `cudaMallocFromPoolAsync`, `cudaMemPoolCreate`, `cudaMemPoolDestroy`, `cudaMemPoolGetAttribute`, `cudaMemPoolSetAttribute`, `cudaMemPoolTrimTo` |  | TRT-LLM, Ollama, ExecuTorch | vLLM, SGLang | [02-memory.md](../guide/02-memory.md) |
| **T1-ZEROCOPY** Mapped host 메모리 (zero-copy) | T1 | `cudaHostAlloc`, `cudaHostGetDevicePointer` | `cuMemHostGetDevicePointer` | vLLM, SGLang, TRT-LLM, Ollama |  | [02-memory.md](../guide/02-memory.md) |
| **T1-BATCHCOPY** 배치 복사 (KV 블록 일괄 전송) | T1 | `cuMemcpyBatchAsync` | `cudaMemcpyBatchAsync` | vLLM, SGLang, TRT-LLM |  | [03-kv-cache.md](../guide/03-kv-cache.md) |
| **T1-IPC** Custom AllReduce (IPC 버퍼 공유) | T1 | `cudaIpcCloseMemHandle`, `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle`, `cuPointerGetAttribute` |  | vLLM, SGLang, TRT-LLM |  | [06-multi-gpu.md](../guide/06-multi-gpu.md) |
| **T1-P2P** Peer access·P2P 복사 | T1 | `cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess`, `cudaMemcpyPeerAsync` |  | SGLang, TRT-LLM, Ollama |  | [06-multi-gpu.md](../guide/06-multi-gpu.md) |
| **T2-VMM-POOL** VMM 확장 풀·arena | T2 | `cuDeviceGetAttribute`, `cuMemAddressFree`, `cuMemAddressReserve`, `cuMemCreate`, `cuMemGetAllocationGranularity`, `cuMemMap`, `cuMemRelease`, `cuMemSetAccess`, `cuMemUnmap` |  | SGLang, TRT-LLM, Ollama |  | [02-memory.md](../guide/02-memory.md) |
| **T2-SLEEP** Sleep / Wake (주소 유지 메모리 반납) | T2 | `cuMemAddressFree`, `cuMemAddressReserve`, `cuMemCreate`, `cuMemGetAllocationGranularity`, `cuMemMap`, `cuMemRelease`, `cuMemSetAccess`, `cuMemUnmap` |  | vLLM, TRT-LLM |  | [02-memory.md](../guide/02-memory.md) |
| **T2-VMM-SHARE** VMM 공유 핸들 (프로세스 간 무복사 공유) | T2 | `cuMemAddressFree`, `cuMemAddressReserve`, `cuMemCreate`, `cuMemExportToShareableHandle`, `cuMemGetAddressRange`, `cuMemGetAllocationGranularity`, `cuMemGetAllocationPropertiesFromHandle`, `cuMemImportFromShareableHandle`, `cuMemMap`, `cuMemRelease`, `cuMemRetainAllocationHandle`, `cuMemSetAccess`, `cuMemUnmap` |  | SGLang, TRT-LLM |  | [02-memory.md](../guide/02-memory.md) |
| **T2-UVM** Unified (managed) memory | T2 | `cudaMallocManaged`, `cudaMemAdvise`, `cudaMemPrefetchAsync` |  | TRT-LLM, Ollama |  | [02-memory.md](../guide/02-memory.md) |
| **T2-KLOAD** 외부 CUBIN·JIT 커널 로딩 | T2 | `cuCtxGetId`, `cudaGetKernel`, `cuFuncGetAttribute`, `cuFuncSetAttribute`, `cuKernelGetFunction`, `cuKernelGetName`, `cuKernelSetAttribute`, `cuLaunchKernel`, `cuLaunchKernelEx`, `cuLibraryEnumerateKernels`, `cuLibraryGetGlobal`, `cuLibraryGetKernel`, `cuLibraryGetKernelCount`, `cuLibraryLoadData`, `cuLibraryUnload`, `cuModuleGetFunction`, `cuModuleGetGlobal`, `cuModuleLoad`, `cuModuleLoadData`, `cuModuleLoadDataEx`, `cuModuleUnload` | `cudaLibraryLoadData`, `cuGraphAddKernelNode`, `cuGraphKernelNodeSetAttribute`, `cuOccupancyMaxPotentialBlockSize` | SGLang, TRT-LLM, Ollama, ExecuTorch |  | [05-kernel-loading.md](../guide/05-kernel-loading.md) |
| **T2-TMA** TMA descriptor | T2 | `cudaGetDriverEntryPoint`, `cudaGetDriverEntryPointByVersion`, `cuTensorMapEncodeTiled` |  | SGLang, TRT-LLM, Ollama, ExecuTorch |  | [04-execution.md](../guide/04-execution.md) |
| **T2-CLUSTER** Thread Block Cluster | T2 | `cudaLaunchKernelEx`, `cudaLaunchKernelExC`, `cudaOccupancyMaxActiveClusters`, `cuLaunchKernelEx` | `cuOccupancyMaxPotentialClusterSize` | SGLang, TRT-LLM, Ollama |  | [04-execution.md](../guide/04-execution.md) |
| **T2-GRAPH-BUILD** 그래프 직접 조립·분석 | T2 | `cudaGraphAddChildGraphNode`, `cudaGraphAddDependencies`, `cudaGraphAddEmptyNode`, `cudaGraphAddKernelNode`, `cudaGraphChildGraphNodeGetGraph`, `cudaGraphCreate`, `cudaGraphGetNodes`, `cudaGraphKernelNodeGetAttribute`, `cudaGraphKernelNodeGetParams`, `cudaGraphKernelNodeSetAttribute`, `cudaGraphMemAllocNodeGetParams`, `cudaGraphMemFreeNodeGetParams`, `cudaGraphNodeGetType` | `cuGraphAddKernelNode`, `cuGraphChildGraphNodeGetGraph`, `cuGraphGetEdges`, `cuGraphGetNodes`, `cuGraphKernelNodeGetAttribute`, `cuGraphKernelNodeGetParams`, `cuGraphKernelNodeSetAttribute`, `cuGraphMemcpyNodeGetParams`, `cuGraphMemsetNodeGetParams`, `cuGraphNodeGetType` | SGLang, Ollama, ExecuTorch |  | [04-execution.md](../guide/04-execution.md) |
| **T2-COOP** Cooperative launch (grid-wide sync) | T2 | `cudaLaunchCooperativeKernel` |  | TRT-LLM, Ollama |  | [04-execution.md](../guide/04-execution.md) |
| **T2-HOSTFN** 스트림 순서 호스트 콜백 | T2 | `cudaLaunchHostFunc`, `cudaLaunchHostFunc_v2`, `cudaStreamAddCallback` | `cuLaunchHostFunc` | TRT-LLM, Ollama |  | [04-execution.md](../guide/04-execution.md) |
| **T2-GREEN** Green Context (SM 분할) | T2 | `cuCtxFromGreenCtx`, `cuDeviceGetDevResource`, `cuDevResourceGenerateDesc`, `cuDevSmResourceSplit`, `cuDevSmResourceSplitByCount`, `cuGreenCtxCreate`, `cuGreenCtxDestroy`, `cuGreenCtxGetDevResource`, `cuGreenCtxStreamCreate` | `cuStreamCreate` | SGLang, TRT-LLM | ExecuTorch | [07-scheduling-isolation.md](../guide/07-scheduling-isolation.md) |
| **T2-PRIO** 스트림 우선순위 | T2 | `cudaDeviceGetStreamPriorityRange`, `cudaStreamCreateWithPriority` |  | TRT-LLM |  | [07-scheduling-isolation.md](../guide/07-scheduling-isolation.md) |
| **T2-STREAMMEM** 스트림 메모리 연산 (GPU 측 신호) | T2 | `cuStreamWaitValue32`, `cuStreamWriteValue32` |  | SGLang |  | [07-scheduling-isolation.md](../guide/07-scheduling-isolation.md) |
| **T2-DRVDETECT** 드라이버만으로 GPU 탐지 | T2 | `cuDeviceGet`, `cuDeviceGetAttribute`, `cuDeviceGetCount`, `cuDeviceGetName`, `cuDeviceGetPCIBusId`, `cuDeviceGetUuid_v2`, `cuDeviceTotalMem`, `cuDriverGetVersion`, `cuInit` |  | Ollama |  | [08-driver-vs-runtime.md](../guide/08-driver-vs-runtime.md) |
| **T2-VERGATE** 버전 의존 심볼 해석 | T2 | `cudaDriverGetVersion`, `cudaGetDriverEntryPoint`, `cudaGetDriverEntryPointByVersion`, `cudaRuntimeGetVersion`, `cuDriverGetVersion`, `cuGetProcAddress` |  | vLLM, SGLang, TRT-LLM |  | [08-driver-vs-runtime.md](../guide/08-driver-vs-runtime.md) |
| **T2-CTX** 컨텍스트 관리 (Driver·Runtime 혼용) | T2 | `cuCtxCreate`, `cuCtxGetCurrent`, `cuCtxGetDevice`, `cuCtxPopCurrent`, `cuCtxPushCurrent`, `cuCtxSetCurrent`, `cuDevicePrimaryCtxRetain` |  | vLLM, SGLang, TRT-LLM |  | [08-driver-vs-runtime.md](../guide/08-driver-vs-runtime.md) |
| **T3-NVLS** NVLS 멀티캐스트 | T3 | `cuDeviceGetAttribute`, `cuMemAddressFree`, `cuMemAddressReserve`, `cuMemCreate`, `cuMemExportToShareableHandle`, `cuMemGetAllocationGranularity`, `cuMemImportFromShareableHandle`, `cuMemMap`, `cuMemRelease`, `cuMemSetAccess`, `cuMemUnmap`, `cuMulticastAddDevice`, `cuMulticastBindMem`, `cuMulticastCreate`, `cuMulticastGetGranularity`, `cuMulticastUnbind` |  | TRT-LLM |  | [06-multi-gpu.md](../guide/06-multi-gpu.md) |
| **T3-MNNVL** MNNVL fabric 메모리 | T3 | `cuDeviceGetAttribute`, `cuMemAddressFree`, `cuMemAddressReserve`, `cuMemCreate`, `cuMemExportToShareableHandle`, `cuMemGetAddressRange`, `cuMemGetAllocationGranularity`, `cuMemImportFromShareableHandle`, `cuMemMap`, `cuMemRelease`, `cuMemSetAccess`, `cuMemUnmap` |  | TRT-LLM |  | [06-multi-gpu.md](../guide/06-multi-gpu.md) |
| **T3-LE** Logical Endpoint | T3 | `cuLogicalEndpointBindMem`, `cuLogicalEndpointCreate`, `cuLogicalEndpointDestroy`, `cuLogicalEndpointExport`, `cuLogicalEndpointIdRelease`, `cuLogicalEndpointIdReserve`, `cuLogicalEndpointImport`, `cuLogicalEndpointQuery`, `cuLogicalEndpointUnbind` |  | TRT-LLM |  | [06-multi-gpu.md](../guide/06-multi-gpu.md) |
| **T3-LINKAWARE** 링크 인지 복사 전략 | T3 | `cudaDeviceGetPCIBusId`, `cudaGetKernel` |  | TRT-LLM |  | [03-kv-cache.md](../guide/03-kv-cache.md) |
| **T3-PCIEAR** PCIe AllReduce (NVLink 없는 TP) | T3 | `cudaHostAlloc`, `cudaHostGetDevicePointer` |  | Ollama |  | [06-multi-gpu.md](../guide/06-multi-gpu.md) |
| **AUX-PROF** 프로파일링·계측 | AUX | `cudaEventElapsedTime`, `cudaProfilerStart`, `cudaProfilerStop`, `cuEventElapsedTime` |  | SGLang, TRT-LLM |  | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **AUX-DEBUG** 디버깅 | AUX | `cudaGraphDebugDotPrint` |  | Ollama |  | [01-t0-essential.md](../guide/01-t0-essential.md) |

## 4. Driver API

| API | 종류 | 등급 | 기능 | 역할 | vLLM | SGLang | TRT-LLM | Ollama | ExecuTorch | 최소 CUDA | 하드웨어 | 비고 |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|---|---|---|
| `cuCtxSynchronize` | driver | T0 | T0-STREAM | alt |  |  | ● |  | Ac |  |  |  |
| `cuEventCreate` | driver | T0 | T0-STREAM | alt |  |  | ● |  |  |  |  |  |
| `cuEventDestroy` | driver | T0 | T0-STREAM | alt |  |  | ● |  |  |  |  |  |
| `cuEventQuery` | driver | T0 | T0-STREAM | alt |  |  | ● |  |  |  |  |  |
| `cuEventRecord` | driver | T0 | T0-STREAM | alt |  |  | ● |  |  |  |  |  |
| `cuEventSynchronize` | driver | T0 | T0-STREAM | alt |  |  | ● |  |  |  |  |  |
| `cuGetErrorName` | driver | T0 | T0-ERR | alt |  | ● | ● |  |  |  |  |  |
| `cuGetErrorString` | driver | T0 | T0-ERR | alt | ● | ● | ● | L● M● | A● |  |  |  |
| `cuMemAlloc` | driver | T0 | T0-ALLOC | alt |  |  | ● |  |  |  |  |  |
| `cuMemcpyAsync` | driver | T0 | T0-XFER | alt |  |  | ● |  |  |  |  |  |
| `cuMemcpyDtoD` | driver | T0 | T0-XFER | alt |  | ● | ● |  |  |  |  |  |
| `cuMemcpyDtoH` | driver | T0 | T0-XFER | alt |  | t | ● |  |  |  |  |  |
| `cuMemcpyDtoHAsync` | driver | T0 | T0-XFER | alt |  |  | t |  |  |  |  |  |
| `cuMemcpyHtoD` | driver | T0 | T0-XFER | alt |  |  | ● |  |  |  |  |  |
| `cuMemcpyHtoDAsync` | driver | T0 | T0-XFER | alt |  |  | ● |  |  |  |  |  |
| `cuMemFree` | driver | T0 | T0-ALLOC | alt |  |  | ● |  |  |  |  |  |
| `cuMemFreeHost` | driver | T0 | T0-XFER | alt |  |  | t |  |  |  |  |  |
| `cuMemGetInfo` | driver | T0 | T0-MEMINFO | alt |  |  | ● |  |  |  |  |  |
| `cuMemHostAlloc` | driver | T0 | T0-XFER | alt |  |  | t |  |  |  |  |  |
| `cuMemHostRegister` | driver | T0 | T0-XFER | alt |  |  | ● |  |  |  |  |  |
| `cuMemHostUnregister` | driver | T0 | T0-XFER | alt |  |  | ● |  |  |  |  |  |
| `cuMemsetD32` | driver | T0 | T0-ALLOC | alt |  |  | ● |  |  |  |  |  |
| `cuMemsetD8` | driver | T0 | T0-ALLOC | alt |  | t | ● |  |  |  |  |  |
| `cuMemsetD8Async` | driver | T0 | T0-ALLOC | alt |  |  | ● |  |  |  |  |  |
| `cuStreamCreate` | driver | T0 | T0-STREAM, T2-GREEN | alt |  | ● | ● |  |  |  |  | green context 구버전 폴백에서도 사용 (SGLang) |
| `cuStreamDestroy` | driver | T0 | T0-STREAM | alt |  |  | ● |  |  |  |  |  |
| `cuStreamSynchronize` | driver | T0 | T0-STREAM | alt |  |  | ● |  |  |  |  |  |
| `cuStreamWaitEvent` | driver | T0 | T0-STREAM | alt |  |  | ● |  |  |  |  |  |
| `cuLaunchKernelEx` | driver | T1 | T2-KLOAD, T1-PDL, T2-CLUSTER | primary |  |  | ● | M● |  | 12.0 |  |  |
| `cuMemcpyBatchAsync` | driver | T1 | T1-BATCHCOPY | primary | ● |  | ● |  |  | 12.8 |  | cuGetProcAddress로 해석, 미지원 시 반복 복사 폴백 |
| `cuMemHostGetDevicePointer` | driver | T1 | T1-ZEROCOPY | alt |  |  | t |  |  |  |  |  |
| `cuOccupancyMaxPotentialBlockSize` | driver | T1 | T1-OCC, T2-KLOAD | alt |  |  |  | M● |  |  |  | JIT 커널(CUfunction)용 |
| `cuPointerGetAttribute` | driver | T1 | T1-IPC | primary | ● | ● | t |  |  |  |  | CU_POINTER_ATTRIBUTE_RANGE_START_ADDR로 캐싱 할당기 블록의 base 주소 |
| `cuCtxCreate` | driver | T2 | T2-CTX | primary |  |  | t |  |  |  |  |  |
| `cuCtxFromGreenCtx` | driver | T2 | T2-GREEN | primary |  | ● |  |  |  | 12.4 |  |  |
| `cuCtxGetCurrent` | driver | T2 | T2-CTX | primary | ● |  | ● |  |  |  |  |  |
| `cuCtxGetDevice` | driver | T2 | T2-CTX | primary | t |  | ● |  |  |  |  |  |
| `cuCtxGetId` | driver | T2 | T2-KLOAD | primary |  |  | ● |  |  |  |  | 컨텍스트별 모듈 캐시 키 |
| `cuCtxPopCurrent` | driver | T2 | T2-CTX | primary |  | ● |  |  |  |  |  |  |
| `cuCtxPushCurrent` | driver | T2 | T2-CTX | primary |  | ● |  |  |  |  |  |  |
| `cuCtxSetCurrent` | driver | T2 | T2-CTX | primary | ● |  | ● |  |  |  |  |  |
| `cuDeviceGet` | driver | T2 | T2-DRVDETECT | primary | t |  | ● | O● L● |  |  |  |  |
| `cuDeviceGetAttribute` | driver | T2 | T2-DRVDETECT, T2-VMM-POOL, T3-NVLS, T3-MNNVL | primary | ● |  | ● | O● L● |  |  |  | VMM·멀티캐스트·fabric 지원 여부 확인 |
| `cuDeviceGetCount` | driver | T2 | T2-DRVDETECT | primary | t |  |  | O● |  |  |  |  |
| `cuDeviceGetDevResource` | driver | T2 | T2-GREEN | primary |  | ● | ● |  |  | 12.4 |  |  |
| `cuDeviceGetName` | driver | T2 | T2-DRVDETECT | primary |  |  |  | O● |  |  |  |  |
| `cuDeviceGetPCIBusId` | driver | T2 | T2-DRVDETECT | primary |  |  |  | O● |  |  |  |  |
| `cuDeviceGetUuid_v2` | driver | T2 | T2-DRVDETECT | primary | t |  |  |  |  |  |  |  |
| `cuDevicePrimaryCtxRetain` | driver | T2 | T2-CTX | primary | ● |  | ● |  |  |  |  |  |
| `cuDeviceTotalMem` | driver | T2 | T2-DRVDETECT | primary |  |  | ● | O● |  |  |  |  |
| `cuDevResourceGenerateDesc` | driver | T2 | T2-GREEN | primary |  | ● | ● |  |  | 12.4 |  |  |
| `cuDevSmResourceSplit` | driver | T2 | T2-GREEN | primary |  |  | ● |  |  |  |  | TRT-LLM locality domain, cuGetProcAddress로 해석 |
| `cuDevSmResourceSplitByCount` | driver | T2 | T2-GREEN | primary |  | ● |  |  |  | 12.4 |  |  |
| `cuDriverGetVersion` | driver | T2 | T2-DRVDETECT, T2-VERGATE | primary |  | ● |  | O● |  |  |  |  |
| `cuFuncGetAttribute` | driver | T2 | T2-KLOAD | primary | t | ● |  |  |  |  |  | CUfunction의 smem·속성 |
| `cuFuncSetAttribute` | driver | T2 | T2-KLOAD | primary |  |  | ● | M● | A● |  |  | CUfunction의 smem·속성 |
| `cuGetProcAddress` | driver | T2 | T2-VERGATE | primary | ● | ● | ● |  |  |  |  |  |
| `cuGraphAddKernelNode` | driver | T2 | T2-GRAPH-BUILD, T2-KLOAD | alt |  |  |  | M● |  |  |  | JIT 커널(CUfunction)을 그래프 노드로 (MLX) |
| `cuGraphChildGraphNodeGetGraph` | driver | T2 | T2-GRAPH-BUILD | alt |  | ● |  |  |  |  |  | 캡처한 그래프 분석·dedup (SGLang) |
| `cuGraphGetEdges` | driver | T2 | T2-GRAPH-BUILD | alt |  | ● |  |  |  |  |  | 캡처한 그래프 분석·dedup (SGLang) |
| `cuGraphGetNodes` | driver | T2 | T2-GRAPH-BUILD | alt |  | ● |  |  |  |  |  | 캡처한 그래프 분석·dedup (SGLang) |
| `cuGraphKernelNodeGetAttribute` | driver | T2 | T2-GRAPH-BUILD | alt |  | ● |  |  |  |  |  | 캡처한 그래프 분석·dedup (SGLang) |
| `cuGraphKernelNodeGetParams` | driver | T2 | T2-GRAPH-BUILD | alt |  | ● | t |  |  |  |  | 캡처한 그래프 분석·dedup (SGLang) |
| `cuGraphKernelNodeSetAttribute` | driver | T2 | T2-GRAPH-BUILD, T2-KLOAD | alt |  |  |  | M● |  |  |  | JIT 커널(CUfunction)을 그래프 노드로 (MLX) |
| `cuGraphMemcpyNodeGetParams` | driver | T2 | T2-GRAPH-BUILD | alt |  | ● |  |  |  |  |  | 캡처한 그래프 분석·dedup (SGLang) |
| `cuGraphMemsetNodeGetParams` | driver | T2 | T2-GRAPH-BUILD | alt |  | ● |  |  |  |  |  | 캡처한 그래프 분석·dedup (SGLang) |
| `cuGraphNodeGetType` | driver | T2 | T2-GRAPH-BUILD | alt |  | ● |  |  |  |  |  | 캡처한 그래프 분석·dedup (SGLang) |
| `cuGreenCtxCreate` | driver | T2 | T2-GREEN | primary |  | ● | ● |  |  | 12.4 |  |  |
| `cuGreenCtxDestroy` | driver | T2 | T2-GREEN | primary |  | ● | ● |  |  | 12.4 |  |  |
| `cuGreenCtxGetDevResource` | driver | T2 | T2-GREEN | primary |  | ● |  |  |  | 12.4 |  |  |
| `cuGreenCtxStreamCreate` | driver | T2 | T2-GREEN | primary |  | ● | ● |  |  | 12.5 |  | cuGetProcAddress로 해석 |
| `cuInit` | driver | T2 | T2-DRVDETECT | primary | t |  | ● | O● |  |  |  |  |
| `cuKernelGetFunction` | driver | T2 | T2-KLOAD | primary |  | ● |  |  |  | 12.0 |  | context 독립 library API |
| `cuKernelGetName` | driver | T2 | T2-KLOAD | primary |  |  | ● |  |  | 12.0 |  | context 독립 library API |
| `cuKernelSetAttribute` | driver | T2 | T2-KLOAD | primary |  |  | ● |  |  | 12.0 |  | context 독립 library API |
| `cuLaunchHostFunc` | driver | T2 | T2-HOSTFN | alt |  |  | ● |  |  |  |  |  |
| `cuLaunchKernel` | driver | T2 | T2-KLOAD | primary |  |  | ● |  | A● |  |  |  |
| `cuLibraryEnumerateKernels` | driver | T2 | T2-KLOAD | primary |  | ● | ● |  |  | 12.0 |  | context 독립 library API |
| `cuLibraryGetGlobal` | driver | T2 | T2-KLOAD | primary |  |  | ● |  |  | 12.0 |  | context 독립 library API |
| `cuLibraryGetKernel` | driver | T2 | T2-KLOAD | primary |  |  | ● |  |  | 12.0 |  | context 독립 library API |
| `cuLibraryGetKernelCount` | driver | T2 | T2-KLOAD | primary |  |  | ● |  |  | 12.0 |  | context 독립 library API |
| `cuLibraryLoadData` | driver | T2 | T2-KLOAD | primary |  | ● | ● |  |  | 12.0 |  | context 독립 library API |
| `cuLibraryUnload` | driver | T2 | T2-KLOAD | primary |  |  | ● |  |  | 12.0 |  | context 독립 library API |
| `cuMemAddressFree` | driver | T2 | T2-VMM-POOL, T2-SLEEP, T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary | ● | ● | ● | L● |  | 10.2 |  | VMM 공통 시퀀스 |
| `cuMemAddressReserve` | driver | T2 | T2-VMM-POOL, T2-SLEEP, T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary | ● | ● | ● | L● |  | 10.2 |  | VMM 공통 시퀀스 |
| `cuMemCreate` | driver | T2 | T2-VMM-POOL, T2-SLEEP, T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary | ● | ● | ● | L● |  | 10.2 |  | VMM 공통 시퀀스 |
| `cuMemExportToShareableHandle` | driver | T2 | T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary |  | ● | ● |  |  | 10.2 |  | POSIX fd / FABRIC 핸들 (FABRIC은 IMEX 필요) |
| `cuMemGetAddressRange` | driver | T2 | T2-VMM-SHARE, T3-MNNVL | primary |  | ● | ● |  |  |  |  | 포인터가 속한 할당의 base·크기 |
| `cuMemGetAllocationGranularity` | driver | T2 | T2-VMM-POOL, T2-SLEEP, T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary | ● | ● | ● | L● |  | 10.2 |  | VMM 공통 시퀀스 |
| `cuMemGetAllocationPropertiesFromHandle` | driver | T2 | T2-VMM-SHARE | primary |  | ● | t |  |  | 10.2 |  |  |
| `cuMemImportFromShareableHandle` | driver | T2 | T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary |  | ● | ● |  |  | 10.2 |  | POSIX fd / FABRIC 핸들 (FABRIC은 IMEX 필요) |
| `cuMemMap` | driver | T2 | T2-VMM-POOL, T2-SLEEP, T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary | ● | ● | ● | L● |  | 10.2 |  | VMM 공통 시퀀스 |
| `cuMemRelease` | driver | T2 | T2-VMM-POOL, T2-SLEEP, T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary | ● | ● | ● | L● |  | 10.2 |  | VMM 공통 시퀀스 |
| `cuMemRetainAllocationHandle` | driver | T2 | T2-VMM-SHARE | primary |  | ● | t |  |  | 11.0 |  | 포인터가 VMM 할당인지 판별 |
| `cuMemSetAccess` | driver | T2 | T2-VMM-POOL, T2-SLEEP, T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary | ● | ● | ● | L● |  | 10.2 |  | VMM 공통 시퀀스 |
| `cuMemUnmap` | driver | T2 | T2-VMM-POOL, T2-SLEEP, T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary | ● | ● | ● | L● |  | 10.2 |  | VMM 공통 시퀀스 |
| `cuModuleGetFunction` | driver | T2 | T2-KLOAD | primary |  |  | ● | M● | A● |  |  |  |
| `cuModuleGetGlobal` | driver | T2 | T2-KLOAD | primary |  |  | t |  |  |  |  |  |
| `cuModuleLoad` | driver | T2 | T2-KLOAD | primary |  |  | t |  | Ac |  |  |  |
| `cuModuleLoadData` | driver | T2 | T2-KLOAD | primary |  |  | ● |  | A● |  |  |  |
| `cuModuleLoadDataEx` | driver | T2 | T2-KLOAD | primary |  |  |  | M● |  |  |  |  |
| `cuModuleUnload` | driver | T2 | T2-KLOAD | primary |  |  | ● | M● | A● |  |  |  |
| `cuOccupancyMaxPotentialClusterSize` | driver | T2 | T2-CLUSTER | alt |  |  | ● |  |  | 11.8 | SM90+ |  |
| `cuStreamWaitValue32` | driver | T2 | T2-STREAMMEM | primary |  | ● |  |  |  |  |  |  |
| `cuStreamWriteValue32` | driver | T2 | T2-STREAMMEM | primary |  | ● |  |  |  |  |  |  |
| `cuTensorMapEncodeTiled` | driver | T2 | T2-TMA | primary |  | ● | ● | M● | A● | 12.0 | SM90+ |  |
| `cuLogicalEndpointBindMem` | driver | T3 | T3-LE | primary |  |  | ● |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuLogicalEndpointCreate` | driver | T3 | T3-LE | primary |  |  | ● |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuLogicalEndpointDestroy` | driver | T3 | T3-LE | primary |  |  | ● |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuLogicalEndpointExport` | driver | T3 | T3-LE | primary |  |  | ● |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuLogicalEndpointIdRelease` | driver | T3 | T3-LE | primary |  |  | ● |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuLogicalEndpointIdReserve` | driver | T3 | T3-LE | primary |  |  | ● |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuLogicalEndpointImport` | driver | T3 | T3-LE | primary |  |  | ● |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuLogicalEndpointQuery` | driver | T3 | T3-LE | primary |  |  | ● |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuLogicalEndpointUnbind` | driver | T3 | T3-LE | primary |  |  | ● |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuMulticastAddDevice` | driver | T3 | T3-NVLS | primary |  |  | ● |  |  | 12.1 | NVSwitch (NVLS) |  |
| `cuMulticastBindMem` | driver | T3 | T3-NVLS | primary |  |  | ● |  |  | 12.1 | NVSwitch (NVLS) |  |
| `cuMulticastCreate` | driver | T3 | T3-NVLS | primary |  |  | ● |  |  | 12.1 | NVSwitch (NVLS) |  |
| `cuMulticastGetGranularity` | driver | T3 | T3-NVLS | primary |  | ● | ● |  |  | 12.1 | NVSwitch (NVLS) |  |
| `cuMulticastUnbind` | driver | T3 | T3-NVLS | primary |  |  | ● |  |  | 12.1 | NVSwitch (NVLS) |  |
| `cuEventElapsedTime` | driver | AUX | AUX-PROF | primary |  |  | t |  |  |  |  |  |

## 5. Runtime API (호스트)

| API | 종류 | 등급 | 기능 | 역할 | vLLM | SGLang | TRT-LLM | Ollama | ExecuTorch | 최소 CUDA | 하드웨어 | 비고 |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|---|---|---|
| `cudaDeviceGetAttribute` | runtime | T0 | T0-CAP | primary | ● | ● | ● | L● M● |  |  |  |  |
| `cudaDeviceGetPCIBusId` | runtime | T0 | T0-CAP, T3-LINKAWARE | primary |  |  | ● | L● |  |  |  | NVML 장치와 매칭 (TRT-LLM KV v2) |
| `cudaDeviceReset` | runtime | T0 | T0-DEV | primary | ● | ● | t |  |  |  |  | P2P 사전 테스트·sleep mode 복원 프로세스 정리 |
| `cudaDeviceSynchronize` | runtime | T0 | T0-STREAM | primary | ● | ● | ● | L● | E● Ah |  |  |  |
| `cudaDriverGetVersion` | runtime | T0 | T0-CAP, T2-VERGATE | primary |  | ● | ● |  |  |  |  |  |
| `cudaEventCreate` | runtime | T0 | T0-STREAM | primary |  |  | ● |  | A● Et |  |  |  |
| `cudaEventCreateWithFlags` | runtime | T0 | T0-STREAM | primary |  |  | ● | L● M● | E● A● |  |  |  |
| `cudaEventDestroy` | runtime | T0 | T0-STREAM | primary |  |  | ● | L● M● | E● A● |  |  |  |
| `cudaEventQuery` | runtime | T0 | T0-STREAM | primary |  |  | ● | M● | A● |  |  |  |
| `cudaEventRecord` | runtime | T0 | T0-STREAM | primary |  | ● | ● | L● M● | E● A● |  |  |  |
| `cudaEventRecordWithFlags` | runtime | T0 | T0-STREAM | primary |  |  | t |  |  |  |  |  |
| `cudaEventSynchronize` | runtime | T0 | T0-STREAM | primary |  |  | ● | L● M● | A● Et |  |  |  |
| `cudaFree` | runtime | T0 | T0-ALLOC | primary | ● | ● | ● | L● M● | E● A● |  |  |  |
| `cudaFreeHost` | runtime | T0 | T0-XFER | primary | ● |  | ● | L● M● | Et |  |  |  |
| `cudaFuncGetAttributes` | runtime | T0 | T0-LAUNCH | primary | ● | ● | ● | L● | E● |  |  |  |
| `cudaFuncSetAttribute` | runtime | T0 | T0-LAUNCH | primary | ● | ● | ● | L● M● |  |  |  |  |
| `cudaGetDevice` | runtime | T0 | T0-DEV | primary | ● | ● | ● | L● M● | E● A● |  |  |  |
| `cudaGetDeviceCount` | runtime | T0 | T0-DEV | primary | ● | ● | ● | L● M● | Et |  |  |  |
| `cudaGetDeviceProperties` | runtime | T0 | T0-CAP | primary | ● | ● | ● | L● M● | E● |  |  |  |
| `cudaGetErrorName` | runtime | T0 | T0-ERR | primary |  |  | t |  |  |  |  |  |
| `cudaGetErrorString` | runtime | T0 | T0-ERR | primary | ● | ● | ● | L● M● | E● A● |  |  |  |
| `cudaGetLastError` | runtime | T0 | T0-ERR | primary | ● | ● | ● | L● M● | E● A● |  |  |  |
| `cudaHostAlloc` | runtime | T0 | T0-XFER, T1-ZEROCOPY, T3-PCIEAR | primary | ● |  | ● | L● |  |  |  | cudaHostAllocMapped로 mapped pinned 메모리 |
| `cudaHostRegister` | runtime | T0 | T0-XFER | primary | ● | ● | ● | L● |  |  |  |  |
| `cudaHostUnregister` | runtime | T0 | T0-XFER | primary | ● | ● | ● | L● |  |  |  |  |
| `cudaLaunchKernel` | runtime | T0 | T0-LAUNCH | primary | ● | ● | ● |  |  |  |  |  |
| `cudaMalloc` | runtime | T0 | T0-ALLOC | primary | ● | ● | ● | L● M● | E● A● |  |  |  |
| `cudaMallocHost` | runtime | T0 | T0-XFER | primary |  |  | ● | L● M● | Et |  |  |  |
| `cudaMemcpy` | runtime | T0 | T0-XFER | primary | ● | ● | ● | M● | E● A● |  |  |  |
| `cudaMemcpy2DAsync` | runtime | T0 | T0-XFER | primary | ● |  | ● | L● |  |  |  |  |
| `cudaMemcpyAsync` | runtime | T0 | T0-XFER | primary | ● | ● | ● | L● M● | E● A● |  |  |  |
| `cudaMemcpyToSymbol` | runtime | T0 | T0-XFER | primary | ● |  | ● |  |  |  |  | constant memory 적재 (vLLM W4A8 LUT) |
| `cudaMemGetInfo` | runtime | T0 | T0-MEMINFO | primary |  |  | ● | L● M● | E● |  |  |  |
| `cudaMemset` | runtime | T0 | T0-ALLOC | primary | ● | ● | ● | L● M● |  |  |  |  |
| `cudaMemsetAsync` | runtime | T0 | T0-ALLOC | primary | ● | ● | ● | L● | E● |  |  |  |
| `cudaPeekAtLastError` | runtime | T0 | T0-ERR | primary |  |  | ● |  |  |  |  |  |
| `cudaPointerGetAttributes` | runtime | T0 | T0-XFER | primary | ● |  | ● |  | E● A● |  |  | pinned·pageable·device 포인터 판별로 복사 경로 선택 |
| `cudaRuntimeGetVersion` | runtime | T0 | T0-CAP, T2-VERGATE | primary | ● | ● |  |  |  |  |  |  |
| `cudaSetDevice` | runtime | T0 | T0-DEV | primary | ● | ● | ● | L● M● | E● A● |  |  |  |
| `cudaSetDeviceFlags` | runtime | T0 | T0-DEV | primary |  |  |  | L● |  |  |  | cudaDeviceScheduleSpin으로 동기화 지연 감소 (llama.cpp) |
| `cudaStreamCreate` | runtime | T0 | T0-STREAM | primary |  |  | ● |  | E● |  |  |  |
| `cudaStreamCreateWithFlags` | runtime | T0 | T0-STREAM | primary |  |  | ● | L● M● | A● Et |  |  |  |
| `cudaStreamDestroy` | runtime | T0 | T0-STREAM | primary |  |  | ● | L● M● | A● Et |  |  |  |
| `cudaStreamQuery` | runtime | T0 | T0-STREAM | primary |  |  | ● |  |  |  |  |  |
| `cudaStreamSynchronize` | runtime | T0 | T0-STREAM | primary | ● |  | ● | L● M● | E● A● |  |  |  |
| `cudaStreamWaitEvent` | runtime | T0 | T0-STREAM | primary |  | ● | ● | L● M● | E● A● |  |  |  |
| `cudaDeviceCanAccessPeer` | runtime | T1 | T1-P2P | primary |  |  | ● | L● |  |  |  |  |
| `cudaDeviceEnablePeerAccess` | runtime | T1 | T1-P2P | primary |  | ● |  | L● |  |  |  |  |
| `cudaDeviceGetDefaultMemPool` | runtime | T1 | T1-POOL | primary |  |  | t | M● |  | 11.2 |  |  |
| `cudaDeviceGetGraphMemAttribute` | runtime | T1 | T1-GRAPH | primary |  |  |  |  | Et |  |  | 그래프 안 할당 메모리 반환·계측 |
| `cudaDeviceGetMemPool` | runtime | T1 | T1-POOL | primary |  |  | t |  | Et | 11.2 |  |  |
| `cudaDeviceGraphMemTrim` | runtime | T1 | T1-GRAPH | primary |  |  |  |  | E● |  |  | 그래프 안 할당 메모리 반환·계측 |
| `cudaFreeAsync` | runtime | T1 | T1-POOL | primary |  |  | ● | M● | E● | 11.2 |  |  |
| `cudaGraphDestroy` | runtime | T1 | T1-GRAPH | primary |  |  | t | L● M● | E● |  |  |  |
| `cudaGraphExecDestroy` | runtime | T1 | T1-GRAPH | primary |  | ● | t | L● M● | E● |  |  |  |
| `cudaGraphExecUpdate` | runtime | T1 | T1-GRAPH | primary |  | ● |  | L● M● |  |  |  |  |
| `cudaGraphInstantiate` | runtime | T1 | T1-GRAPH | primary |  |  | t | L● M● | E● |  |  |  |
| `cudaGraphInstantiateWithFlags` | runtime | T1 | T1-GRAPH | primary |  | ● |  |  |  | 11.4 |  |  |
| `cudaGraphLaunch` | runtime | T1 | T1-GRAPH | primary |  | ● | t | L● M● | E● |  |  |  |
| `cudaHostGetDevicePointer` | runtime | T1 | T1-ZEROCOPY, T3-PCIEAR | primary | ● | ● | ● | L● |  |  |  |  |
| `cudaIpcCloseMemHandle` | runtime | T1 | T1-IPC | primary | ● | ● | ● |  |  |  |  |  |
| `cudaIpcGetMemHandle` | runtime | T1 | T1-IPC | primary | ● | ● | ● |  |  |  |  |  |
| `cudaIpcOpenMemHandle` | runtime | T1 | T1-IPC | primary | ● | ● | ● |  |  |  |  |  |
| `cudaLaunchKernelEx` | runtime | T1 | T1-PDL, T2-CLUSTER | primary | ● | ● | ● | L● |  | 11.8 |  |  |
| `cudaLaunchKernelExC` | runtime | T1 | T1-PDL, T2-CLUSTER | primary |  |  | ● | M● |  | 11.8 |  | C 인터페이스 (attribute 배열 직접 전달) |
| `cudaMallocAsync` | runtime | T1 | T1-POOL | primary |  |  | ● | M● | E● | 11.2 |  |  |
| `cudaMallocFromPoolAsync` | runtime | T1 | T1-POOL | primary |  |  |  |  | E● | 11.2 |  |  |
| `cudaMemcpyBatchAsync` | runtime | T1 | T1-BATCHCOPY | alt |  | ● | ● |  |  | 12.8 |  | dlsym·함수 포인터로 해석 (SGLang) |
| `cudaMemcpyPeerAsync` | runtime | T1 | T1-P2P | primary |  |  |  | L● |  |  |  |  |
| `cudaMemPoolCreate` | runtime | T1 | T1-POOL | primary |  |  | ● |  | E● | 11.2 |  |  |
| `cudaMemPoolDestroy` | runtime | T1 | T1-POOL | primary |  |  | ● |  |  | 11.2 |  |  |
| `cudaMemPoolGetAttribute` | runtime | T1 | T1-POOL | primary |  |  | ● | M● | Et | 11.2 |  |  |
| `cudaMemPoolSetAttribute` | runtime | T1 | T1-POOL | primary |  |  | ● |  | E● | 11.2 |  |  |
| `cudaMemPoolTrimTo` | runtime | T1 | T1-POOL | primary |  |  | ● | M● | E● | 11.2 |  |  |
| `cudaOccupancyAvailableDynamicSMemPerBlock` | runtime | T1 | T1-OCC | primary |  | ● |  |  |  |  |  |  |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | runtime | T1 | T1-OCC | primary | ● | ● | ● | L● M● |  |  |  |  |
| `cudaOccupancyMaxPotentialBlockSize` | runtime | T1 | T1-OCC | primary |  |  |  | M● |  |  |  |  |
| `cudaStreamBeginCapture` | runtime | T1 | T1-GRAPH | primary |  |  | t | L● M● | E● |  |  |  |
| `cudaStreamEndCapture` | runtime | T1 | T1-GRAPH | primary |  |  | t | L● M● | E● |  |  |  |
| `cudaStreamGetCaptureInfo` | runtime | T1 | T1-CAPAWARE | primary | ● | ● | ● |  |  |  |  |  |
| `cudaStreamIsCapturing` | runtime | T1 | T1-CAPAWARE | primary | ● | ● | ● | L● M● |  |  |  |  |
| `cudaThreadExchangeStreamCaptureMode` | runtime | T1 | T1-CAPAWARE | primary | ● |  |  |  |  |  |  |  |
| `cudaDeviceGetStreamPriorityRange` | runtime | T2 | T2-PRIO | primary |  |  | ● |  |  |  |  |  |
| `cudaGetDriverEntryPoint` | runtime | T2 | T2-TMA, T2-VERGATE | primary |  |  | ● |  |  |  |  | Runtime에서 Driver 심볼 해석 (주로 cuTensorMapEncodeTiled). ByVersion은 12.5+ |
| `cudaGetDriverEntryPointByVersion` | runtime | T2 | T2-TMA, T2-VERGATE | primary |  | ● | ● |  |  |  |  | Runtime에서 Driver 심볼 해석 (주로 cuTensorMapEncodeTiled). ByVersion은 12.5+ |
| `cudaGetKernel` | runtime | T2 | T2-KLOAD, T3-LINKAWARE | primary |  |  | ● |  |  |  |  | 런타임 커널 핸들을 Driver로 실행 (TRT-LLM KV v2) |
| `cudaGraphAddChildGraphNode` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  | M● |  |  |  |  |
| `cudaGraphAddDependencies` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  | M● |  |  |  |  |
| `cudaGraphAddEmptyNode` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  | M● |  |  |  |  |
| `cudaGraphAddKernelNode` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  | M● |  |  |  |  |
| `cudaGraphChildGraphNodeGetGraph` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  | M● |  |  |  |  |
| `cudaGraphCreate` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  | t | M● |  |  |  |  |
| `cudaGraphGetNodes` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  | t | M● | E● |  |  |  |
| `cudaGraphKernelNodeGetAttribute` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  | M● |  |  |  |  |
| `cudaGraphKernelNodeGetParams` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  | t |  |  |  |  |  |
| `cudaGraphKernelNodeSetAttribute` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  | M● |  |  |  |  |
| `cudaGraphMemAllocNodeGetParams` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  |  | E● |  |  |  |
| `cudaGraphMemFreeNodeGetParams` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  |  | E● |  |  |  |
| `cudaGraphNodeGetType` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  | t | M● | E● |  |  |  |
| `cudaLaunchCooperativeKernel` | runtime | T2 | T2-COOP | primary |  |  | ● | L● |  |  |  |  |
| `cudaLaunchHostFunc` | runtime | T2 | T2-HOSTFN | primary |  |  | ● | M● | Et |  |  |  |
| `cudaLaunchHostFunc_v2` | runtime | T2 | T2-HOSTFN | primary |  |  | ● |  |  |  |  | TRT-LLM nanobind hostfunc |
| `cudaLibraryLoadData` | runtime | T2 | T2-KLOAD | alt |  | ● |  |  |  |  |  |  |
| `cudaMallocManaged` | runtime | T2 | T2-UVM | primary |  |  | ● | L● M● |  |  |  |  |
| `cudaMemAdvise` | runtime | T2 | T2-UVM | primary |  |  | ● | Lh M● |  |  |  |  |
| `cudaMemPrefetchAsync` | runtime | T2 | T2-UVM | primary |  |  | t |  |  |  |  |  |
| `cudaOccupancyMaxActiveClusters` | runtime | T2 | T2-CLUSTER | primary |  | ● |  |  |  | 11.8 | SM90+ |  |
| `cudaStreamAddCallback` | runtime | T2 | T2-HOSTFN | primary |  |  | ● |  |  |  |  |  |
| `cudaStreamCreateWithPriority` | runtime | T2 | T2-PRIO | primary |  |  | ● |  |  |  |  |  |
| `cudaEventElapsedTime` | runtime | AUX | AUX-PROF | primary |  |  | t |  | Et |  |  |  |
| `cudaGraphDebugDotPrint` | runtime | AUX | AUX-DEBUG | primary |  |  |  | M● |  |  |  |  |
| `cudaProfilerStart` | runtime | AUX | AUX-PROF | primary | t | ● | ● |  |  |  |  |  |
| `cudaProfilerStop` | runtime | AUX | AUX-PROF | primary | t | ● | ● |  |  |  |  |  |

## 6. Runtime API (디바이스 측)

| API | 종류 | 등급 | 기능 | 역할 | vLLM | SGLang | TRT-LLM | Ollama | ExecuTorch | 최소 CUDA | 하드웨어 | 비고 |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|---|---|---|
| `cudaGridDependencySynchronize` | device | T1 | T1-PDL | primary | ● | ● | ● | L● |  |  | SM90+ | 디바이스 측 API |
| `cudaTriggerProgrammaticLaunchCompletion` | device | T1 | T1-PDL | primary | ● | ● | ● | L● |  |  | SM90+ | 디바이스 측 API |
