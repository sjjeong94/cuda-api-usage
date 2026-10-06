# CUDA API 카탈로그

> 이 파일은 `scripts/gen_api_catalog.py`가 `docs/reference/data/*.csv`에서 생성합니다. 직접 고치지 말고 CSV를 고친 뒤 다시 생성하세요.

표기: ● 본체 코드에서 사용, t 테스트·CI·벤치마크에서만 사용, c 설정에 따라 사용, h HIP(ROCm) 빌드 전용.
Ollama의 O/L/M은 Ollama 본체/llama.cpp/MLX, ExecuTorch의 E/A는 ExecuTorch 런타임/AOTI 생성 코드입니다.
PyTorch는 vLLM·SGLang·TRT-LLM이 의존하는 라이브러리라서 함께 실었습니다. PyTorch 열이 ●이면 본체 코드 어딘가에서 쓰고, 어느 구성 요소인지는 `PyTorch 위치` 열에 있습니다.
등급은 그 API가 쓰이는 기능 중 가장 낮은 등급입니다. 역할 `alt`는 같은 기능을 하는 다른 계층(Driver↔Runtime)의 API가 기본이라는 뜻입니다.

## 1. 프레임워크별 집계 (본체 코드, 중복 제외)

| 프레임워크 | Driver | Runtime (호스트) | Runtime (디바이스 측) |
|---|---:|---:|---:|
| vLLM | 16 | 35 | 2 |
| SGLang | 46 | 44 | 2 |
| TRT-LLM | 84 | 72 | 2 |
| Ollama | 26 | 70 | 2 |
| └ llama.cpp | 11 | 48 | 2 |
| └ mlx | 10 | 55 | 0 |
| └ ollama | 8 | 0 | 0 |
| ExecuTorch | 7 | 42 | 0 |
| └ aoti | 7 | 19 | 0 |
| └ runtime | 0 | 37 | 0 |
| PyTorch | 50 | 99 | 0 |
| └ aten | 8 | 39 | 0 |
| └ c10 | 13 | 56 | 0 |
| └ c10d | 19 | 11 | 0 |
| └ inductor | 14 | 19 | 0 |
| └ inductor-gen | 6 | 3 | 0 |
| └ jit | 5 | 0 | 0 |
| └ nativert | 5 | 0 | 0 |
| └ profiler | 2 | 7 | 0 |
| └ torch.cuda | 22 | 35 | 0 |

## 2. 등급별 API 수

| 등급 | Driver | Runtime (호스트) | Runtime (디바이스 측) | 합계 |
|---|---:|---:|---:|---:|
| T0 | 30 | 47 | 0 | 77 |
| T1 | 6 | 43 | 2 | 51 |
| T2 | 75 | 42 | 0 | 117 |
| T3 | 14 | 0 | 0 | 14 |
| AUX | 2 | 5 | 0 | 7 |
| 합계 | 127 | 137 | 2 | 266 |

## 3. 기능별 API

`직접 구현`은 그 기능을 자기 코드에서 CUDA API로 구현한 프레임워크, `위임`은 PyTorch 등에 맡기거나 호출자에게 넘기는 프레임워크입니다.

| 기능 | 등급 | 기본 API | 다른 계층 대응 API | 직접 구현 | 위임 | 가이드 |
|---|:---:|---|---|---|---|---|
| **T0-DEV** 디바이스 열거·선택 | T0 | `cudaDeviceReset`, `cudaGetDevice`, `cudaGetDeviceCount`, `cudaSetDevice`, `cudaSetDeviceFlags` |  | vLLM, SGLang, TRT-LLM, Ollama, ExecuTorch, PyTorch |  | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **T0-CAP** 하드웨어·버전 조회 | T0 | `cudaDeviceGetAttribute`, `cudaDeviceGetPCIBusId`, `cudaDriverGetVersion`, `cudaGetDeviceProperties`, `cudaRuntimeGetVersion` |  | vLLM, SGLang, TRT-LLM, Ollama, ExecuTorch, PyTorch |  | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **T0-MEMINFO** 메모리 용량 산정 | T0 | `cudaMemGetInfo` | `cuMemGetInfo` | TRT-LLM, Ollama, ExecuTorch, PyTorch | vLLM, SGLang | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **T0-ALLOC** 디바이스 메모리 할당·초기화 | T0 | `cudaFree`, `cudaMalloc`, `cudaMemset`, `cudaMemsetAsync` | `cuMemAlloc`, `cuMemFree`, `cuMemsetD32`, `cuMemsetD32Async`, `cuMemsetD8`, `cuMemsetD8Async` | vLLM, SGLang, TRT-LLM, Ollama, ExecuTorch, PyTorch |  | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **T0-XFER** Host↔Device 전송과 pinned 버퍼 | T0 | `cudaFreeHost`, `cudaGetSymbolAddress`, `cudaHostAlloc`, `cudaHostRegister`, `cudaHostUnregister`, `cudaMallocHost`, `cudaMemcpy`, `cudaMemcpy2DAsync`, `cudaMemcpyAsync`, `cudaMemcpyToSymbol`, `cudaPointerGetAttributes` | `cuMemcpyAsync`, `cuMemcpyDtoD`, `cuMemcpyDtoH`, `cuMemcpyDtoHAsync`, `cuMemcpyHtoD`, `cuMemcpyHtoDAsync`, `cuMemFreeHost`, `cuMemHostAlloc`, `cuMemHostRegister`, `cuMemHostUnregister` | vLLM, SGLang, TRT-LLM, Ollama, ExecuTorch, PyTorch |  | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **T0-STREAM** 스트림·이벤트 동기화 | T0 | `cudaDeviceSynchronize`, `cudaEventCreate`, `cudaEventCreateWithFlags`, `cudaEventDestroy`, `cudaEventQuery`, `cudaEventRecord`, `cudaEventRecordWithFlags`, `cudaEventSynchronize`, `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamDestroy`, `cudaStreamQuery`, `cudaStreamSynchronize`, `cudaStreamWaitEvent` | `cuCtxSynchronize`, `cuEventCreate`, `cuEventDestroy`, `cuEventQuery`, `cuEventRecord`, `cuEventSynchronize`, `cuStreamCreate`, `cuStreamDestroy`, `cuStreamSynchronize`, `cuStreamWaitEvent` | TRT-LLM, Ollama, ExecuTorch, PyTorch | vLLM, SGLang | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **T0-LAUNCH** 커널 실행·설정 | T0 | `cudaFuncGetAttributes`, `cudaFuncSetAttribute`, `cudaLaunchKernel` |  | vLLM, SGLang, TRT-LLM, Ollama, ExecuTorch, PyTorch |  | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **T0-ERR** 에러 처리 | T0 | `cudaGetErrorName`, `cudaGetErrorString`, `cudaGetLastError`, `cudaPeekAtLastError` | `cuGetErrorName`, `cuGetErrorString`, `cuLogsRegisterCallback` | vLLM, SGLang, TRT-LLM, Ollama, ExecuTorch, PyTorch |  | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **T1-GRAPH** CUDA Graph (캡처·재실행) | T1 | `cudaDeviceGetGraphMemAttribute`, `cudaDeviceGraphMemTrim`, `cudaDeviceSetGraphMemAttribute`, `cudaGraphDestroy`, `cudaGraphExecDestroy`, `cudaGraphExecUpdate`, `cudaGraphInstantiate`, `cudaGraphInstantiateWithFlags`, `cudaGraphLaunch`, `cudaGraphRetainUserObject`, `cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaUserObjectCreate`, `cudaUserObjectRelease` |  | Ollama, ExecuTorch, PyTorch | vLLM, SGLang, TRT-LLM | [04-execution.md](../guide/04-execution.md) |
| **T1-CAPAWARE** 캡처 인지 동작 | T1 | `cudaStreamGetCaptureInfo`, `cudaStreamIsCapturing`, `cudaThreadExchangeStreamCaptureMode` |  | vLLM, SGLang, TRT-LLM, Ollama, PyTorch |  | [04-execution.md](../guide/04-execution.md) |
| **T1-PDL** PDL (Programmatic Dependent Launch) | T1 | `cudaGridDependencySynchronize`, `cudaLaunchKernelEx`, `cudaLaunchKernelExC`, `cudaTriggerProgrammaticLaunchCompletion`, `cuLaunchKernelEx` |  | vLLM, SGLang, TRT-LLM, Ollama |  | [04-execution.md](../guide/04-execution.md) |
| **T1-OCC** Occupancy 기반 launch 구성 | T1 | `cudaOccupancyAvailableDynamicSMemPerBlock`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaOccupancyMaxPotentialBlockSize` | `cuOccupancyMaxActiveBlocksPerMultiprocessor`, `cuOccupancyMaxPotentialBlockSize` | vLLM, SGLang, TRT-LLM, Ollama, PyTorch |  | [04-execution.md](../guide/04-execution.md) |
| **T1-POOL** 스트림 순서 메모리 풀 | T1 | `cudaDeviceGetDefaultMemPool`, `cudaDeviceGetMemPool`, `cudaFreeAsync`, `cudaMallocAsync`, `cudaMallocFromPoolAsync`, `cudaMemPoolCreate`, `cudaMemPoolDestroy`, `cudaMemPoolGetAttribute`, `cudaMemPoolSetAccess`, `cudaMemPoolSetAttribute`, `cudaMemPoolTrimTo` |  | TRT-LLM, Ollama, ExecuTorch, PyTorch | vLLM, SGLang | [02-memory.md](../guide/02-memory.md) |
| **T1-ZEROCOPY** Mapped host 메모리 (zero-copy) | T1 | `cudaHostAlloc`, `cudaHostGetDevicePointer` | `cuMemHostGetDevicePointer` | vLLM, SGLang, TRT-LLM, Ollama |  | [02-memory.md](../guide/02-memory.md) |
| **T1-BATCHCOPY** 배치 복사 (KV 블록 일괄 전송) | T1 | `cuMemcpyBatchAsync` | `cudaMemcpyBatchAsync` | vLLM, SGLang, TRT-LLM |  | [03-kv-cache.md](../guide/03-kv-cache.md) |
| **T1-IPC** Custom AllReduce (IPC 버퍼 공유) | T1 | `cudaIpcCloseMemHandle`, `cudaIpcGetEventHandle`, `cudaIpcGetMemHandle`, `cudaIpcOpenEventHandle`, `cudaIpcOpenMemHandle`, `cuPointerGetAttribute` |  | vLLM, SGLang, TRT-LLM, PyTorch |  | [06-multi-gpu.md](../guide/06-multi-gpu.md) |
| **T1-P2P** Peer access·P2P 복사 | T1 | `cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess`, `cudaMemcpyPeerAsync`, `cudaMemPoolSetAccess` |  | SGLang, TRT-LLM, Ollama, PyTorch |  | [06-multi-gpu.md](../guide/06-multi-gpu.md) |
| **T2-VMM-POOL** VMM 확장 풀·arena | T2 | `cuDeviceGetAttribute`, `cuMemAddressFree`, `cuMemAddressReserve`, `cuMemCreate`, `cuMemGetAllocationGranularity`, `cuMemMap`, `cuMemRelease`, `cuMemSetAccess`, `cuMemUnmap` |  | SGLang, TRT-LLM, Ollama, PyTorch |  | [02-memory.md](../guide/02-memory.md) |
| **T2-SLEEP** Sleep / Wake (주소 유지 메모리 반납) | T2 | `cuMemAddressFree`, `cuMemAddressReserve`, `cuMemCreate`, `cuMemGetAllocationGranularity`, `cuMemMap`, `cuMemRelease`, `cuMemSetAccess`, `cuMemUnmap` |  | vLLM, TRT-LLM |  | [02-memory.md](../guide/02-memory.md) |
| **T2-VMM-SHARE** VMM 공유 핸들 (프로세스 간 무복사 공유) | T2 | `cuMemAddressFree`, `cuMemAddressReserve`, `cuMemCreate`, `cuMemExportToShareableHandle`, `cuMemGetAddressRange`, `cuMemGetAllocationGranularity`, `cuMemGetAllocationPropertiesFromHandle`, `cuMemImportFromShareableHandle`, `cuMemMap`, `cuMemRelease`, `cuMemRetainAllocationHandle`, `cuMemSetAccess`, `cuMemUnmap` |  | SGLang, TRT-LLM, PyTorch |  | [02-memory.md](../guide/02-memory.md) |
| **T2-UVM** Unified (managed) memory | T2 | `cudaMallocManaged`, `cudaMemAdvise`, `cudaMemPrefetchAsync` |  | TRT-LLM, Ollama, PyTorch |  | [02-memory.md](../guide/02-memory.md) |
| **T2-KLOAD** 외부 CUBIN·JIT 커널 로딩 | T2 | `cuCtxGetId`, `cudaGetKernel`, `cuFuncGetAttribute`, `cuFuncSetAttribute`, `cuFuncSetCacheConfig`, `cuKernelGetFunction`, `cuKernelGetName`, `cuKernelSetAttribute`, `cuLaunchKernel`, `cuLaunchKernelEx`, `cuLibraryEnumerateKernels`, `cuLibraryGetGlobal`, `cuLibraryGetKernel`, `cuLibraryGetKernelCount`, `cuLibraryLoadData`, `cuLibraryUnload`, `cuModuleGetFunction`, `cuModuleGetGlobal`, `cuModuleLoad`, `cuModuleLoadData`, `cuModuleLoadDataEx`, `cuModuleUnload` | `cudaLibraryLoadData`, `cuGraphAddKernelNode`, `cuGraphKernelNodeSetAttribute`, `cuOccupancyMaxPotentialBlockSize` | SGLang, TRT-LLM, Ollama, ExecuTorch, PyTorch |  | [05-kernel-loading.md](../guide/05-kernel-loading.md) |
| **T2-TMA** TMA descriptor | T2 | `cudaGetDriverEntryPoint`, `cudaGetDriverEntryPointByVersion`, `cuTensorMapEncodeTiled` |  | SGLang, TRT-LLM, Ollama, ExecuTorch, PyTorch |  | [04-execution.md](../guide/04-execution.md) |
| **T2-CLUSTER** Thread Block Cluster | T2 | `cudaLaunchKernelEx`, `cudaLaunchKernelExC`, `cudaOccupancyMaxActiveClusters`, `cuLaunchKernelEx` | `cuOccupancyMaxPotentialClusterSize` | SGLang, TRT-LLM, Ollama |  | [04-execution.md](../guide/04-execution.md) |
| **T2-GRAPH-BUILD** 그래프 직접 조립·분석 | T2 | `cudaGraphAddChildGraphNode`, `cudaGraphAddDependencies`, `cudaGraphAddEmptyNode`, `cudaGraphAddKernelNode`, `cudaGraphAddMemAllocNode`, `cudaGraphChildGraphNodeGetGraph`, `cudaGraphCreate`, `cudaGraphEventRecordNodeGetEvent`, `cudaGraphEventWaitNodeGetEvent`, `cudaGraphExecGetId`, `cudaGraphGetEdges`, `cudaGraphGetId`, `cudaGraphGetNodes`, `cudaGraphGetRootNodes`, `cudaGraphHostNodeGetParams`, `cudaGraphKernelNodeGetAttribute`, `cudaGraphKernelNodeGetParams`, `cudaGraphKernelNodeSetAttribute`, `cudaGraphMemAllocNodeGetParams`, `cudaGraphMemFreeNodeGetParams`, `cudaGraphNodeGetDependencies`, `cudaGraphNodeGetDependentNodes`, `cudaGraphNodeGetType` | `cuGraphAddKernelNode`, `cuGraphChildGraphNodeGetGraph`, `cuGraphGetEdges`, `cuGraphGetNodes`, `cuGraphKernelNodeGetAttribute`, `cuGraphKernelNodeGetParams`, `cuGraphKernelNodeSetAttribute`, `cuGraphMemcpyNodeGetParams`, `cuGraphMemsetNodeGetParams`, `cuGraphNodeGetType` | SGLang, Ollama, ExecuTorch, PyTorch |  | [04-execution.md](../guide/04-execution.md) |
| **T2-GRAPH-COND** 그래프 조건 노드 (데이터 의존 제어 흐름) | T2 | `cudaGraphAddNode`, `cudaGraphConditionalHandleCreate`, `cudaStreamBeginCaptureToGraph`, `cudaStreamUpdateCaptureDependencies` |  | PyTorch |  | [04-execution.md](../guide/04-execution.md) |
| **T2-COOP** Cooperative launch (grid-wide sync) | T2 | `cudaLaunchCooperativeKernel` |  | TRT-LLM, Ollama |  | [04-execution.md](../guide/04-execution.md) |
| **T2-HOSTFN** 스트림 순서 호스트 콜백 | T2 | `cudaLaunchHostFunc`, `cudaLaunchHostFunc_v2`, `cudaStreamAddCallback` | `cuLaunchHostFunc` | TRT-LLM, Ollama, PyTorch |  | [04-execution.md](../guide/04-execution.md) |
| **T2-GREEN** Green Context (SM 분할) | T2 | `cuCtxFromGreenCtx`, `cuDeviceGetDevResource`, `cuDevResourceGenerateDesc`, `cuDevSmResourceSplit`, `cuDevSmResourceSplitByCount`, `cuGreenCtxCreate`, `cuGreenCtxDestroy`, `cuGreenCtxGetDevResource`, `cuGreenCtxStreamCreate` | `cuStreamCreate` | SGLang, TRT-LLM, PyTorch | ExecuTorch | [07-scheduling-isolation.md](../guide/07-scheduling-isolation.md) |
| **T2-PRIO** 스트림 우선순위 | T2 | `cudaDeviceGetStreamPriorityRange`, `cudaStreamCreateWithPriority`, `cudaStreamGetPriority` |  | TRT-LLM, PyTorch |  | [07-scheduling-isolation.md](../guide/07-scheduling-isolation.md) |
| **T2-STREAMMEM** 스트림 메모리 연산 (GPU 측 신호) | T2 | `cuStreamWaitValue32`, `cuStreamWriteValue32` |  | SGLang, PyTorch |  | [07-scheduling-isolation.md](../guide/07-scheduling-isolation.md) |
| **T2-DRVDETECT** 드라이버만으로 GPU 탐지 | T2 | `cuDeviceGet`, `cuDeviceGetAttribute`, `cuDeviceGetCount`, `cuDeviceGetName`, `cuDeviceGetPCIBusId`, `cuDeviceGetUuid_v2`, `cuDeviceTotalMem`, `cuDriverGetVersion`, `cuInit` |  | Ollama |  | [08-driver-vs-runtime.md](../guide/08-driver-vs-runtime.md) |
| **T2-VERGATE** 버전 의존 심볼 해석 | T2 | `cudaDriverGetVersion`, `cudaGetDriverEntryPoint`, `cudaGetDriverEntryPointByVersion`, `cudaRuntimeGetVersion`, `cuDriverGetVersion`, `cuGetProcAddress` |  | vLLM, SGLang, TRT-LLM, PyTorch |  | [08-driver-vs-runtime.md](../guide/08-driver-vs-runtime.md) |
| **T2-CTX** 컨텍스트 관리 (Driver·Runtime 혼용) | T2 | `cuCtxCreate`, `cuCtxGetCurrent`, `cuCtxGetDevice`, `cuCtxPopCurrent`, `cuCtxPushCurrent`, `cuCtxSetCurrent`, `cuDevicePrimaryCtxGetState`, `cuDevicePrimaryCtxRetain` |  | vLLM, SGLang, TRT-LLM, PyTorch |  | [08-driver-vs-runtime.md](../guide/08-driver-vs-runtime.md) |
| **T3-NVLS** NVLS 멀티캐스트 | T3 | `cuDeviceGetAttribute`, `cuMemAddressFree`, `cuMemAddressReserve`, `cuMemCreate`, `cuMemExportToShareableHandle`, `cuMemGetAllocationGranularity`, `cuMemImportFromShareableHandle`, `cuMemMap`, `cuMemRelease`, `cuMemSetAccess`, `cuMemUnmap`, `cuMulticastAddDevice`, `cuMulticastBindMem`, `cuMulticastCreate`, `cuMulticastGetGranularity`, `cuMulticastUnbind` |  | TRT-LLM, PyTorch | vLLM, SGLang | [06-multi-gpu.md](../guide/06-multi-gpu.md) |
| **T3-MNNVL** MNNVL fabric 메모리 | T3 | `cuDeviceGetAttribute`, `cuMemAddressFree`, `cuMemAddressReserve`, `cuMemCreate`, `cuMemExportToShareableHandle`, `cuMemGetAddressRange`, `cuMemGetAllocationGranularity`, `cuMemImportFromShareableHandle`, `cuMemMap`, `cuMemRelease`, `cuMemSetAccess`, `cuMemUnmap` |  | TRT-LLM, PyTorch |  | [06-multi-gpu.md](../guide/06-multi-gpu.md) |
| **T3-LE** Logical Endpoint | T3 | `cuLogicalEndpointBindMem`, `cuLogicalEndpointCreate`, `cuLogicalEndpointDestroy`, `cuLogicalEndpointExport`, `cuLogicalEndpointIdRelease`, `cuLogicalEndpointIdReserve`, `cuLogicalEndpointImport`, `cuLogicalEndpointQuery`, `cuLogicalEndpointUnbind` |  | TRT-LLM |  | [06-multi-gpu.md](../guide/06-multi-gpu.md) |
| **T3-LINKAWARE** 링크 인지 복사 전략 | T3 | `cudaDeviceGetPCIBusId`, `cudaGetKernel` |  | TRT-LLM |  | [03-kv-cache.md](../guide/03-kv-cache.md) |
| **T3-PCIEAR** PCIe AllReduce (NVLink 없는 TP) | T3 | `cudaHostAlloc`, `cudaHostGetDevicePointer` |  | Ollama |  | [06-multi-gpu.md](../guide/06-multi-gpu.md) |
| **AUX-PROF** 프로파일링·계측 | AUX | `cudaEventElapsedTime`, `cudaGraphNodeGetToolsId`, `cudaProfilerStart`, `cudaProfilerStop`, `cuEventElapsedTime` |  | SGLang, TRT-LLM, PyTorch |  | [01-t0-essential.md](../guide/01-t0-essential.md) |
| **AUX-DEBUG** 디버깅 | AUX | `cudaGraphDebugDotPrint`, `cuFuncGetName` |  | Ollama, PyTorch |  | [01-t0-essential.md](../guide/01-t0-essential.md) |

## 4. Driver API

| API | 종류 | 등급 | 기능 | 역할 | vLLM | SGLang | TRT-LLM | Ollama | ExecuTorch | PyTorch | PyTorch 위치 | 최소 CUDA | 하드웨어 | 비고 |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|---|---|---|
| `cuCtxSynchronize` | driver | T0 | T0-STREAM | alt |  |  | ● |  | Ac | c |  |  |  |  |
| `cuEventCreate` | driver | T0 | T0-STREAM | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuEventDestroy` | driver | T0 | T0-STREAM | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuEventQuery` | driver | T0 | T0-STREAM | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuEventRecord` | driver | T0 | T0-STREAM | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuEventSynchronize` | driver | T0 | T0-STREAM | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuGetErrorName` | driver | T0 | T0-ERR | alt |  | ● | ● |  |  |  |  |  |  |  |
| `cuGetErrorString` | driver | T0 | T0-ERR | alt | ● | ● | ● | L● M● | A● | ● | aten, c10, inductor, inductor-gen, nativert, torch.cuda |  |  |  |
| `cuLogsRegisterCallback` | driver | T0 | T0-ERR | alt |  |  |  |  |  | ● | c10 | 12.9 |  | 드라이버 에러 로그 콜백 (c10 CUDAException) |
| `cuMemAlloc` | driver | T0 | T0-ALLOC | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuMemcpyAsync` | driver | T0 | T0-XFER | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuMemcpyDtoD` | driver | T0 | T0-XFER | alt |  | ● | ● |  |  |  |  |  |  |  |
| `cuMemcpyDtoH` | driver | T0 | T0-XFER | alt |  | t | ● |  |  |  |  |  |  |  |
| `cuMemcpyDtoHAsync` | driver | T0 | T0-XFER | alt |  |  | t |  |  |  |  |  |  |  |
| `cuMemcpyHtoD` | driver | T0 | T0-XFER | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuMemcpyHtoDAsync` | driver | T0 | T0-XFER | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuMemFree` | driver | T0 | T0-ALLOC | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuMemFreeHost` | driver | T0 | T0-XFER | alt |  |  | t |  |  |  |  |  |  |  |
| `cuMemGetInfo` | driver | T0 | T0-MEMINFO | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuMemHostAlloc` | driver | T0 | T0-XFER | alt |  |  | t |  |  |  |  |  |  |  |
| `cuMemHostRegister` | driver | T0 | T0-XFER | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuMemHostUnregister` | driver | T0 | T0-XFER | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuMemsetD32` | driver | T0 | T0-ALLOC | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuMemsetD32Async` | driver | T0 | T0-ALLOC | alt |  |  |  |  |  | ● | c10d |  |  | symmetric memory signal pad 초기화 |
| `cuMemsetD8` | driver | T0 | T0-ALLOC | alt |  | t | ● |  |  |  |  |  |  |  |
| `cuMemsetD8Async` | driver | T0 | T0-ALLOC | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuStreamCreate` | driver | T0 | T0-STREAM, T2-GREEN | alt |  | ● | ● |  |  |  |  |  |  | green context 구버전 폴백에서도 사용 (SGLang) |
| `cuStreamDestroy` | driver | T0 | T0-STREAM | alt |  |  | ● |  |  | ● | torch.cuda |  |  |  |
| `cuStreamSynchronize` | driver | T0 | T0-STREAM | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuStreamWaitEvent` | driver | T0 | T0-STREAM | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuLaunchKernelEx` | driver | T1 | T2-KLOAD, T1-PDL, T2-CLUSTER | primary |  |  | ● | M● |  |  |  | 12.0 |  |  |
| `cuMemcpyBatchAsync` | driver | T1 | T1-BATCHCOPY | primary | ● |  | ● |  |  |  |  | 12.8 |  | cuGetProcAddress로 해석, 미지원 시 반복 복사 폴백 |
| `cuMemHostGetDevicePointer` | driver | T1 | T1-ZEROCOPY | alt |  |  | t |  |  |  |  |  |  |  |
| `cuOccupancyMaxActiveBlocksPerMultiprocessor` | driver | T1 | T1-OCC | alt |  |  |  |  |  | ● | jit |  |  | JIT 커널(CUfunction)용 (TorchScript fuser) |
| `cuOccupancyMaxPotentialBlockSize` | driver | T1 | T1-OCC, T2-KLOAD | alt |  |  |  | M● |  |  |  |  |  | JIT 커널(CUfunction)용 |
| `cuPointerGetAttribute` | driver | T1 | T1-IPC | primary | ● | ● | t |  |  | ● | inductor |  |  | CU_POINTER_ATTRIBUTE_RANGE_START_ADDR로 캐싱 할당기 블록의 base 주소 |
| `cuCtxCreate` | driver | T2 | T2-CTX | primary |  |  | t |  |  |  |  |  |  |  |
| `cuCtxFromGreenCtx` | driver | T2 | T2-GREEN | primary |  | ● |  |  |  | ● | torch.cuda | 12.4 |  |  |
| `cuCtxGetCurrent` | driver | T2 | T2-CTX | primary | ● |  | ● |  |  | ● | aten, c10d, inductor, profiler, torch.cuda |  |  |  |
| `cuCtxGetDevice` | driver | T2 | T2-CTX | primary | t |  | ● |  |  |  |  |  |  |  |
| `cuCtxGetId` | driver | T2 | T2-KLOAD | primary |  |  | ● |  |  |  |  |  |  | 컨텍스트별 모듈 캐시 키 |
| `cuCtxPopCurrent` | driver | T2 | T2-CTX | primary |  | ● |  |  |  | ● | torch.cuda |  |  |  |
| `cuCtxPushCurrent` | driver | T2 | T2-CTX | primary |  | ● |  |  |  | ● | torch.cuda |  |  |  |
| `cuCtxSetCurrent` | driver | T2 | T2-CTX | primary | ● |  | ● |  |  | ● | aten, c10d, inductor, torch.cuda |  |  |  |
| `cuDeviceGet` | driver | T2 | T2-DRVDETECT | primary | t |  | ● | O● L● |  | ● | inductor, torch.cuda |  |  |  |
| `cuDeviceGetAttribute` | driver | T2 | T2-DRVDETECT, T2-VMM-POOL, T3-NVLS, T3-MNNVL | primary | ● |  | ● | O● L● |  | ● | c10, c10d, inductor |  |  | VMM·멀티캐스트·fabric 지원 여부 확인 |
| `cuDeviceGetCount` | driver | T2 | T2-DRVDETECT | primary | t |  |  | O● |  |  |  |  |  |  |
| `cuDeviceGetDevResource` | driver | T2 | T2-GREEN | primary |  | ● | ● |  |  | ● | torch.cuda | 12.4 |  |  |
| `cuDeviceGetName` | driver | T2 | T2-DRVDETECT | primary |  |  |  | O● |  |  |  |  |  |  |
| `cuDeviceGetPCIBusId` | driver | T2 | T2-DRVDETECT | primary |  |  |  | O● |  |  |  |  |  |  |
| `cuDeviceGetUuid_v2` | driver | T2 | T2-DRVDETECT | primary | t |  |  |  |  |  |  |  |  |  |
| `cuDevicePrimaryCtxGetState` | driver | T2 | T2-CTX | primary |  |  |  |  |  | ● | aten |  |  | primary context가 이미 있는지 확인 (PyTorch lazy init) |
| `cuDevicePrimaryCtxRetain` | driver | T2 | T2-CTX | primary | ● |  | ● |  |  | ● | aten, c10d, inductor |  |  |  |
| `cuDeviceTotalMem` | driver | T2 | T2-DRVDETECT | primary |  |  | ● | O● |  |  |  |  |  |  |
| `cuDevResourceGenerateDesc` | driver | T2 | T2-GREEN | primary |  | ● | ● |  |  | ● | torch.cuda | 12.4 |  |  |
| `cuDevSmResourceSplit` | driver | T2 | T2-GREEN | primary |  |  | ● |  |  |  |  |  |  | TRT-LLM locality domain, cuGetProcAddress로 해석. cuda-python 바인딩은 v13.2.0부터 포함 |
| `cuDevSmResourceSplitByCount` | driver | T2 | T2-GREEN | primary |  | ● |  |  |  | ● | torch.cuda | 12.4 |  | Runtime 대응 cudaDevSmResourceSplitByCount는 CUDA 13.x |
| `cuDriverGetVersion` | driver | T2 | T2-DRVDETECT, T2-VERGATE | primary |  | ● |  | O● |  | ● | torch.cuda |  |  |  |
| `cuFuncGetAttribute` | driver | T2 | T2-KLOAD | primary | t | ● |  |  |  | ● | inductor |  |  | CUfunction의 smem·속성 |
| `cuFuncSetAttribute` | driver | T2 | T2-KLOAD | primary |  |  | ● | M● | A● | ● | inductor, inductor-gen, torch.cuda |  |  | CUfunction의 smem·속성 |
| `cuFuncSetCacheConfig` | driver | T2 | T2-KLOAD | primary |  |  |  |  |  | ● | inductor |  |  | L1/smem 분할 설정 (Inductor static launcher) |
| `cuGetProcAddress` | driver | T2 | T2-VERGATE | primary | ● | ● | ● |  |  |  |  |  |  |  |
| `cuGraphAddKernelNode` | driver | T2 | T2-GRAPH-BUILD, T2-KLOAD | alt |  |  |  | M● |  |  |  |  |  | JIT 커널(CUfunction)을 그래프 노드로 (MLX) |
| `cuGraphChildGraphNodeGetGraph` | driver | T2 | T2-GRAPH-BUILD | alt |  | ● |  |  |  |  |  |  |  | 캡처한 그래프 분석·dedup (SGLang) |
| `cuGraphGetEdges` | driver | T2 | T2-GRAPH-BUILD | alt |  | ● |  |  |  |  |  |  |  | 캡처한 그래프 분석·dedup (SGLang) |
| `cuGraphGetNodes` | driver | T2 | T2-GRAPH-BUILD | alt |  | ● |  |  |  |  |  |  |  | 캡처한 그래프 분석·dedup (SGLang) |
| `cuGraphKernelNodeGetAttribute` | driver | T2 | T2-GRAPH-BUILD | alt |  | ● |  |  |  |  |  |  |  | 캡처한 그래프 분석·dedup (SGLang) |
| `cuGraphKernelNodeGetParams` | driver | T2 | T2-GRAPH-BUILD | alt |  | ● | t |  |  | ● | torch.cuda |  |  | 캡처한 그래프 분석·dedup (SGLang) |
| `cuGraphKernelNodeSetAttribute` | driver | T2 | T2-GRAPH-BUILD, T2-KLOAD | alt |  |  |  | M● |  |  |  |  |  | JIT 커널(CUfunction)을 그래프 노드로 (MLX) |
| `cuGraphMemcpyNodeGetParams` | driver | T2 | T2-GRAPH-BUILD | alt |  | ● |  |  |  |  |  |  |  | 캡처한 그래프 분석·dedup (SGLang) |
| `cuGraphMemsetNodeGetParams` | driver | T2 | T2-GRAPH-BUILD | alt |  | ● |  |  |  |  |  |  |  | 캡처한 그래프 분석·dedup (SGLang) |
| `cuGraphNodeGetType` | driver | T2 | T2-GRAPH-BUILD | alt |  | ● |  |  |  | ● | torch.cuda |  |  | 캡처한 그래프 분석·dedup (SGLang) |
| `cuGreenCtxCreate` | driver | T2 | T2-GREEN | primary |  | ● | ● |  |  | ● | torch.cuda | 12.4 |  | Runtime 대응 cudaGreenCtxCreate는 CUDA 13.x (cuda-python v13.2.0부터) |
| `cuGreenCtxDestroy` | driver | T2 | T2-GREEN | primary |  | ● | ● |  |  | ● | torch.cuda | 12.4 |  |  |
| `cuGreenCtxGetDevResource` | driver | T2 | T2-GREEN | primary |  | ● |  |  |  |  |  | 12.4 |  |  |
| `cuGreenCtxStreamCreate` | driver | T2 | T2-GREEN | primary |  | ● | ● |  |  | ● | torch.cuda | 12.5 |  | cuGetProcAddress로 해석 |
| `cuInit` | driver | T2 | T2-DRVDETECT | primary | t |  | ● | O● |  | ● | profiler |  |  |  |
| `cuKernelGetFunction` | driver | T2 | T2-KLOAD | primary |  | ● |  |  |  |  |  | 12.0 |  | context 독립 library API |
| `cuKernelGetName` | driver | T2 | T2-KLOAD | primary |  |  | ● |  |  |  |  | 12.0 |  | context 독립 library API |
| `cuKernelSetAttribute` | driver | T2 | T2-KLOAD | primary |  |  | ● |  |  |  |  | 12.0 |  | context 독립 library API |
| `cuLaunchHostFunc` | driver | T2 | T2-HOSTFN | alt |  |  | ● |  |  |  |  |  |  |  |
| `cuLaunchKernel` | driver | T2 | T2-KLOAD | primary |  |  | ● |  | A● | ● | aten, inductor, inductor-gen, jit, nativert, torch.cuda |  |  |  |
| `cuLibraryEnumerateKernels` | driver | T2 | T2-KLOAD | primary |  | ● | ● |  |  |  |  | 12.0 |  | context 독립 library API |
| `cuLibraryGetGlobal` | driver | T2 | T2-KLOAD | primary |  |  | ● |  |  |  |  | 12.0 |  | context 독립 library API |
| `cuLibraryGetKernel` | driver | T2 | T2-KLOAD | primary |  |  | ● |  |  |  |  | 12.0 |  | context 독립 library API |
| `cuLibraryGetKernelCount` | driver | T2 | T2-KLOAD | primary |  |  | ● |  |  |  |  | 12.0 |  | context 독립 library API |
| `cuLibraryLoadData` | driver | T2 | T2-KLOAD | primary |  | ● | ● |  |  |  |  | 12.0 |  | context 독립 library API |
| `cuLibraryUnload` | driver | T2 | T2-KLOAD | primary |  |  | ● |  |  |  |  | 12.0 |  | context 독립 library API |
| `cuMemAddressFree` | driver | T2 | T2-VMM-POOL, T2-SLEEP, T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary | ● | ● | ● | L● |  | ● | c10 | 10.2 |  | VMM 공통 시퀀스 |
| `cuMemAddressReserve` | driver | T2 | T2-VMM-POOL, T2-SLEEP, T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary | ● | ● | ● | L● |  | ● | c10, c10d | 10.2 |  | VMM 공통 시퀀스 |
| `cuMemCreate` | driver | T2 | T2-VMM-POOL, T2-SLEEP, T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary | ● | ● | ● | L● |  | ● | c10, c10d | 10.2 |  | VMM 공통 시퀀스 |
| `cuMemExportToShareableHandle` | driver | T2 | T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary |  | ● | ● |  |  | ● | c10, c10d | 10.2 |  | POSIX fd / FABRIC 핸들 (FABRIC은 IMEX 필요). 스트림 순서 풀(cudaMemPool) 메모리는 Runtime cudaMemPoolExportToShareableHandle로도 공유 가능 |
| `cuMemGetAddressRange` | driver | T2 | T2-VMM-SHARE, T3-MNNVL | primary |  | ● | ● |  |  |  |  |  |  | 포인터가 속한 할당의 base·크기 |
| `cuMemGetAllocationGranularity` | driver | T2 | T2-VMM-POOL, T2-SLEEP, T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary | ● | ● | ● | L● |  | ● | c10, c10d | 10.2 |  | VMM 공통 시퀀스 |
| `cuMemGetAllocationPropertiesFromHandle` | driver | T2 | T2-VMM-SHARE | primary |  | ● | t |  |  |  |  | 10.2 |  |  |
| `cuMemImportFromShareableHandle` | driver | T2 | T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary |  | ● | ● |  |  | ● | c10, c10d | 10.2 |  | POSIX fd / FABRIC 핸들 (FABRIC은 IMEX 필요) |
| `cuMemMap` | driver | T2 | T2-VMM-POOL, T2-SLEEP, T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary | ● | ● | ● | L● |  | ● | c10, c10d | 10.2 |  | VMM 공통 시퀀스 |
| `cuMemRelease` | driver | T2 | T2-VMM-POOL, T2-SLEEP, T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary | ● | ● | ● | L● |  | ● | c10, c10d | 10.2 |  | VMM 공통 시퀀스 |
| `cuMemRetainAllocationHandle` | driver | T2 | T2-VMM-SHARE | primary |  | ● | t |  |  |  |  | 11.0 |  | 포인터가 VMM 할당인지 판별 |
| `cuMemSetAccess` | driver | T2 | T2-VMM-POOL, T2-SLEEP, T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary | ● | ● | ● | L● |  | ● | c10, c10d | 10.2 |  | VMM 공통 시퀀스 |
| `cuMemUnmap` | driver | T2 | T2-VMM-POOL, T2-SLEEP, T2-VMM-SHARE, T3-NVLS, T3-MNNVL | primary | ● | ● | ● | L● |  | ● | c10, c10d | 10.2 |  | VMM 공통 시퀀스 |
| `cuModuleGetFunction` | driver | T2 | T2-KLOAD | primary |  |  | ● | M● | A● | ● | aten, inductor, inductor-gen, jit, nativert, torch.cuda |  |  |  |
| `cuModuleGetGlobal` | driver | T2 | T2-KLOAD | primary |  |  | t |  |  |  |  |  |  |  |
| `cuModuleLoad` | driver | T2 | T2-KLOAD | primary |  |  | t |  | Ac | ● | inductor, nativert |  |  |  |
| `cuModuleLoadData` | driver | T2 | T2-KLOAD | primary |  |  | ● |  | A● | ● | aten, inductor-gen, jit, torch.cuda |  |  |  |
| `cuModuleLoadDataEx` | driver | T2 | T2-KLOAD | primary |  |  |  | M● |  |  |  |  |  |  |
| `cuModuleUnload` | driver | T2 | T2-KLOAD | primary |  |  | ● | M● | A● | ● | inductor, jit, nativert |  |  |  |
| `cuOccupancyMaxPotentialClusterSize` | driver | T2 | T2-CLUSTER | alt |  |  | ● |  |  |  |  | 11.8 | SM90+ |  |
| `cuStreamWaitValue32` | driver | T2 | T2-STREAMMEM | primary |  | ● |  |  |  |  |  |  |  |  |
| `cuStreamWriteValue32` | driver | T2 | T2-STREAMMEM | primary |  | ● |  |  |  | ● | c10d |  |  |  |
| `cuTensorMapEncodeTiled` | driver | T2 | T2-TMA | primary |  | ● | ● | M● | A● | ● | inductor-gen | 12.0 | SM90+ |  |
| `cuLogicalEndpointBindMem` | driver | T3 | T3-LE | primary |  |  | ● |  |  |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuLogicalEndpointCreate` | driver | T3 | T3-LE | primary |  |  | ● |  |  |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuLogicalEndpointDestroy` | driver | T3 | T3-LE | primary |  |  | ● |  |  |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuLogicalEndpointExport` | driver | T3 | T3-LE | primary |  |  | ● |  |  |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuLogicalEndpointIdRelease` | driver | T3 | T3-LE | primary |  |  | ● |  |  |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuLogicalEndpointIdReserve` | driver | T3 | T3-LE | primary |  |  | ● |  |  |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuLogicalEndpointImport` | driver | T3 | T3-LE | primary |  |  | ● |  |  |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuLogicalEndpointQuery` | driver | T3 | T3-LE | primary |  |  | ● |  |  |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuLogicalEndpointUnbind` | driver | T3 | T3-LE | primary |  |  | ● |  |  |  |  | 13.4 | NVSwitch fabric + IMEX | cuGetProcAddress로 해석 |
| `cuMulticastAddDevice` | driver | T3 | T3-NVLS | primary |  |  | ● |  |  | ● | c10d | 12.1 | NVSwitch (NVLS) |  |
| `cuMulticastBindMem` | driver | T3 | T3-NVLS | primary |  |  | ● |  |  | ● | c10d | 12.1 | NVSwitch (NVLS) |  |
| `cuMulticastCreate` | driver | T3 | T3-NVLS | primary |  |  | ● |  |  | ● | c10d | 12.1 | NVSwitch (NVLS) |  |
| `cuMulticastGetGranularity` | driver | T3 | T3-NVLS | primary |  | ● | ● |  |  |  |  | 12.1 | NVSwitch (NVLS) |  |
| `cuMulticastUnbind` | driver | T3 | T3-NVLS | primary |  |  | ● |  |  | ● | c10d | 12.1 | NVSwitch (NVLS) |  |
| `cuEventElapsedTime` | driver | AUX | AUX-PROF | primary |  |  | t |  |  |  |  |  |  |  |
| `cuFuncGetName` | driver | AUX | AUX-DEBUG | primary |  |  |  |  |  | ● | torch.cuda |  |  | 그래프 노드의 커널 이름 조회 (torch.cuda.graphs) |

## 5. Runtime API (호스트)

| API | 종류 | 등급 | 기능 | 역할 | vLLM | SGLang | TRT-LLM | Ollama | ExecuTorch | PyTorch | PyTorch 위치 | 최소 CUDA | 하드웨어 | 비고 |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|---|---|---|
| `cudaDeviceGetAttribute` | runtime | T0 | T0-CAP | primary | ● | ● | ● | L● M● |  | ● | aten, c10, torch.cuda |  |  |  |
| `cudaDeviceGetPCIBusId` | runtime | T0 | T0-CAP, T3-LINKAWARE | primary |  |  | ● | L● |  |  |  |  |  | NVML 장치와 매칭 (TRT-LLM KV v2) |
| `cudaDeviceReset` | runtime | T0 | T0-DEV | primary | ● | ● | t |  |  |  |  |  |  | P2P 사전 테스트·sleep mode 복원 프로세스 정리 |
| `cudaDeviceSynchronize` | runtime | T0 | T0-STREAM | primary | ● | ● | ● | L● | E● Ah | ● | aten, c10, c10d, profiler |  |  |  |
| `cudaDriverGetVersion` | runtime | T0 | T0-CAP, T2-VERGATE | primary |  | ● | ● |  |  | ● | c10 |  |  |  |
| `cudaEventCreate` | runtime | T0 | T0-STREAM | primary |  |  | ● |  | A● Et | ● | aten, inductor, profiler |  |  |  |
| `cudaEventCreateWithFlags` | runtime | T0 | T0-STREAM | primary |  |  | ● | L● M● | E● A● | ● | c10, inductor, torch.cuda |  |  |  |
| `cudaEventDestroy` | runtime | T0 | T0-STREAM | primary |  |  | ● | L● M● | E● A● | ● | c10, inductor, profiler, torch.cuda |  |  |  |
| `cudaEventQuery` | runtime | T0 | T0-STREAM | primary |  |  | ● | M● | A● | ● | c10, c10d, inductor |  |  |  |
| `cudaEventRecord` | runtime | T0 | T0-STREAM | primary |  | ● | ● | L● M● | E● A● | ● | aten, c10, inductor, inductor-gen, profiler, torch.cuda |  |  |  |
| `cudaEventRecordWithFlags` | runtime | T0 | T0-STREAM | primary |  |  | t |  |  | ● | c10 |  |  |  |
| `cudaEventSynchronize` | runtime | T0 | T0-STREAM | primary |  |  | ● | L● M● | A● Et | ● | aten, c10, inductor, inductor-gen, profiler |  |  |  |
| `cudaFree` | runtime | T0 | T0-ALLOC | primary | ● | ● | ● | L● M● | E● A● | ● | aten, c10, c10d, inductor, torch.cuda |  |  |  |
| `cudaFreeHost` | runtime | T0 | T0-XFER | primary | ● |  | ● | L● M● | Et | ● | aten |  |  |  |
| `cudaFuncGetAttributes` | runtime | T0 | T0-LAUNCH | primary | ● | ● | ● | L● | E● | ● | aten |  |  |  |
| `cudaFuncSetAttribute` | runtime | T0 | T0-LAUNCH | primary | ● | ● | ● | L● M● |  | ● | aten |  |  |  |
| `cudaGetDevice` | runtime | T0 | T0-DEV | primary | ● | ● | ● | L● M● | E● A● | ● | c10, inductor, torch.cuda |  |  |  |
| `cudaGetDeviceCount` | runtime | T0 | T0-DEV | primary | ● | ● | ● | L● M● | Et | ● | c10, c10d |  |  |  |
| `cudaGetDeviceProperties` | runtime | T0 | T0-CAP | primary | ● | ● | ● | L● M● | E● | ● | aten, c10, c10d |  |  |  |
| `cudaGetErrorName` | runtime | T0 | T0-ERR | primary |  |  | t |  |  | ● | c10 |  |  |  |
| `cudaGetErrorString` | runtime | T0 | T0-ERR | primary | ● | ● | ● | L● M● | E● A● | ● | c10, inductor, profiler, torch.cuda |  |  |  |
| `cudaGetLastError` | runtime | T0 | T0-ERR | primary | ● | ● | ● | L● M● | E● A● | ● | aten, c10, c10d, inductor, torch.cuda |  |  |  |
| `cudaGetSymbolAddress` | runtime | T0 | T0-XFER | primary |  |  |  |  |  | ● | aten |  |  | __device__/__constant__ 심볼의 디바이스 주소 (cuBLAS device pointer mode 상수) |
| `cudaHostAlloc` | runtime | T0 | T0-XFER, T1-ZEROCOPY, T3-PCIEAR | primary | ● |  | ● | L● |  | ● | aten |  |  | cudaHostAllocMapped로 mapped pinned 메모리 |
| `cudaHostRegister` | runtime | T0 | T0-XFER | primary | ● | ● | ● | L● |  | ● | aten, torch.cuda |  |  |  |
| `cudaHostUnregister` | runtime | T0 | T0-XFER | primary | ● | ● | ● | L● |  | ● | aten, torch.cuda |  |  |  |
| `cudaLaunchKernel` | runtime | T0 | T0-LAUNCH | primary | ● | ● | ● |  |  |  |  |  |  |  |
| `cudaMalloc` | runtime | T0 | T0-ALLOC | primary | ● | ● | ● | L● M● | E● A● | ● | c10, inductor |  |  |  |
| `cudaMallocHost` | runtime | T0 | T0-XFER | primary |  |  | ● | L● M● | Et |  |  |  |  |  |
| `cudaMemcpy` | runtime | T0 | T0-XFER | primary | ● | ● | ● | M● | E● A● | ● | c10, c10d, inductor, torch.cuda |  |  |  |
| `cudaMemcpy2DAsync` | runtime | T0 | T0-XFER | primary | ● |  | ● | L● |  |  |  |  |  |  |
| `cudaMemcpyAsync` | runtime | T0 | T0-XFER | primary | ● | ● | ● | L● M● | E● A● | ● | aten, c10, c10d, inductor, torch.cuda |  |  |  |
| `cudaMemcpyToSymbol` | runtime | T0 | T0-XFER | primary | ● |  | ● |  |  | ● | aten |  |  | constant memory 적재 (vLLM W4A8 LUT) |
| `cudaMemGetInfo` | runtime | T0 | T0-MEMINFO | primary |  |  | ● | L● M● | E● | ● | c10, torch.cuda |  |  |  |
| `cudaMemset` | runtime | T0 | T0-ALLOC | primary | ● | ● | ● | L● M● |  | ● | c10d |  |  |  |
| `cudaMemsetAsync` | runtime | T0 | T0-ALLOC | primary | ● | ● | ● | L● | E● | ● | aten, c10d |  |  |  |
| `cudaPeekAtLastError` | runtime | T0 | T0-ERR | primary |  |  | ● |  |  |  |  |  |  |  |
| `cudaPointerGetAttributes` | runtime | T0 | T0-XFER | primary | ● |  | ● |  | E● A● | ● | aten, inductor |  |  | pinned·pageable·device 포인터 판별로 복사 경로 선택 |
| `cudaRuntimeGetVersion` | runtime | T0 | T0-CAP, T2-VERGATE | primary | ● | ● |  |  |  | ● | aten |  |  |  |
| `cudaSetDevice` | runtime | T0 | T0-DEV | primary | ● | ● | ● | L● M● | E● A● | ● | aten, c10, inductor |  |  |  |
| `cudaSetDeviceFlags` | runtime | T0 | T0-DEV | primary |  |  |  | L● |  |  |  |  |  | cudaDeviceScheduleSpin으로 동기화 지연 감소 (llama.cpp) |
| `cudaStreamCreate` | runtime | T0 | T0-STREAM | primary |  |  | ● |  | E● | ● | torch.cuda |  |  |  |
| `cudaStreamCreateWithFlags` | runtime | T0 | T0-STREAM | primary |  |  | ● | L● M● | A● Et | ● | aten, inductor |  |  |  |
| `cudaStreamDestroy` | runtime | T0 | T0-STREAM | primary |  |  | ● | L● M● | A● Et | ● | aten, inductor, torch.cuda |  |  |  |
| `cudaStreamQuery` | runtime | T0 | T0-STREAM | primary |  |  | ● |  |  | ● | c10 |  |  |  |
| `cudaStreamSynchronize` | runtime | T0 | T0-STREAM | primary | ● |  | ● | L● M● | E● A● | ● | c10, c10d, inductor |  |  |  |
| `cudaStreamWaitEvent` | runtime | T0 | T0-STREAM | primary |  | ● | ● | L● M● | E● A● | ● | c10, inductor-gen |  |  |  |
| `cudaDeviceCanAccessPeer` | runtime | T1 | T1-P2P | primary |  |  | ● | L● |  | ● | aten, c10 |  |  |  |
| `cudaDeviceEnablePeerAccess` | runtime | T1 | T1-P2P | primary |  | ● |  | L● |  | ● | c10, torch.cuda |  |  |  |
| `cudaDeviceGetDefaultMemPool` | runtime | T1 | T1-POOL | primary |  |  | t | M● |  | ● | c10 | 11.2 |  |  |
| `cudaDeviceGetGraphMemAttribute` | runtime | T1 | T1-GRAPH | primary |  |  |  |  | Et | ● | c10 |  |  | 그래프 안 할당 메모리 반환·계측 |
| `cudaDeviceGetMemPool` | runtime | T1 | T1-POOL | primary |  |  | t |  | Et |  |  | 11.2 |  |  |
| `cudaDeviceGraphMemTrim` | runtime | T1 | T1-GRAPH | primary |  |  |  |  | E● |  |  |  |  | 그래프 안 할당 메모리 반환·계측 |
| `cudaDeviceSetGraphMemAttribute` | runtime | T1 | T1-GRAPH | primary |  |  |  |  |  | ● | c10 |  |  | 그래프 메모리 high watermark 초기화 |
| `cudaFreeAsync` | runtime | T1 | T1-POOL | primary |  |  | ● | M● | E● | ● | c10 | 11.2 |  |  |
| `cudaGraphDestroy` | runtime | T1 | T1-GRAPH | primary |  |  | t | L● M● | E● | ● | aten |  |  |  |
| `cudaGraphExecDestroy` | runtime | T1 | T1-GRAPH | primary |  | ● | t | L● M● | E● | ● | aten |  |  |  |
| `cudaGraphExecUpdate` | runtime | T1 | T1-GRAPH | primary |  | ● |  | L● M● |  |  |  |  |  |  |
| `cudaGraphInstantiate` | runtime | T1 | T1-GRAPH | primary |  |  | t | L● M● | E● |  |  |  |  |  |
| `cudaGraphInstantiateWithFlags` | runtime | T1 | T1-GRAPH | primary |  | ● |  |  |  | ● | aten | 11.4 |  |  |
| `cudaGraphLaunch` | runtime | T1 | T1-GRAPH | primary |  | ● | t | L● M● | E● | ● | aten |  |  |  |
| `cudaGraphRetainUserObject` | runtime | T1 | T1-GRAPH | primary |  |  |  |  |  | ● | c10 |  |  | 리소스 수명을 그래프에 묶음 |
| `cudaHostGetDevicePointer` | runtime | T1 | T1-ZEROCOPY, T3-PCIEAR | primary | ● | ● | ● | L● |  |  |  |  |  |  |
| `cudaIpcCloseMemHandle` | runtime | T1 | T1-IPC | primary | ● | ● | ● |  |  | ● | c10 |  |  |  |
| `cudaIpcGetEventHandle` | runtime | T1 | T1-IPC | primary |  |  |  |  |  | ● | c10, torch.cuda |  |  | 프로세스 간 이벤트 공유 (torch.multiprocessing) |
| `cudaIpcGetMemHandle` | runtime | T1 | T1-IPC | primary | ● | ● | ● |  |  | ● | c10 |  |  |  |
| `cudaIpcOpenEventHandle` | runtime | T1 | T1-IPC | primary |  |  |  |  |  | ● | c10 |  |  | 프로세스 간 이벤트 공유 (torch.multiprocessing) |
| `cudaIpcOpenMemHandle` | runtime | T1 | T1-IPC | primary | ● | ● | ● |  |  | ● | c10 |  |  |  |
| `cudaLaunchKernelEx` | runtime | T1 | T1-PDL, T2-CLUSTER | primary | ● | ● | ● | L● |  |  |  | 11.8 |  |  |
| `cudaLaunchKernelExC` | runtime | T1 | T1-PDL, T2-CLUSTER | primary |  |  | ● | M● |  |  |  | 11.8 |  | C 인터페이스 (attribute 배열 직접 전달) |
| `cudaMallocAsync` | runtime | T1 | T1-POOL | primary |  |  | ● | M● | E● | ● | c10 | 11.2 |  |  |
| `cudaMallocFromPoolAsync` | runtime | T1 | T1-POOL | primary |  |  |  |  | E● |  |  | 11.2 |  |  |
| `cudaMemcpyBatchAsync` | runtime | T1 | T1-BATCHCOPY | alt |  | ● | ● |  |  |  |  | 12.8 |  | dlsym·함수 포인터로 해석 (SGLang) |
| `cudaMemcpyPeerAsync` | runtime | T1 | T1-P2P | primary |  |  |  | L● |  | ● | c10 |  |  |  |
| `cudaMemPoolCreate` | runtime | T1 | T1-POOL | primary |  |  | ● |  | E● |  |  | 11.2 |  |  |
| `cudaMemPoolDestroy` | runtime | T1 | T1-POOL | primary |  |  | ● |  |  |  |  | 11.2 |  |  |
| `cudaMemPoolGetAttribute` | runtime | T1 | T1-POOL | primary |  |  | ● | M● | Et | ● | c10 | 11.2 |  |  |
| `cudaMemPoolSetAccess` | runtime | T1 | T1-POOL, T1-P2P | primary |  |  |  |  |  | ● | c10 | 11.2 |  | 메모리 풀을 다른 GPU에서 접근 |
| `cudaMemPoolSetAttribute` | runtime | T1 | T1-POOL | primary |  |  | ● |  | E● | ● | c10 | 11.2 |  |  |
| `cudaMemPoolTrimTo` | runtime | T1 | T1-POOL | primary |  |  | ● | M● | E● | ● | c10 | 11.2 |  |  |
| `cudaOccupancyAvailableDynamicSMemPerBlock` | runtime | T1 | T1-OCC | primary |  | ● |  |  |  |  |  |  |  |  |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | runtime | T1 | T1-OCC | primary | ● | ● | ● | L● M● |  | ● | aten |  |  |  |
| `cudaOccupancyMaxPotentialBlockSize` | runtime | T1 | T1-OCC | primary |  |  |  | M● |  | ● | aten |  |  |  |
| `cudaStreamBeginCapture` | runtime | T1 | T1-GRAPH | primary |  |  | t | L● M● | E● | ● | aten |  |  |  |
| `cudaStreamEndCapture` | runtime | T1 | T1-GRAPH | primary |  |  | t | L● M● | E● | ● | aten |  |  |  |
| `cudaStreamGetCaptureInfo` | runtime | T1 | T1-CAPAWARE | primary | h | ● | ● |  |  | ● | aten, c10, torch.cuda |  |  |  |
| `cudaStreamIsCapturing` | runtime | T1 | T1-CAPAWARE | primary | ● | ● | ● | L● M● |  | ● | c10, inductor |  |  |  |
| `cudaThreadExchangeStreamCaptureMode` | runtime | T1 | T1-CAPAWARE | primary | ● |  |  |  |  | ● | c10 |  |  |  |
| `cudaUserObjectCreate` | runtime | T1 | T1-GRAPH | primary |  |  |  |  |  | ● | c10 |  |  | 리소스 수명을 그래프에 묶음 |
| `cudaUserObjectRelease` | runtime | T1 | T1-GRAPH | primary |  |  |  |  |  | ● | c10 |  |  | 리소스 수명을 그래프에 묶음 |
| `cudaDeviceGetStreamPriorityRange` | runtime | T2 | T2-PRIO | primary |  |  | ● |  |  | ● | c10 |  |  |  |
| `cudaGetDriverEntryPoint` | runtime | T2 | T2-TMA, T2-VERGATE | primary |  |  | ● |  |  | ● | c10 |  |  | Runtime에서 Driver 심볼 해석 (주로 cuTensorMapEncodeTiled). ByVersion은 12.5+ |
| `cudaGetDriverEntryPointByVersion` | runtime | T2 | T2-TMA, T2-VERGATE | primary |  | ● | ● |  |  | ● | c10 |  |  | Runtime에서 Driver 심볼 해석 (주로 cuTensorMapEncodeTiled). ByVersion은 12.5+ |
| `cudaGetKernel` | runtime | T2 | T2-KLOAD, T3-LINKAWARE | primary |  |  | ● |  |  |  |  |  |  | 런타임 커널 핸들을 Driver로 실행 (TRT-LLM KV v2) |
| `cudaGraphAddChildGraphNode` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  | M● |  |  |  |  |  |  |
| `cudaGraphAddDependencies` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  | M● |  |  |  |  |  |  |
| `cudaGraphAddEmptyNode` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  | M● |  |  |  |  |  |  |
| `cudaGraphAddKernelNode` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  | M● |  |  |  |  |  |  |
| `cudaGraphAddMemAllocNode` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  |  |  | t |  |  |  |  |
| `cudaGraphAddNode` | runtime | T2 | T2-GRAPH-COND | primary |  |  |  |  |  | ● | aten |  |  | 조건 노드 본문을 캡처 (torch.cond·while_loop를 그래프 안에서) |
| `cudaGraphChildGraphNodeGetGraph` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  | M● |  |  |  |  |  |  |
| `cudaGraphConditionalHandleCreate` | runtime | T2 | T2-GRAPH-COND | primary |  |  |  |  |  | ● | aten | 12.3 |  | 조건 노드 본문을 캡처 (torch.cond·while_loop를 그래프 안에서) |
| `cudaGraphCreate` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  | t | M● |  | t |  |  |  |  |
| `cudaGraphEventRecordNodeGetEvent` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  |  |  | ● | torch.cuda |  |  | 캡처한 그래프 분석 (torch.cuda.graphs, cuda-python) |
| `cudaGraphEventWaitNodeGetEvent` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  |  |  | ● | torch.cuda |  |  | 캡처한 그래프 분석 (torch.cuda.graphs, cuda-python) |
| `cudaGraphExecGetId` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  |  |  | ● | torch.cuda |  |  | 캡처한 그래프 분석 (torch.cuda.graphs, cuda-python) |
| `cudaGraphGetEdges` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  |  |  | ● | torch.cuda |  |  | 캡처한 그래프 분석 (torch.cuda.graphs, cuda-python) |
| `cudaGraphGetId` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  |  |  | ● | torch.cuda |  |  | 캡처한 그래프 분석 (torch.cuda.graphs, cuda-python) |
| `cudaGraphGetNodes` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  | t | M● | E● | ● | aten, torch.cuda |  |  |  |
| `cudaGraphGetRootNodes` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  |  |  | ● | torch.cuda |  |  | 캡처한 그래프 분석 (torch.cuda.graphs, cuda-python) |
| `cudaGraphHostNodeGetParams` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  |  |  | ● | torch.cuda |  |  | 캡처한 그래프 분석 (torch.cuda.graphs, cuda-python) |
| `cudaGraphKernelNodeGetAttribute` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  | M● |  |  |  |  |  |  |
| `cudaGraphKernelNodeGetParams` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  | t |  |  |  |  |  |  |  |
| `cudaGraphKernelNodeSetAttribute` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  | M● |  |  |  |  |  |  |
| `cudaGraphMemAllocNodeGetParams` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  |  | E● |  |  |  |  |  |
| `cudaGraphMemFreeNodeGetParams` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  |  | E● |  |  |  |  |  |
| `cudaGraphNodeGetDependencies` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  |  |  | ● | c10 |  |  | 캡처 중 블록 재사용 판단 (CUDACachingAllocator) |
| `cudaGraphNodeGetDependentNodes` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  |  |  |  | ● | torch.cuda |  |  | 캡처한 그래프 분석 (torch.cuda.graphs, cuda-python) |
| `cudaGraphNodeGetType` | runtime | T2 | T2-GRAPH-BUILD | primary |  |  | t | M● | E● | ● | torch.cuda |  |  |  |
| `cudaLaunchCooperativeKernel` | runtime | T2 | T2-COOP | primary |  |  | ● | L● |  |  |  |  |  |  |
| `cudaLaunchHostFunc` | runtime | T2 | T2-HOSTFN | primary |  |  | ● | M● | Et | ● | torch.cuda |  |  |  |
| `cudaLaunchHostFunc_v2` | runtime | T2 | T2-HOSTFN | primary |  |  | ● |  |  |  |  |  |  | TRT-LLM nanobind hostfunc |
| `cudaLibraryLoadData` | runtime | T2 | T2-KLOAD | alt |  | ● |  |  |  |  |  |  |  | Runtime library API (cuda-python 바인딩은 v12.8.0부터 포함) |
| `cudaMallocManaged` | runtime | T2 | T2-UVM | primary |  |  | ● | L● M● |  | ● | c10, torch.cuda |  |  |  |
| `cudaMemAdvise` | runtime | T2 | T2-UVM | primary |  |  | ● | Lh M● |  | ● | c10, torch.cuda |  |  |  |
| `cudaMemPrefetchAsync` | runtime | T2 | T2-UVM | primary |  |  | t |  |  |  |  |  |  |  |
| `cudaOccupancyMaxActiveClusters` | runtime | T2 | T2-CLUSTER | primary |  | ● |  |  |  |  |  | 11.8 | SM90+ |  |
| `cudaStreamAddCallback` | runtime | T2 | T2-HOSTFN | primary |  |  | ● |  |  |  |  |  |  |  |
| `cudaStreamBeginCaptureToGraph` | runtime | T2 | T2-GRAPH-COND | primary |  |  |  |  |  | ● | aten | 12.3 |  | 조건 노드 본문을 캡처 (torch.cond·while_loop를 그래프 안에서) |
| `cudaStreamCreateWithPriority` | runtime | T2 | T2-PRIO | primary |  |  | ● |  |  | ● | c10 |  |  |  |
| `cudaStreamGetPriority` | runtime | T2 | T2-PRIO | primary |  |  |  |  |  | ● | c10 |  |  |  |
| `cudaStreamUpdateCaptureDependencies` | runtime | T2 | T2-GRAPH-COND | primary |  |  |  |  |  | ● | aten |  |  | 조건 노드 본문을 캡처 (torch.cond·while_loop를 그래프 안에서) |
| `cudaEventElapsedTime` | runtime | AUX | AUX-PROF | primary |  |  | t |  | Et | ● | aten, c10, profiler |  |  |  |
| `cudaGraphDebugDotPrint` | runtime | AUX | AUX-DEBUG | primary |  |  |  | M● |  | ● | torch.cuda |  |  |  |
| `cudaGraphNodeGetToolsId` | runtime | AUX | AUX-PROF | primary |  |  |  |  |  | ● | torch.cuda |  |  | 그래프 노드와 프로파일러 이벤트 연결 |
| `cudaProfilerStart` | runtime | AUX | AUX-PROF | primary | t | ● | ● |  |  | ● | torch.cuda |  |  |  |
| `cudaProfilerStop` | runtime | AUX | AUX-PROF | primary | t | ● | ● |  |  | ● | torch.cuda |  |  |  |

## 6. Runtime API (디바이스 측)

| API | 종류 | 등급 | 기능 | 역할 | vLLM | SGLang | TRT-LLM | Ollama | ExecuTorch | PyTorch | PyTorch 위치 | 최소 CUDA | 하드웨어 | 비고 |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|---|---|---|
| `cudaGridDependencySynchronize` | device | T1 | T1-PDL | primary | ● | ● | ● | L● |  |  |  |  | SM90+ | 디바이스 측 API |
| `cudaTriggerProgrammaticLaunchCompletion` | device | T1 | T1-PDL | primary | ● | ● | ● | L● |  |  |  |  | SM90+ | 디바이스 측 API |
