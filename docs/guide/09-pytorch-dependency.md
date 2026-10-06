# 09. PyTorch 의존: [PT] 프로파일에서 직접 해야 하는 것

vLLM, SGLang, TensorRT-LLM은 PyTorch 위에서 동작합니다. 이 장은 **PyTorch를 쓰면 어떤 CUDA API를 직접 부르지 않아도 되고, 무엇은 여전히 직접 해야 하는지**를 기능별로 정리합니다. 근거는 PyTorch `v2.14.0` 분석([evidence](../evidence/pytorch_cuda_api_usage.md))입니다.

PyTorch는 본체 코드에서 Driver API 50개, Runtime API 99개를 씁니다. 개수로는 SGLang보다 많지만, 이 API들이 서빙 엔진에 필요한 기능을 모두 덮지는 않습니다.

---

## 1. 기능별 정리: PyTorch가 주는 것과 남는 것

| 기능 | 등급 | PyTorch가 제공 (Python API) | PyTorch 안의 CUDA API | [PT] 엔진이 직접 할 일 |
|---|:---:|---|---|---|
| 디바이스·속성·버전 | T0 | `torch.cuda.device_count`, `get_device_properties`, `set_device` | `cudaGetDeviceCount`, `cudaGetDeviceProperties`(캐시), `cudaSetDevice` | 없음 |
| 메모리 용량 산정 | T0 | `torch.cuda.mem_get_info()` | `cudaMemGetInfo` | 없음 |
| 디바이스 메모리 | T0 | 캐싱 할당기 (`torch.empty` 등) | `cudaMalloc` + 블록 캐시, 이벤트 기반 재사용 | 없음 |
| Pinned host 버퍼 | T0 | `pin_memory()`, `torch.cuda.cudart().cudaHostRegister` | `cudaHostAlloc` / `cudaHostRegister` | 큰 기존 영역(KV host pool, mmap)을 청크 단위로 pin하는 정책 |
| 스트림·이벤트 | T0 | `torch.cuda.Stream`, `Event` | 우선순위별 스트림 풀, `cudaEventCreateWithFlags` | 없음 |
| 커널 실행·smem 설정 | T0 | (자기 커널은 직접) | — | **직접 작성한 커널의 `cudaFuncSetAttribute`, launch** |
| CUDA Graph | T1 | `torch.cuda.CUDAGraph`, `torch.cuda.graph`, `graph_pool_handle` | `cudaStreamBeginCapture`, `cudaGraphInstantiateWithFlags(AutoFreeOnLaunch\|UseNodePriority)`, `cudaGraphLaunch` | batch 크기별 캡처·padding 정책, 직접 만든 코드의 **캡처 인지 동작** |
| 캡처 인지 동작 | T1 | `torch.cuda.is_current_stream_capturing()` | `cudaStreamIsCapturing`, `cudaThreadExchangeStreamCaptureMode` | Custom AllReduce처럼 캡처 중 IPC 버퍼를 다루는 C++ 코드는 **직접** `cudaStreamIsCapturing` 등을 호출 (vLLM, SGLang) |
| PDL | T1 | **없음** (ATen 커널은 PDL을 쓰지 않음) | — | **직접** `cudaLaunchKernelEx` + 디바이스 API |
| Occupancy 기반 구성 | T1 | 없음 | — | **직접** |
| 스트림 순서 메모리 풀 | T1 | `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` | `cudaMallocAsync`, `cudaMemPoolTrimTo` … | 없음 (캐싱 할당기로 충분한 경우가 많음) |
| KV 배치 복사 | T1 | **없음** | — | **직접** `cuMemcpyBatchAsync` / `cudaMemcpyBatchAsync` (vLLM, SGLang, TRT-LLM) |
| Custom AllReduce (IPC) | T1 | symmetric memory AllReduce (`torch.ops.symm_mem.one_shot_all_reduce`, `two_shot_all_reduce_`), `torch.multiprocessing` 텐서 공유(용도가 다름) | `cuMemCreate`·`cuMemExportToShareableHandle` (symmetric memory), `cudaIpcGetMemHandle` | PyTorch symmetric memory로 위임하거나(vLLM `symm_mem.py`, SGLang `torch_symm_mem.py`), **직접** `cudaIpc*` + `cuPointerGetAttribute` (vLLM, SGLang은 ctypes, TRT-LLM) |
| VMM 확장 풀 | T2 | `expandable_segments:True` | `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap` … | 단편화만 줄이면 되면 이걸로 충분. **arena·공유·Sleep이 필요하면 직접** |
| Sleep / Wake | T2 | **없음** (`MemPool`·pluggable allocator로 연결 지점만 제공) | — | **직접** VMM 할당기를 만들고 `CUDAPluggableAllocator` / `MemPool`로 꽂음 (vLLM `CuMemAllocator`) |
| VMM 공유 핸들 | T2 | expandable segment 공유(POSIX fd·FABRIC), symmetric memory | `cuMemExportToShareableHandle` … | 자기 버퍼를 원하는 방식으로 공유하려면 **직접** (SGLang, TRT-LLM) |
| 외부 CUBIN·JIT 로딩 | T2 | `torch.compile`(Triton), `torch.cuda._compile_kernel`(NVRTC) | `cuModuleLoad(Data)`, `cuLaunchKernel` | 자체 JIT·CUBIN 체계(XQA, DeepGEMM)는 **직접** |
| TMA | T2 | Triton/Inductor 생성 코드 안에서만 | `cuTensorMapEncodeTiled` (생성 코드) | 직접 작성한 Hopper 커널은 **직접** |
| 그래프 분석 | T2 | `torch.cuda.graphs`의 분석 기능 (cuda-python) | `cudaGraphGetNodes`, `cudaGraphGetEdges` … | dedup처럼 분석 결과로 exec를 공유하는 최적화는 **직접** (SGLang) |
| 그래프 조건 노드 | T2 | `CUDAGraph.begin_capture_to_conditional_node()` | `cudaGraphConditionalHandleCreate`, `cudaStreamBeginCaptureToGraph` | 없음 |
| Green Context | T2 | `torch.cuda.green_contexts.GreenContext` | `cuGreenCtxCreate`, `cuDevSmResourceSplitByCount`, work queue 설정 | 이걸로 충분. SGLang·TRT-LLM은 자체 C++ 구현 |
| 스트림 메모리 연산 | T2 | symmetric memory 내부에서만 | `cuStreamWriteValue32` | 프로세스 간 신호는 **직접** (SGLang `cuStreamWaitValue32`) |
| NVLS 멀티캐스트 | T3 | `torch.distributed._symmetric_memory` (`multimem_all_reduce_`) | `cuMulticastCreate`, `cuMulticastBindMem` … | 이걸로 충분. vLLM·SGLang은 이걸 쓰고, TRT-LLM만 자체 구현 |
| MNNVL / Logical Endpoint | T3 | FABRIC 핸들 지원 확인, expandable segment FABRIC 공유 | `cuMemCreate(FABRIC)` | Logical Endpoint는 **직접** (TRT-LLM) |

### 한 줄 요약

- **PyTorch가 대신해 주는 것**: T0 전체(자기 커널 launch는 제외), CUDA Graph 캡처·실행, 할당기, pinned 메모리, 스트림·이벤트.
- **PyTorch가 제공하지만 서빙 엔진은 보통 직접 구현하는 것**: VMM, green context. PyTorch 구현은 범용이라, sleep/wake, KV arena, PD multiplexing 같은 서빙 전용 요구를 맞추기 어렵기 때문입니다.
- **PyTorch에 맡기기도 하고 직접 구현하기도 하는 것**: AllReduce와 NVLS 멀티캐스트. vLLM·SGLang은 PyTorch symmetric memory 경로와 자체 IPC Custom AllReduce를 함께 두고, TRT-LLM은 둘 다 자체 구현합니다.
- **PyTorch에 없어서 반드시 직접 하는 것**: PDL, KV 배치 복사, Sleep/Wake, 자체 JIT·CUBIN 로딩, 그래프 dedup, Logical Endpoint.

---

## 2. PyTorch와 함께 쓸 때 지켜야 할 규칙

PyTorch의 CUDA 사용 방식에서 나오는 제약입니다. 위반하면 crash나 잘못된 결과가 아니라 **조용한 성능 저하나 간헐적 오류**로 나타나기 쉽습니다.

### 2.1 Primary context를 공유하세요

PyTorch는 primary context를 씁니다. cuBLAS·cuFFT·Triton을 부르기 전에 현재 스레드에 context가 없으면 `cuDevicePrimaryCtxRetain` → `cuCtxSetCurrent`로 설정합니다(`CublasHandlePool.cpp`는 이때 경고를 출력). 자체 Driver 코드에서 `cuCtxCreate`로 별도 context를 만들면, 거기서 만든 메모리와 스트림은 PyTorch 텐서·스트림과 섞어 쓸 수 없습니다. → [08 컨텍스트 관리](08-driver-vs-runtime.md#3-t2-ctx-컨텍스트-관리-driverruntime-혼용)

### 2.2 PyTorch의 현재 스트림을 따르세요

직접 만든 커널이나 복사는 `at::cuda::getCurrentCUDAStream()`(Python에서는 `torch.cuda.current_stream().cuda_stream`)에 제출하세요. 다른 스트림을 쓰면 다음이 깨집니다.
- 캐싱 할당기의 재사용 판단: 다른 스트림에서 텐서를 쓰면 `tensor.record_stream(s)`로 알려야 합니다. 그래야 할당기가 그 스트림의 이벤트를 기록하고 끝날 때까지 블록을 재사용하지 않습니다.
- CUDA Graph 캡처: 캡처는 현재 스트림을 기준으로 이뤄집니다. 다른 스트림의 작업은 이벤트로 fork·join해야 그래프에 들어갑니다.

### 2.3 캡처 모드를 이해하세요

`torch.cuda.graph`의 기본 캡처 모드는 `cudaStreamCaptureModeGlobal`(가장 보수적)입니다. 이 모드에서는 캡처가 진행되는 동안 캡처하는 스레드뿐 아니라 **다른 스레드**에서도 캡처에 안전하지 않은 CUDA API(예: `cudaMalloc`)를 부르면 에러가 납니다. 캡처 중에 할당해야 하는 코드는 `cudaThreadExchangeStreamCaptureMode(Relaxed)`로 잠깐 모드를 바꿉니다. PyTorch의 pinned 할당기와 vLLM의 Custom AllReduce가 이렇게 합니다.

### 2.4 자체 할당기는 PyTorch에 연결하세요

VMM 할당기(Sleep/Wake, KV arena)를 직접 만들었다면, 그 메모리를 PyTorch 텐서로 쓰기 위해 다음 중 하나를 씁니다.

| 방법 | 쓰임 | 근거 |
|---|---|---|
| `torch.cuda.memory.CUDAPluggableAllocator` + `torch.cuda.MemPool` + `use_mem_pool` | 특정 코드 구간의 할당만 자체 할당기로 | vLLM `CuMemAllocator`, SGLang `kv_vmm_backing.py` |
| `torch.cuda.memory.change_current_allocator` | 프로세스 전체 할당기 교체 (첫 CUDA 할당 전에만) | — |
| DLPack, C++ `torch::from_blob` | 이미 있는 포인터를 텐서로 감싸기 (할당기를 거치지 않으므로 수명은 직접 관리) | — |

### 2.5 그래프 메모리 풀을 공유하세요

여러 그래프(batch 크기별, prefill·decode)를 캡처할 때 `graph_pool_handle()`로 풀 하나를 공유하면, 동시에 replay하지 않는 그래프끼리 메모리를 겹쳐 씁니다(SGLang `runner_utils/pool.py`). 기본 캐싱 할당기에서 그래프 안의 할당은 이 private pool에서 나오고, 그래프가 살아 있는 동안 풀이 유지됩니다. 그래서 풀을 공유한 그래프를 동시에 replay하면 안 됩니다. PyTorch가 인스턴스화에 붙이는 `AutoFreeOnLaunch`는 cudaMallocAsync 백엔드에서 생기는 그래프 메모리 노드용이고(`CUDAGraph.cpp` 주석), 기본 할당기의 풀 공유와는 관계없습니다.

### 2.6 PyTorch 설정으로 끝나는 최적화를 먼저 확인하세요

| 하고 싶은 것 | 설정 |
|---|---|
| 큰 할당이 섞여 생기는 단편화 줄이기 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (VMM) |
| 드라이버 메모리 풀 쓰기 | `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` |
| pinned 할당의 driver 전역 lock 피하기 | `PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True` (`malloc` + `cudaHostRegister`) |
| 메모리 상한 근처에서 블록 분할 제한 | `max_split_size_mb`, `garbage_collection_threshold` |

---

## 3. 프로파일별 직접 호출 API 비교

같은 기능을 만들 때 프로파일에 따라 직접 부르는 API가 얼마나 달라지는지 예를 들면 다음과 같습니다.

| 기능 | [SA] 독립형 (llama.cpp, MLX, ExecuTorch) | [PT] PyTorch 기반 (vLLM, SGLang, TRT-LLM) |
|---|---|---|
| 디코딩 CUDA Graph | `cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaGraphInstantiate`, `cudaGraphExecUpdate`, `cudaGraphLaunch`, `cudaGraphExecDestroy`, `cudaGraphDestroy` | `torch.cuda.CUDAGraph` + (직접 만든 코드에서) `cudaStreamIsCapturing` |
| 요청별 할당 | `cudaMallocAsync` 풀 또는 VMM 풀을 직접 | 캐싱 할당기 (호출 없음) |
| 스트림·이벤트 | RAII 래퍼를 직접 | `torch.cuda.Stream`/`Event` (호출 없음) |
| KV 오프로드 | `cudaHostRegister`, `cudaMemcpyAsync`, 이벤트 + 배치 복사 | 배치 복사만 직접, pin은 `cudaHostRegister` 직접 또는 `cudart()` |
| Sleep/Wake | VMM 전체 | VMM 전체 + pluggable allocator 연결 |

**결론**: [PT] 프로파일은 T0와 CUDA Graph 기반을 PyTorch에서 얻습니다. 직접 다루는 CUDA API는 **PyTorch가 제공하지 않는 T1 최적화(PDL, 배치 복사), PyTorch 구현으로 부족할 때의 Custom AllReduce, 서빙 전용 T2~T3 기능**에 집중됩니다. vLLM의 Driver API 16개가 모두 VMM·배치 복사·IPC base 주소에 쓰이는 것이 그 예입니다.
