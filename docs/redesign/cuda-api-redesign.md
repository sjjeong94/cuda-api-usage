# CUDA Driver·Runtime API 재설계안

> 이 문서는 사고 실험입니다. "지금까지 정리한 서빙 프레임워크의 요구 사항(기능 41개)을 그대로 만족하는 API를 처음부터 설계한다면 어떻게 될까"를 다룹니다. NVIDIA에 보내는 제안서가 아니며, 실제로 쓸 수 있는 용도는 [10절](#10-서빙-프레임워크에-적용하기)의 **내부 추상화 계층**입니다.
>
> 근거 데이터: [`api_meta.csv`](../reference/data/api_meta.csv)(현재 API 266개), [`features.csv`](../reference/data/features.csv)(기능 41개), [`redesign/data/api_redesign_map.csv`](data/api_redesign_map.csv)(현재 API → 재설계 API 매핑, `python3 scripts/redesign/build_map.py`로 생성·검증)

---

## 0. 요약

| | 현재 | 재설계 |
|---|---:|---:|
| 호스트 API 수 (다섯 서빙 프레임워크 + PyTorch가 쓰는 것) | **266** (Driver 127, Runtime 137, 디바이스 측 2) | **56** |
| 계층 | Driver + Runtime 두 개 | 하나 |
| 암묵적 상태 | 현재 디바이스, context, last error, 전역 캡처 모드 | 없음 (모든 호출이 핸들을 명시) |

현재 API 266개는 다음과 같이 정리됩니다.
- **242개**: 재설계 함수 56개 중 하나로 합쳐집니다. 대부분 descriptor의 필드나 배열 항목이 됩니다.
- **17개**: 설계상 필요 없어집니다. context 9개, last error 2개, 캡처 모드 1개, peer access 활성화 1개 등입니다.
- **7개**: 코어 ABI 밖의 선택형 헬퍼로 옮깁니다. 현재 디바이스, 동기 대기, DOT 덤프 등입니다.

41개 기능 중 40개는 그대로 표현됩니다. 남은 1개(T2-CTX 컨텍스트 관리)는 context라는 개념이 사라지면서 **기능 자체가 필요 없어집니다**.

설계의 핵심은 아홉 가지 원칙입니다([3절](#3-설계-원칙)).
1. 계층은 하나만 두고, 모든 호출이 핸들을 명시합니다.
2. 함수 변형 대신 descriptor를 씁니다.
3. 복사·채우기는 배치가 기본입니다.
4. 큐에 넣는 작업은 모두 `Enqueue*`입니다.
5. 동기화는 timeline fence 하나로 합칩니다.
6. 메모리는 VMM 3층 구조를 기반으로 하고, 풀을 표준 할당기로 둡니다.
7. 커널은 출처와 상관없이 같은 핸들입니다.
8. 기능 확인은 capability와 버전 테이블로 합니다.
9. 객체 해제는 `cuxRelease` 하나입니다.

---

## 1. 전제와 범위

- **요구 사항**: [가이드](../guide/)의 기능 41개(T0~T3, AUX)가 요구 사항의 전부입니다. 지금 다섯 서빙 프레임워크와 PyTorch가 실제로 쓰는 기능만 다룹니다.
- **범위**: 호스트 API만 다룹니다. 디바이스 측 API(`cudaGridDependencySynchronize` 등)는 커널 언어의 일부이므로 그대로 둡니다. cuBLAS, NCCL 같은 라이브러리 API도 범위 밖입니다.
- **이름**: 기존 API와 섞이지 않도록 접두사 `cux`를 씁니다.
- **하지 않는 것**: 성능 모델을 바꾸지 않습니다. 모든 재설계 함수는 현재 CUDA 기능 위에 구현할 수 있어야 합니다([10절](#10-서빙-프레임워크에-적용하기)).

---

## 2. 진단: 왜 266개나 되는가

### 2.1 사용 분포

| 이 API를 본체 코드에서 쓰는 코드베이스 수 (PyTorch 포함 6개 중) | API 수 |
|---|---:|
| 0 (테스트·조건부에서만) | 12 |
| **1** | **113** |
| 2 | 42 |
| 3 | 39 |
| 4 | 27 |
| 5 | 20 |
| **6 (모두)** | **13** |

여섯 곳이 모두 쓰는 API는 13개뿐입니다: `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemsetAsync`, `cudaGetDevice`, `cudaSetDevice`, `cudaGetDeviceProperties`, `cudaFuncGetAttributes`, `cudaDeviceSynchronize`, `cudaGetLastError`, `cudaGetErrorString`, `cuGetErrorString`. 반대로 절반에 가까운 113개는 한 곳에서만 씁니다. 기능이 특수해서인 경우도 있지만, 대부분은 **같은 일을 하는 다른 이름**이기 때문입니다.

### 2.2 수가 늘어난 여섯 가지 원인

| 원인 | 규모 | 예 |
|---|---|---|
| **① 두 계층의 중복** | `alt` 역할(다른 계층에 같은 기능이 있음) 47개 | `cudaMalloc`/`cuMemAlloc`, `cudaEventRecord`/`cuEventRecord`, `cudaStreamCreate`/`cuStreamCreate`, `cudaMemsetAsync`/`cuMemsetD8Async`, `cudaGetErrorString`/`cuGetErrorString` |
| **② 함수 이름에 붙는 변형** | launch 6개, 복사 14개, 채우기 6개, 스트림 생성 4개, 이벤트 생성 3개 | `cudaLaunchKernel` / `Ex` / `ExC` / `Cooperative`, `cudaMemcpy` / `Async` / `2DAsync` / `PeerAsync` / `BatchAsync` / `ToSymbol`, `cuMemcpyHtoD` / `HtoDAsync` / `DtoH` / `DtoHAsync` / `DtoD`, `cuMemsetD8` / `D8Async` / `D32` / `D32Async`, `cudaStreamCreate` / `WithFlags` / `WithPriority` |
| **③ 객체·필드마다 getter/setter** | 그래프 관련 44개 중 노드 조회 18개 | `cudaGraphKernelNodeGetParams`, `cudaGraphHostNodeGetParams`, `cudaGraphMemAllocNodeGetParams`, `cudaGraphEventRecordNodeGetEvent`, `cuGraphMemcpyNodeGetParams` …. occupancy 7개, 디바이스 속성 8개 |
| **④ 같은 개념, 다른 메커니즘** | 동기화 3종, 공유 2종, 로딩 2종, 할당 8종 | 이벤트 / `cuStreamWaitValue32` / IPC 이벤트, `cudaIpc*` / VMM shareable handle, `cuModule*` / `cuLibrary*`, `cudaMalloc` / `MallocHost` / `HostAlloc` / `MallocManaged` / `MallocAsync` / `MallocFromPoolAsync` / `cuMemAlloc` / `cuMemHostAlloc` |
| **⑤ 암묵적 상태가 만든 API** | 17개 (재설계에서 제거) | 현재 디바이스(`cudaSetDevice`/`GetDevice`), context 9개(`cuCtxGetCurrent`, `cuDevicePrimaryCtxRetain` …), last error(`cudaGetLastError`, `cudaPeekAtLastError`), 전역 캡처 모드(`cudaThreadExchangeStreamCaptureMode`), peer access 활성화 |
| **⑥ 버전 호환을 위한 심볼 해석** | 해석 API 3개 + 버전 조회 3개 | `cuGetProcAddress`, `cudaGetDriverEntryPoint(ByVersion)`, `dlsym`, `_v2` 접미사. vLLM·SGLang·TRT-LLM·PyTorch가 모두 각자 해석 코드를 가짐 |

⑤는 서빙 엔진에서 특히 비용이 큽니다. vLLM의 sleep mode 할당기, PyTorch의 cuBLAS 호출 경로, TRT-LLM의 Logical Endpoint 코드가 모두 "**현재 스레드에 context가 없을 수 있다**"는 문제를 따로 처리합니다([08](../guide/08-driver-vs-runtime.md#3-t2-ctx-컨텍스트-관리-driverruntime-혼용)). vLLM과 PyTorch는 캡처 중 할당 때문에 캡처 모드를 바꾸는 코드도 각자 갖고 있습니다.

---

## 3. 설계 원칙

### P1. 계층은 하나, 모든 호출이 핸들을 명시
Driver/Runtime 구분을 없애고 C ABI 하나만 둡니다. "현재 디바이스", "현재 context", "프로세스 전역 last error"를 두지 않습니다. 모든 함수는 대상 객체(디바이스, 큐, 풀 …)를 인자로 받고 상태 코드를 반환합니다. → 원인 ①⑤ 해소

### P2. 함수 변형 대신 descriptor
옵션이 늘 때마다 `Ex`, `WithFlags`, `_v2`를 만들지 않습니다. 함수는 descriptor 구조체를 받고, 구조체 앞부분에 `size`(또는 type + next 체인) 헤더를 둬서 **필드를 추가해도 ABI가 유지**되게 합니다. 0으로 초기화한 descriptor가 기본 동작입니다. → 원인 ② 해소

### P3. 복사·채우기는 배치가 기본
`cuxEnqueueCopy(queue, items, n)`. 1D·2D·peer·host↔device·symbol·prefetch는 항목의 필드로 구분하고, 방향은 포인터로 판별합니다. 항목 하나짜리 호출이 지금의 `cudaMemcpyAsync`입니다. KV 블록 오프로드의 "블록 수백 개를 한 번에" 요구([T1-BATCHCOPY](../guide/03-kv-cache.md#t1-batchcopy-배치-복사))가 기본 경로가 됩니다.

### P4. 큐에 넣는 작업은 모두 `Enqueue*`
복사, 채우기, 커널 실행, 호스트 함수, fence signal·wait, 그래프 실행이 같은 형태의 함수입니다. 스트림 순서 할당은 풀 함수에 큐 인자로 표현합니다(`queue = NULL`이면 즉시 사용 가능). 그래프 캡처는 큐에 들어가는 이 명령들을 기록하는 것이므로, 캡처 가능한 명령의 범위가 명확해집니다.

### P5. 동기화는 timeline fence 하나
64-bit 값을 가진 fence 하나로 다음을 모두 표현합니다.
- 이벤트 기록(`cudaEventRecord`)과 스트림 간 대기(`cudaStreamWaitEvent`)
- GPU 메모리 값 쓰기·대기(`cuStreamWriteValue32` / `cuStreamWaitValue32`)
- 프로세스 간 이벤트(`cudaIpc*EventHandle`)
- 큐·디바이스 동기화(`cudaStreamSynchronize`, `cudaDeviceSynchronize`)

큐마다 내장 fence가 있어 "이 큐가 N번째 명령까지 끝났는가"를 조회할 수 있습니다. timing 옵션을 켠 fence는 signal 시각을 기록해 `cudaEventElapsedTime`을 대신합니다. → 원인 ④ 해소

### P6. 메모리는 VMM 3층 구조 + 표준 풀
- **주소 범위**(`cuxAddressReserve`), **물리 메모리**(`cuxMemCreate`: device / host / managed / 기존 host 메모리 import), **매핑**(`cuxMemMap`, 접근 권한 포함)이 기본 구성 요소입니다.
- 일반 할당은 그 위의 **풀**(`cuxPoolCreate`/`Alloc`/`Free`)입니다. 디바이스마다 기본 풀이 있고, 풀의 location으로 device / pinned host / mapped host / managed를 구분합니다. 지금의 할당 함수 8종이 `cuxPoolAlloc` 하나가 됩니다.
- 공유는 `cuxExport`/`cuxImport` 하나입니다. handle type(POSIX fd, FABRIC, legacy IPC)으로 구분하고, 메모리·fence·endpoint에 똑같이 씁니다.
- peer 접근은 매핑이나 풀의 access 목록으로 지정하므로 `cudaDeviceEnablePeerAccess`가 필요 없습니다.

Sleep/Wake, KV arena, 공유처럼 지금은 "고급 최적화(T2)"인 기능이 기본 구성 요소의 조합이 됩니다.

### P7. 커널은 출처와 상관없이 같은 핸들
nvcc로 함께 빌드한 커널(`cuxKernelFromSymbol`), 미리 빌드한 CUBIN이나 JIT 결과(`cuxLibraryLoad` → `cuxLibraryGetKernels`)가 모두 같은 `cuxKernel` 핸들이 됩니다. 실행은 `cuxEnqueueLaunch` 하나이고, 속성·이름·occupancy는 `cuxKernelQuery` 하나로 조회합니다. 지금처럼 `<<<>>>`·`cudaLaunchKernelEx`·`cuLaunchKernelEx`, `cudaFuncSetAttribute`·`cuFuncSetAttribute`·`cuKernelSetAttribute`로 갈리지 않습니다. module(context 종속)과 library(context 독립)의 구분도 context가 없으니 사라집니다.

### P8. 기능 확인은 capability와 버전 테이블
`cuxGetApi(requested_version, &table)`가 함수 포인터 테이블을 돌려줍니다. 지원 여부는 함수가 있는지가 아니라 `cuxDeviceQuery`의 capability 비트(VMM, multicast, fabric, RDMA, TMA, cluster, conditional graph, batch copy …)로 판단합니다. 프레임워크마다 따로 있는 `cuGetProcAddress`·`dlsym` 코드가 필요 없어집니다. PyTorch c10 Driver 테이블([08](../guide/08-driver-vs-runtime.md#2-t2-vergate-버전-의존-심볼-해석))이 하는 일을 API가 기본으로 제공하는 셈입니다.

### P9. 객체 해제는 `cuxRelease` 하나
`cudaStreamDestroy`, `cudaEventDestroy`, `cuMemRelease`, `cuMemAddressFree`, `cuModuleUnload`, `cudaGraphExecDestroy`, `cuGreenCtxDestroy` … 19개가 하나로 합쳐집니다.

---

## 4. 객체 모델

```
Device ─┬─ Partition (SM 분할; 기본 파티션 = 디바이스 전체)
        │     └─ Queue (우선순위, 내장 fence)
        ├─ Pool (location: device | host | mapped-host | managed; 기본 풀 1개)
        └─ (capability, 속성, peer 링크 정보)

Fence (64-bit timeline; timing·shareable 옵션)
AddressRange ── Memory (물리; device | host | managed | host-import) ── 매핑(access 목록)
Multicast ──── Memory 바인딩
Endpoint (Logical Endpoint)
Library ──── Kernel  ←── KernelFromSymbol (nvcc 빌드 커널)
Graph ──── GraphExec
```

모든 객체는 `cuxRelease`로 해제하고, 공유 가능한 객체(Memory, Fence, Endpoint)는 `cuxExport`/`cuxImport`로 주고받습니다.

---

## 5. API 전체 목록 (56개)

`→` 뒤는 합쳐지는 현재 API 수입니다. 전체 대응은 [매핑 CSV](data/api_redesign_map.csv)에 있습니다.

### 초기화·에러 (3)
| 함수 | 역할 | 대체 |
|---|---|---|
| `cuxGetApi(uint32 version, cuxApi* table)` | 버전 지정 함수 테이블. 초기화 포함 | → 7 (`cuInit`, `cuGetProcAddress`, `cudaGetDriverEntryPoint(ByVersion)`, 버전 조회 3) |
| `cuxGetErrorInfo(cuxStatus, cuxErrorInfo*)` | 이름·메시지 | → 4 |
| `cuxSetLogCallback(cb, user)` | 드라이버 로그 콜백 | → 1 |

### 공통 객체 (3)
| 함수 | 역할 | 대체 |
|---|---|---|
| `cuxRelease(cuxObject)` | 모든 객체 해제 | → 19 |
| `cuxExport(cuxObject, cuxHandleType, void* out)` | 메모리·fence·endpoint 공유 핸들 | → 4 |
| `cuxImport(const void* handle, cuxHandleType, cuxObject*)` | 가져오기 | → 4 |

### 디바이스 (4)
| 함수 | 역할 | 대체 |
|---|---|---|
| `cuxDeviceList(cuxDevice* out, uint32* n)` | 열거 | → 3 |
| `cuxDeviceQuery(cuxDevice, cuxDeviceInfo*)` | 정적 속성, capability, granularity, 기본 풀 | → 12 |
| `cuxDeviceMemInfo(cuxDevice, size_t* free, size_t* total)` | 용량 산정 | → 2 |
| `cuxDevicePeerQuery(cuxDevice a, cuxDevice b, cuxPeerInfo*)` | 접근 가능 여부, 링크 종류(NVLink / C2C / PCIe), atomics | → 1 |

### 파티션·큐 (3)
| 함수 | 역할 | 대체 |
|---|---|---|
| `cuxPartitionCreate(cuxDevice, const cuxPartitionDesc*, cuxPartition*)` | SM 수, work queue 설정 | → 6 (green context) |
| `cuxQueueCreate(const cuxQueueDesc*, cuxQueue*)` | partition, priority, flags | → 6 |
| `cuxQueueQuery(cuxQueue, cuxQueueInfo*)` | 내장 fence, 완료 값, 캡처 상태, 비동기 에러, 우선순위 | → 4 |

### Fence (4)
| 함수 | 역할 | 대체 |
|---|---|---|
| `cuxFenceCreate(const cuxFenceDesc*, cuxFence*)` | timing, shareable, 초기 값 | → 3 |
| `cuxFenceQuery(cuxFence, uint64* value, cuxFenceTime*)` | 현재 값, signal 시각 | → 4 |
| `cuxFenceWait(cuxFence, uint64 value, const cuxWaitDesc*)` | 호스트 대기 (spin/yield/block, timeout) | → 5 |
| `cuxFenceSignal(cuxFence, uint64 value)` | 호스트에서 signal | **신규** (호스트 → GPU 의존성) |

### 메모리 (6)
| 함수 | 역할 | 대체 |
|---|---|---|
| `cuxAddressReserve(const cuxAddressDesc*, cuxAddressRange*)` | 가상 주소 예약 | → 1 |
| `cuxMemCreate(const cuxMemDesc*, cuxMemory*)` | 물리 메모리 (device / host / managed / host-import, handle type, RDMA) | → 3 |
| `cuxMemMap(cuxAddressRange, size_t offset, cuxObject mem_or_multicast, size_t size, const cuxAccess* access, uint32 n)` | 매핑 + 접근 권한 | → 2 |
| `cuxMemUnmap(cuxAddressRange, size_t offset, size_t size)` | 매핑 해제 | → 1 |
| `cuxPointerQuery(const void* ptr, cuxPointerInfo*)` | base, size, location, 디바이스 측 주소, 메모리 핸들, VMM 여부 | → 7 |
| `cuxMemAdvise(const void* ptr, size_t, const cuxAdvice*)` | managed memory 힌트 | → 1 |

### 풀 (6)
| 함수 | 역할 | 대체 |
|---|---|---|
| `cuxPoolCreate(const cuxPoolDesc*, cuxPool*)` | location, release threshold, access, handle type | → 1 |
| `cuxPoolAlloc(cuxPool, size_t, cuxQueue /*NULL=즉시*/, void**)` | 모든 종류의 할당 | → 8 |
| `cuxPoolFree(void*, cuxQueue /*NULL=즉시*/)` | 해제 | → 5 |
| `cuxPoolQuery(cuxPool, cuxPoolStats*)` | 사용량 (그래프 풀 포함) | → 3 |
| `cuxPoolTrim(cuxPool, size_t keep)` | 캐시 반환 | → 2 |
| `cuxPoolUpdate(cuxPool, const cuxPoolDesc*)` | threshold, access 변경 | → 2 |

### 명령 (7)
| 함수 | 역할 | 대체 |
|---|---|---|
| `cuxEnqueueCopy(cuxQueue, const cuxCopyItem*, uint32 n)` | 1D/2D/peer/symbol/prefetch 배치 복사 | → 14 |
| `cuxEnqueueFill(cuxQueue, const cuxFillItem*, uint32 n)` | 1/2/4 B 원소 채우기 배치 | → 6 |
| `cuxEnqueueLaunch(cuxQueue, cuxKernel, const cuxLaunchDesc*, void** args)` | grid, block, smem, cluster, cooperative, PDL, priority | → 6 |
| `cuxEnqueueHostFn(cuxQueue, void (*fn)(void*), void*)` | 스트림 순서 호스트 콜백 | → 4 |
| `cuxEnqueueSignal(cuxQueue, cuxFence, uint64 value)` | fence signal (메모리 값 쓰기 포함) | → 4 |
| `cuxEnqueueWait(cuxQueue, cuxFence, uint64 value)` | fence 대기 | → 3 |
| `cuxEnqueueGraph(cuxQueue, cuxGraphExec)` | 그래프 실행 | → 1 |

### 커널 (7)
| 함수 | 역할 | 대체 |
|---|---|---|
| `cuxLibraryLoad(const cuxLibraryDesc*, cuxLibrary*)` | CUBIN/PTX/fatbin, 메모리 또는 파일, JIT 옵션 | → 5 |
| `cuxLibraryGetKernels(cuxLibrary, const char** names, uint32 n, cuxKernel*)` | 이름으로 또는 전체 나열 (`names = NULL`) | → 5 |
| `cuxKernelFromSymbol(const void* host_symbol, cuxKernel*)` | nvcc 빌드 커널을 같은 핸들로 | → 1 |
| `cuxKernelQuery(cuxKernel, const cuxLaunchDesc* /*nullable*/, cuxKernelInfo*)` | 속성, 이름, (launch desc가 있으면) occupancy | → 11 |
| `cuxKernelSetAttr(cuxKernel, cuxKernelAttr, int64)` | max dynamic smem, cache config | → 4 |
| `cuxGetGlobal(cuxLibrary /*NULL=정적*/, const void* name_or_symbol, void** ptr, size_t*)` | `__device__`/`__constant__` 주소 | → 3 |
| `cuxTensorMapEncode(const cuxTensorMapDesc*, cuxTensorMap*)` | TMA descriptor | → 1 |

### 그래프 (9)
| 함수 | 역할 | 대체 |
|---|---|---|
| `cuxCaptureBegin(cuxQueue, const cuxCaptureDesc*)` | 새 그래프 또는 조건 노드 본문으로 캡처 | → 2 |
| `cuxCaptureEnd(cuxQueue, cuxGraph*)` | 캡처 종료·복귀 | → 2 |
| `cuxGraphCreate(cuxGraph*)` | 빈 그래프 | → 1 |
| `cuxGraphAddNode(cuxGraph, const cuxNodeDesc*, const cuxNode* deps, uint32 n, cuxNode*)` | kernel / copy / fill / host / child / conditional / alloc / signal / wait | → 8 |
| `cuxGraphInspect(cuxGraph, cuxNode* nodes, uint32* n, cuxEdge* edges, uint32* m)` | 구조 조회 | → 7 |
| `cuxGraphNodeQuery(cuxNode, cuxNodeDesc*)` | 노드 descriptor, id | → 18 |
| `cuxGraphInstantiate(cuxGraph, const cuxInstantiateDesc*, cuxGraphExec*)` | auto-free, node priority | → 2 |
| `cuxGraphUpdate(cuxGraphExec, cuxGraph)` | 구조가 같은 그래프로 갱신 | → 3 |
| `cuxGraphRetain(cuxGraph, cuxObject or destructor)` | 리소스 수명을 그래프에 묶음 | → 2 |

### 패브릭 (4)
| 함수 | 역할 | 대체 |
|---|---|---|
| `cuxMulticastCreate(const cuxMulticastDesc*, cuxMulticast*)` | 참여 디바이스, 크기 | → 2 |
| `cuxMulticastBind(cuxMulticast, size_t offset, cuxMemory /*NULL=unbind*/)` | 물리 메모리 바인딩 | → 2 |
| `cuxEndpointCreate(const cuxEndpointDesc*, cuxEndpoint*)` | ID 예약, peer, 버퍼 바인딩 | → 3 |
| `cuxEndpointQuery(cuxEndpoint, cuxEndpointInfo*)` | 상태 | → 1 |

---

## 6. 핵심 descriptor 스케치

모든 descriptor는 `cuxHeader`로 시작합니다. 확장할 때는 `size`가 커지거나 `next`에 확장 구조체를 붙입니다.

```c
typedef struct { uint32_t size; uint32_t type; const void* next; } cuxHeader;

typedef struct {
  cuxHeader h;
  uint32_t grid[3], block[3];
  uint32_t dynamic_smem;
  uint32_t cluster[3];          // 0 = cluster 없음
  uint32_t flags;               // CUX_LAUNCH_COOPERATIVE | CUX_LAUNCH_PDL | ...
  int32_t  priority;
} cuxLaunchDesc;                // cudaLaunchKernel/Ex/ExC/Cooperative, cuLaunchKernel/Ex 대체

typedef struct {
  void* dst; const void* src;
  size_t width;                 // 1D면 bytes
  size_t height;                // 0 또는 1이면 1D
  size_t dst_pitch, src_pitch;
  uint32_t flags;               // CUX_COPY_PREFETCH (src = NULL, dst location만) ...
} cuxCopyItem;                  // cudaMemcpy*, cuMemcpy*, cudaMemcpyBatchAsync 대체

typedef struct {
  cuxHeader h;
  cuxLocation location;         // {DEVICE, id} | {HOST} | {MANAGED} | {HOST_IMPORT, ptr}
  size_t size;
  uint32_t handle_types;        // POSIX_FD | FABRIC | LEGACY_IPC
  uint32_t flags;               // RDMA_CAPABLE ...
} cuxMemDesc;                   // cuMemCreate, cudaHostRegister, cuMemHostRegister 대체

typedef struct {
  cuxHeader h;
  cuxLocation location;         // DEVICE | HOST_PINNED | HOST_MAPPED | MANAGED
  size_t release_threshold;
  const cuxAccess* access; uint32_t n_access;   // peer GPU 접근
  uint32_t handle_types;
} cuxPoolDesc;                  // cudaMemPoolCreate, MallocHost/HostAlloc/MallocManaged 경로 대체

typedef struct {
  cuxHeader h;
  cuxPartition partition;       // NULL = 디바이스 기본 파티션
  int32_t priority;
  uint32_t flags;               // NON_BLOCKING ...
} cuxQueueDesc;                 // cudaStreamCreate/WithFlags/WithPriority, cuGreenCtxStreamCreate 대체

typedef struct {
  cuxHeader h;
  uint32_t sm_count;            // 0 = 제한 없음
  uint32_t workqueue_scope;     // device | balanced
  uint32_t workqueue_limit;
} cuxPartitionDesc;             // cuDevSmResourceSplit*, cuDevResourceGenerateDesc, cuGreenCtxCreate 대체
```

---

## 7. 사용 예: 현재 API와 비교

### 7.1 KV 블록 host 오프로드 (블록 N개)

**현재** (vLLM·SGLang·TRT-LLM이 각자 구현하는 패턴)
```c
// 1) 배치 복사 심볼 해석 (CUDA 12.8+), 없으면 폴백
//    (인자는 CUDA 12.8 시그니처를 간략화한 것. CUDA 13에서 일부 인자가 바뀜)
if (cuGetProcAddress("cuMemcpyBatchAsync", &fn, 12080, 0, &st) == CUDA_SUCCESS && fn) {
  fn(dsts, srcs, sizes, N, attrs, attr_idx, n_attrs, &fail_idx, stream);
} else {
  for (int i = 0; i < N; i++) cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDeviceToHost, stream);
}
cudaEventRecord(done, stream);
cudaStreamWaitEvent(compute_stream, done, 0);
```

**재설계**
```c
cuxEnqueueCopy(xfer_q, items, N);                    // 항상 배치. 지원하지 않으면 구현이 내부에서 폴백
cuxEnqueueSignal(xfer_q, kv_fence, ++kv_value);
cuxEnqueueWait(compute_q, kv_fence, kv_value);
```

### 7.2 Sleep / Wake

**현재**: `cuCtxGetCurrent` → (없으면) `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent` → `cuMemGetAllocationGranularity` → `cuMemAddressReserve` → `cuMemCreate` → `cuMemMap` → `cuMemSetAccess`. sleep할 때는 `cudaMemcpy`(백업) → `cuMemUnmap` → `cuMemRelease`, wake할 때는 다시 `cuMemCreate` → `cuMemMap` → `cuMemSetAccess` → `cudaMemcpy`(복원). 그다음 PyTorch pluggable allocator로 연결합니다.

**재설계**
```c
cuxAddressReserve(&(cuxAddressDesc){.size = total}, &va);              // context 처리 없음
cuxMemCreate(&(cuxMemDesc){.location = dev0, .size = sz}, &mem);
cuxMemMap(va, 0, mem, sz, &(cuxAccess){dev0, CUX_RW}, 1);              // 매핑 + 권한
// sleep
cuxEnqueueCopy(q, &(cuxCopyItem){.dst = host_backup, .src = va_ptr, .width = sz}, 1);
cuxEnqueueSignal(q, f, ++v); cuxFenceWait(f, v, NULL);
cuxMemUnmap(va, 0, sz); cuxRelease(mem);
// wake: cuxMemCreate → cuxMemMap → cuxEnqueueCopy (복원)
```

### 7.3 디코딩 step: CUDA Graph + PDL + green context

**현재**: green context 7단계(`cuDeviceGetDevResource` → `cuDevSmResourceSplitByCount` → `cuDevResourceGenerateDesc` → `cuGreenCtxCreate` → `cuGreenCtxStreamCreate`, 구버전은 `cuCtxFromGreenCtx` + `cuCtxPushCurrent` + `cuStreamCreate` + `cuCtxPopCurrent`) → `cudaStreamBeginCapture` → `cudaLaunchKernelEx`(PDL 속성) 반복 → `cudaStreamEndCapture` → `cudaGraphInstantiateWithFlags` → `cudaGraphLaunch`

**재설계**
```c
cuxPartitionCreate(dev, &(cuxPartitionDesc){.sm_count = decode_sms}, &decode_part);
cuxQueueCreate(&(cuxQueueDesc){.partition = decode_part}, &decode_q);
cuxCaptureBegin(decode_q, NULL);
for (...) cuxEnqueueLaunch(decode_q, k[i], &(cuxLaunchDesc){..., .flags = CUX_LAUNCH_PDL}, args[i]);
cuxCaptureEnd(decode_q, &g);
cuxGraphInstantiate(g, &(cuxInstantiateDesc){.flags = CUX_AUTO_FREE}, &exec);
cuxEnqueueGraph(decode_q, exec);
```

---

## 8. 기능 커버리지 검증

기능 41개를 재설계 API로 표현할 수 있는지 확인했습니다. 각 기능에 태그된 현재 API를 매핑 CSV로 바꿔 모았습니다.

| 기능 | 현재 API 수 | 재설계 함수 |
|---|---:|---|
| T0-DEV 디바이스 열거·선택 | 5 | `cuxDeviceList` (+ 헬퍼: 현재 디바이스) |
| T0-CAP 하드웨어·버전 조회 | 5 | `cuxDeviceQuery`, `cuxGetApi` |
| T0-MEMINFO 메모리 용량 산정 | 2 | `cuxDeviceMemInfo` |
| T0-ALLOC 디바이스 메모리 | 10 | `cuxPoolAlloc`, `cuxPoolFree`, `cuxEnqueueFill` |
| T0-XFER Host↔Device 전송 | 21 | `cuxEnqueueCopy`, `cuxPoolAlloc`/`Free`, `cuxMemCreate`(host import), `cuxPointerQuery`, `cuxGetGlobal`, `cuxRelease` |
| T0-STREAM 스트림·이벤트 | 24 | `cuxQueueCreate`, `cuxQueueQuery`, `cuxFenceCreate`/`Query`/`Wait`, `cuxEnqueueSignal`/`Wait`, `cuxRelease` |
| T0-LAUNCH 커널 실행·설정 | 3 | `cuxEnqueueLaunch`, `cuxKernelQuery`, `cuxKernelSetAttr` |
| T0-ERR 에러 처리 | 7 | `cuxGetErrorInfo`, `cuxSetLogCallback` (last error 2개 제거) |
| T1-GRAPH CUDA Graph | 14 | `cuxCaptureBegin`/`End`, `cuxGraphInstantiate`, `cuxGraphUpdate`, `cuxEnqueueGraph`, `cuxGraphRetain`, `cuxPoolQuery`/`Trim`, `cuxRelease` |
| T1-CAPAWARE 캡처 인지 | 3 | `cuxQueueQuery` (캡처 모드 교환 제거) |
| T1-PDL | 5 | `cuxEnqueueLaunch`(flag) + 디바이스 측 API 유지 |
| T1-OCC Occupancy | 5 | `cuxKernelQuery` |
| T1-POOL 스트림 순서 풀 | 11 | `cuxPoolCreate`/`Alloc`/`Free`/`Query`/`Trim`/`Update`, `cuxDeviceQuery`, `cuxRelease` |
| T1-ZEROCOPY Mapped host | 3 | `cuxPoolAlloc`(mapped-host 풀), `cuxPointerQuery` |
| T1-BATCHCOPY 배치 복사 | 2 | `cuxEnqueueCopy` (기본 동작) |
| T1-IPC Custom AllReduce | 6 | `cuxExport`, `cuxImport`, `cuxPointerQuery`(base), `cuxRelease` |
| T1-P2P Peer access | 4 | `cuxDevicePeerQuery`, `cuxEnqueueCopy`, `cuxPoolUpdate` (peer 활성화 제거) |
| T2-VMM-POOL VMM 풀 | 9 | `cuxAddressReserve`, `cuxMemCreate`, `cuxMemMap`, `cuxMemUnmap`, `cuxDeviceQuery`, `cuxRelease` |
| T2-SLEEP Sleep/Wake | 8 | 위와 같음 |
| T2-VMM-SHARE 공유 핸들 | 13 | 위 + `cuxExport`, `cuxImport`, `cuxPointerQuery` |
| T2-UVM Unified memory | 3 | `cuxPoolAlloc`(managed 풀), `cuxMemAdvise`, `cuxEnqueueCopy`(prefetch) |
| T2-KLOAD CUBIN·JIT 로딩 | 26 | `cuxLibraryLoad`, `cuxLibraryGetKernels`, `cuxKernelFromSymbol`, `cuxKernelQuery`, `cuxKernelSetAttr`, `cuxGetGlobal`, `cuxEnqueueLaunch`, `cuxGraphAddNode`, `cuxGraphUpdate`, `cuxRelease` |
| T2-TMA | 3 | `cuxTensorMapEncode` (entry point 해석은 `cuxGetApi`로) |
| T2-CLUSTER | 5 | `cuxEnqueueLaunch`, `cuxKernelQuery` |
| T2-GRAPH-BUILD 그래프 조립·분석 | 33 | `cuxGraphCreate`, `cuxGraphAddNode`, `cuxGraphInspect`, `cuxGraphNodeQuery`, `cuxGraphUpdate` |
| T2-GRAPH-COND 조건 노드 | 4 | `cuxCaptureBegin`, `cuxCaptureEnd`, `cuxGraphAddNode` |
| T2-COOP Cooperative launch | 1 | `cuxEnqueueLaunch`(flag) |
| T2-HOSTFN 호스트 콜백 | 4 | `cuxEnqueueHostFn` |
| T2-GREEN SM 분할 | 10 | `cuxPartitionCreate`, `cuxQueueCreate`, `cuxRelease` |
| T2-PRIO 스트림 우선순위 | 3 | `cuxQueueCreate`, `cuxQueueQuery` |
| T2-STREAMMEM GPU 측 신호 | 2 | `cuxEnqueueSignal`, `cuxEnqueueWait` |
| T2-DRVDETECT 드라이버만으로 탐지 | 9 | `cuxGetApi`, `cuxDeviceList`, `cuxDeviceQuery` |
| T2-VERGATE 버전 의존 해석 | 6 | `cuxGetApi` |
| **T2-CTX 컨텍스트 관리** | 8 | **없음 — context 개념을 없애 기능 자체가 필요 없음** |
| T3-NVLS 멀티캐스트 | 16 | VMM 함수 + `cuxMulticastCreate`, `cuxMulticastBind`, `cuxExport`/`Import` |
| T3-MNNVL fabric 메모리 | 12 | VMM 함수 + `cuxExport`/`Import`(FABRIC), `cuxPointerQuery` |
| T3-LE Logical Endpoint | 9 | `cuxEndpointCreate`, `cuxEndpointQuery`, `cuxExport`/`Import`, `cuxRelease` |
| T3-LINKAWARE 링크 인지 복사 | 2 | `cuxDeviceQuery`/`cuxDevicePeerQuery`(링크 종류), `cuxKernelFromSymbol` |
| T3-PCIEAR PCIe AllReduce | 2 | `cuxPoolAlloc`(mapped-host 풀), `cuxPointerQuery` |
| AUX-PROF 프로파일링 | 5 | `cuxFenceQuery`(timing), `cuxGraphNodeQuery`(tools id). 프로파일러 시작·종료는 도구 API로 |
| AUX-DEBUG 디버깅 | 2 | `cuxKernelQuery`(이름) + 헬퍼: 그래프 DOT |

---

## 9. 없앤 것과 대가

### 9.1 설계상 제거한 현재 API (17개)

| 현재 API | 제거 이유 |
|---|---|
| `cuCtxGetCurrent`, `cuCtxSetCurrent`, `cuCtxPushCurrent`, `cuCtxPopCurrent`, `cuCtxGetDevice`, `cuCtxCreate`, `cuCtxGetId`, `cuDevicePrimaryCtxRetain`, `cuDevicePrimaryCtxGetState`, `cuCtxFromGreenCtx` | 사용자에게 보이는 context가 없음. 객체는 디바이스·파티션에 속함 |
| `cudaGetLastError`, `cudaPeekAtLastError` | 모든 호출이 상태를 반환. 비동기 에러는 큐별로 `cuxQueueQuery` |
| `cudaThreadExchangeStreamCaptureMode` | 캡처는 캡처 중인 큐에만 영향. 풀 할당은 캡처 안전 |
| `cudaDeviceEnablePeerAccess` | 접근 권한은 매핑·풀의 access 목록 |
| `cudaDeviceReset` | 객체 단위 해제로 충분 |
| `cudaProfilerStart`, `cudaProfilerStop` | 도구 제어는 런타임 ABI가 아니라 도구 API(CUPTI·NVTX) 몫 |

### 9.2 대가와 위험

| 대가 | 설명 | 완화 |
|---|---|---|
| **코드가 길어짐** | `cudaMalloc(&p, n)` 한 줄이 `cuxPoolAlloc(default_pool, n, NULL, &p)`가 되고, launch마다 descriptor를 채워야 함 | 헬퍼 계층(9.3). 0 초기화 descriptor가 기본값 |
| **생태계 호환** | PyTorch, NCCL, cuBLAS는 primary context와 현재 디바이스를 가정함 | 구현은 내부적으로 디바이스마다 primary context 하나를 쓰고, 현재 context를 맞춰 줌 (지금 PyTorch가 하는 일과 같음) |
| **캡처 안전성 검사 약화** | 전역 캡처 모드는 다른 스레드의 위험한 호출을 잡아 줬음 | 큐 단위 캡처 + 캡처 중 금지 명령은 해당 큐에서 에러. 다른 스레드의 일반 할당은 풀이 처리 |
| **작은 할당 오버헤드** | VMM 3층 구조를 기본으로 하면 granularity(보통 2MB)가 큼 | 일반 할당은 풀이 큰 덩어리를 잘라 씀. 지금의 캐싱 할당기와 같은 구조 |
| **fence 의미 차이** | 이벤트는 "마지막 기록 시점", fence는 "단조 증가 값". 이벤트를 재기록하는 코드는 값 관리로 바꿔야 함 | 큐 내장 fence로 대부분의 경우를 덮음 |
| **descriptor ABI 복잡도** | `size`·`next` 체인 관리, 검증 비용 | Vulkan 등에서 검증된 방식. 핫 패스 함수(`EnqueueLaunch`, `EnqueueCopy`)는 descriptor를 재사용 |
| **배치 강제** | 항목 하나짜리 복사도 배열로 넘김 | 비용은 포인터 하나. 헬퍼가 단일 복사 함수를 제공 |

### 9.3 헬퍼 계층 (코어 ABI 아님, 7개 대체)

| 헬퍼 | 대체하는 현재 API | 내용 |
|---|---|---|
| 현재 디바이스 | `cudaGetDevice`, `cudaSetDevice` | 스레드 로컬 기본 디바이스·큐. 단일 GPU 코드를 짧게 |
| 디바이스 동기화 | `cudaDeviceSynchronize`, `cuCtxSynchronize` | 디바이스의 모든 큐 fence에 대기 |
| 그래프 DOT | `cudaGraphDebugDotPrint` | `cuxGraphInspect` 결과로 생성 |
| 디바이스 측 API | `cudaGridDependencySynchronize`, `cudaTriggerProgrammaticLaunchCompletion` | 커널 언어 쪽이라 그대로 유지 (헬퍼로 분류만 함) |

그 밖에 `CUX_CHECK` 매크로, 단일 항목 `cuxCopy`/`cuxFill`, `<<<>>>` 대체 템플릿 같은 편의 기능도 이 계층에 둡니다.

---

## 10. 서빙 프레임워크에 적용하기

CUDA 자체를 바꿀 수는 없지만, **서빙 프레임워크 안에 이 형태의 내부 추상화 계층을 두는 것**은 지금 바로 할 수 있습니다. 엔진 코드는 `cux*`만 부르고, 계층이 현재 CUDA API로 구현합니다. [매핑 CSV](data/api_redesign_map.csv)를 거꾸로 읽으면 그대로 구현 표가 됩니다.

| 재설계 함수 | 현재 CUDA 위의 구현 |
|---|---|
| `cuxGetApi` | 시작할 때 PyTorch c10 테이블처럼 `cudaGetDriverEntryPointByVersion`으로 Driver 심볼을 한 번에 해석. capability 비트는 `cuDeviceGetAttribute`·심볼 유무로 채움 |
| `cuxEnqueueCopy` | 12.8+면 `cuMemcpyBatchAsync`, 아니면 항목별 `cudaMemcpyAsync`/`cudaMemcpy2DAsync`/`cudaMemcpyPeerAsync` |
| `cuxEnqueueLaunch` | `cudaLaunchKernelEx`(정적 커널) 또는 `cuLaunchKernelEx`(library 커널). PDL·cluster·cooperative를 launch attribute로 |
| Fence | 값마다 이벤트 하나를 기록하는 링 버퍼 + (shareable이면) IPC 이벤트, 또는 VMM 메모리 + `cuStreamWriteValue32`/`WaitValue32` |
| `cuxPoolAlloc` | [PT]면 PyTorch 할당기나 `MemPool`, [SA]면 `cudaMallocAsync` 풀. host·managed는 `cudaHostAlloc`·`cudaMallocManaged` |
| `cuxPartitionCreate` | green context(`cuGreenCtxCreate`), 미지원이면 기본 파티션으로 폴백 |
| context 처리 | 계층 안에서 `cuCtxGetCurrent` → `cuDevicePrimaryCtxRetain` → `cuCtxSetCurrent`를 한 곳에서 처리 |
| `cuxCaptureBegin`/`End` | `cudaStreamBeginCapture`(Relaxed 또는 ThreadLocal 모드)/`EndCapture`, 조건 노드는 `cudaStreamBeginCaptureToGraph` |

**이 계층을 두면 얻는 것**
- 엔진 코드에서 버전 분기, context 처리, 폴백이 사라집니다. 지금은 vLLM, SGLang, TRT-LLM, PyTorch가 같은 처리를 각자 반복합니다.
- [SA]와 [PT] 프로파일 사이에서 할당기·그래프 백엔드를 바꿔 끼울 수 있습니다.
- HIP처럼 다른 백엔드로 옮길 때 바꿀 곳이 계층 하나로 줄어듭니다. llama.cpp의 `vendors/hip.h`가 같은 효과를 이름 매핑으로 얻고 있습니다.

**구현 순서 제안**: T0 대체(디바이스, 풀, `EnqueueCopy`/`Fill`/`Launch`, 큐, fence) → 캡처·그래프 → VMM·공유 → 파티션 → 패브릭

---

## 11. 데이터와 재현

| 파일 | 내용 |
|---|---|
| [`data/api_redesign_map.csv`](data/api_redesign_map.csv) | 현재 API 266개 → `new_api`, `group`, `change`(merged / removed / helper), `reason` |
| [`scripts/redesign/build_map.py`](../../scripts/redesign/build_map.py) | 매핑 정의와 재설계 API 56개 목록. 실행하면 `api_meta.csv`의 모든 API가 매핑됐는지 검사하고 CSV를 생성 |

새 API가 카탈로그에 추가되면 `build_map.py`가 "unmapped APIs"로 실패하므로, 재설계안이 데이터와 어긋나지 않게 유지됩니다.
