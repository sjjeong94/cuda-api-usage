# 02. 메모리: 할당기, VMM, Sleep/Wake, 공유

서빙 엔진의 메모리 설계는 세 가지를 정해야 합니다. (1) 요청마다 생기는 할당을 **동기화 없이** 처리하는 방법, (2) 메모리를 **해제했다가 다시 쓰는** 방법, (3) 메모리를 **다른 프로세스·GPU와 공유**하는 방법입니다. (1)은 Runtime API로 충분하고, (2)와 (3)부터는 Driver의 VMM API가 필요합니다.

| 기능 | 등급 | 핵심 API | 종류 |
|---|:---:|---|---|
| [스트림 순서 메모리 풀](#t1-pool-스트림-순서-메모리-풀) | T1 | `cudaMallocAsync`, `cudaMemPoolCreate` | Runtime |
| [Mapped host 메모리](#t1-zerocopy-mapped-host-메모리-zero-copy) | T1 | `cudaHostAlloc(Mapped)`, `cudaHostGetDevicePointer` | Runtime |
| [VMM 확장 풀·arena](#t2-vmm-pool-vmm-확장-풀arena) | T2 | `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap` | **Driver** |
| [Sleep / Wake](#t2-sleep-sleep--wake) | T2 | VMM + 백업 복사 | **Driver** |
| [VMM 공유 핸들](#t2-vmm-share-vmm-공유-핸들) | T2 | `cuMemExportToShareableHandle` | **Driver** |
| [Unified memory](#t2-uvm-unified-managed-memory) | T2 | `cudaMallocManaged`, `cudaMemAdvise` | Runtime |

---

## T1-POOL 스트림 순서 메모리 풀

- **목적**: `cudaMalloc`/`cudaFree`는 디바이스 동기화를 일으킵니다. 스트림 순서 할당은 free가 스트림 순서대로 처리되어 동기화가 필요 없고, 해제한 메모리를 풀에 남겨 다음 할당에 재사용합니다. CUDA Graph 캡처 안에서도 할당할 수 있습니다.
- **필수 API**: `cudaMallocAsync`, `cudaFreeAsync` (디바이스 기본 풀)
- **선택 API**
  - 전용 풀: `cudaMemPoolCreate`, `cudaMallocFromPoolAsync`, `cudaMemPoolDestroy` (ExecuTorch, TRT-LLM)
  - 풀 유지: `cudaMemPoolSetAttribute(cudaMemPoolAttrReleaseThreshold)`. 기본값(0)이면 동기화 시점에 메모리를 OS에 돌려주므로 재사용 효과가 줄어듭니다.
  - 캐시 반환: `cudaMemPoolTrimTo(pool, 0)` (메모리 압박 시, 모델 교체 시)
  - 사용량 조회: `cudaMemPoolGetAttribute` (MLX는 Windows WDDM 계측에도 사용)
- **폴백**: 풀을 지원하지 않는 디바이스는 `cudaMalloc`으로 (MLX). 전용 풀 생성이 실패하면 기본 풀로 (ExecuTorch).
- **요구 사항**: CUDA 11.2+, 디바이스 속성 `cudaDevAttrMemoryPoolsSupported`
- **프로파일**: [PT]는 PyTorch 캐싱 할당기가 같은 역할을 하므로 vLLM, SGLang은 직접 쓰지 않습니다. [SA]에서는 가장 간단한 고성능 할당기 선택지입니다.
- **근거**: MLX `allocator.cpp`, ExecuTorch `cuda_allocator.cpp`(`CudaAllocator`), TRT-LLM `runtime/cudaMemPool.cpp`, Thrust 임시 버퍼(ExecuTorch `sort.cu`)
- **결론**: [SA] 엔진이라면 **할당기의 첫 번째 선택지**입니다. 직접 캐싱 할당기를 만들 필요가 없습니다. 주소를 유지한 채 메모리를 반납해야 한다면 T2-VMM으로 가야 합니다.

## T1-ZEROCOPY Mapped host 메모리 (zero-copy)

- **목적**: host 메모리를 GPU 주소 공간에 매핑해, 복사 없이 커널이 PCIe/C2C 너머로 직접 읽게 합니다. 가중치나 KV를 GPU에 다 둘 수 없을 때 씁니다.
- **필수 API**: `cudaHostAlloc(cudaHostAllocMapped)`(또는 `cudaHostRegister(cudaHostRegisterMapped)`), `cudaHostGetDevicePointer`
- **근거**
  - vLLM UVA CPU 가중치 오프로드(`csrc/libtorch_stable/cuda_view.cu`, `get_cuda_view_from_cpu_tensor`)
  - SGLang HiCache zero-copy 전송 커널(`transfer.cu`, `runtime.cuh`)
  - llama.cpp PCIe AllReduce 스테이징(`allreduce.cu`)
  - TRT-LLM NIXL bounce 버퍼(`ExecPool.cpp`)
- **결론**: 커널이 host 데이터를 **한 번만 읽는** 경우(오프로드한 가중치 스트리밍, 흩어진 KV 블록 gather)에는 복사 후 실행보다 단순하고 빠를 수 있습니다. 여러 번 읽는 데이터에는 맞지 않습니다.

## T2-VMM-POOL VMM 확장 풀·arena

- **목적**: 가상 주소 공간을 크게 예약해 두고 물리 메모리를 필요한 만큼만 매핑합니다. 주소가 연속으로 유지되므로 풀을 키워도 단편화가 생기지 않고, 포인터가 바뀌지 않습니다.
- **필수 API (Driver)**

  | 단계 | API |
  |---|---|
  | 지원 확인 | `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED)` |
  | 준비 | `cuMemGetAllocationGranularity`, `cuMemAddressReserve` |
  | 확장 | `cuMemCreate` → `cuMemMap` → `cuMemSetAccess` (→ 필요하면 바로 `cuMemRelease`, 매핑은 유지됨) |
  | 해제 | `cuMemUnmap` → `cuMemRelease` → `cuMemAddressFree` |

- **선택 API**: `cuMemSetAccess`에 access descriptor를 여러 개 넘겨 여러 GPU에서 접근 (llama.cpp multi-GPU)
- **폴백**: VMM 미지원 GPU에서는 `cudaMalloc` 기반 legacy 풀 (llama.cpp, `GGML_CUDA_NO_VMM`)
- **요구 사항**: CUDA 10.2+, VMM 지원 디바이스. 할당 단위는 granularity(보통 2MB)의 배수
- **프로파일**: [PT]에서는 VMM 영역을 `torch.cuda.MemPool`로 노출해 PyTorch 텐서와 연결합니다 (SGLang `kv_vmm_backing.py`).
- **근거**: llama.cpp VMM 풀(기본 켜짐), SGLang KV cache VMM arena, TRT-LLM KV Cache Manager v2 GPU 계층
- **결론**: "**최대 크기를 모르는 버퍼를 포인터를 바꾸지 않고 키워야 할 때**" Driver API가 필요합니다. CUDA Graph가 캡처한 포인터를 그대로 유지해야 하는 엔진에서 특히 가치가 큽니다.

## T2-SLEEP Sleep / Wake

- **목적**: RLHF나 온라인 학습에서 학습과 추론이 GPU를 번갈아 쓸 때, 추론 엔진이 메모리를 **내려놓았다가 같은 주소로 다시 올립니다**. 가상 주소는 그대로 두고 물리 메모리만 해제·재생성하므로, CUDA Graph와 텐서 포인터를 다시 만들 필요가 없습니다.
- **필수 API**
  - VMM 해제·재생성: `cuMemUnmap`, `cuMemRelease` / `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`
  - 보존할 내용(가중치 등) 백업·복원: `cudaMemcpy` (vLLM) 또는 `cuMemcpyDtoH`, `cuMemcpyHtoD(Async)` (TRT-LLM)
- **선택 API**
  - 복원 후 초기화: `cuMemsetD8Async` (TRT-LLM)
  - 멀티캐스트 객체 재바인딩: `cuMulticastBindMem`, `cuMulticastUnbind` (TRT-LLM, [T3-NVLS](06-multi-gpu.md))
  - 컨텍스트 준비: `cuCtxGetCurrent`, `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent` (vLLM, 할당기가 컨텍스트 없는 스레드에서 불릴 수 있어서)
- **설계 포인트**: 할당마다 태그를 달아(가중치 / KV 등) 무엇을 백업하고 무엇을 버릴지 고릅니다 (TRT-LLM `sleep(sleep_tags)`). KV처럼 버려도 되는 메모리는 백업 없이 해제만 하면 됩니다.
- **프로파일**: [PT]에서는 PyTorch의 pluggable allocator로 VMM 할당기를 꽂습니다 (vLLM `CuMemAllocator`).
- **근거**: vLLM `csrc/cumem_allocator.cpp`, TRT-LLM `runtime/virtualMemory.{h,cpp}`(`CUDAVirtualMemoryChunk`)
- **결론**: RL 프레임워크와 연동할 서빙 엔진이라면 필요합니다. **Runtime API만으로는 만들 수 없습니다**(`cudaFree` 후 `cudaMalloc`은 주소를 보장하지 않음).

## T2-VMM-SHARE VMM 공유 핸들

- **목적**: 물리 메모리 핸들을 다른 프로세스나 노드로 내보내, 복사 없이 같은 메모리를 매핑합니다.
- **필수 API**: `cuMemCreate`(핸들 타입 지정) → `cuMemExportToShareableHandle` → (전달) → `cuMemImportFromShareableHandle` → `cuMemMap`, `cuMemSetAccess`
- **핸들 타입**
  - `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`: 같은 노드의 프로세스 간 (Unix socket으로 fd 전달)
  - `CU_MEM_HANDLE_TYPE_FABRIC`: 노드 간 (MNNVL, IMEX 데몬 필요, [T3-MNNVL](06-multi-gpu.md))
- **선택 API**
  - `cuMemRetainAllocationHandle`: 포인터가 VMM 할당인지 판별 (`cudaMalloc` 포인터에서는 실패)
  - `cuMemGetAllocationPropertiesFromHandle`: 핸들 속성 조회
  - `cuMemGetAddressRange`: 포인터가 속한 할당의 base와 크기 (전송 등록, graph 입력 매핑)
- **근거**
  - SGLang: 멀티모달 feature를 tokenizer 프로세스에서 scheduler로 무복사 전달(`--mm-feature-transport=cuda_vmm`), DWDP 전문가 가중치 공유, Custom AllReduce v2
  - TRT-LLM: NVLS·UserBuffers, MNNVL, DWDP
- **결론**: legacy IPC(`cudaIpc*`, [T1-IPC](06-multi-gpu.md))는 `cudaMalloc` 메모리를 같은 노드에서만 공유합니다. **VMM 메모리 공유, 노드 간 공유, 멀티캐스트**가 필요하면 이 Driver API가 필요합니다.

## T2-UVM Unified (managed) memory

- **목적**: CPU와 GPU가 같은 포인터로 접근하고, 드라이버가 페이지를 옮깁니다.
- **API**: `cudaMallocManaged`, `cudaMemAdvise`, (`cudaMemPrefetchAsync`, TRT-LLM 테스트에서만)
- **근거**
  - MLX: integrated GPU, concurrent managed access를 지원하는 디바이스에서 기본 할당 경로
  - llama.cpp: `GGML_CUDA_ENABLE_UNIFIED_MEMORY`로 opt-in (VRAM 초과 모델을 일단 돌리기 위해)
  - TRT-LLM EPLB: CPU가 GPU의 전문가 통계를 직접 읽는 메모리 (`hostAccessibleDeviceAllocator.cpp`)
- **결론**: discrete GPU 서빙의 핫 패스에서는 페이지 폴트 비용 때문에 피합니다. **integrated GPU(Jetson 등)**, 그리고 CPU와 GPU가 작은 상태를 공유하는 용도(EPLB 통계)에서만 씁니다.

---

## 선택 가이드

| 필요한 것 | 선택 |
|---|---|
| 요청별 임시 버퍼를 동기화 없이 | [PT] PyTorch 할당기 / [SA] `cudaMallocAsync` 풀 |
| 크기를 모르는 KV 영역을 포인터 변화 없이 키우기 | VMM arena (Driver) |
| 메모리를 반납했다가 같은 주소로 복원 (RL) | VMM Sleep/Wake (Driver) |
| 같은 노드의 다른 프로세스와 `cudaMalloc` 버퍼 공유 | `cudaIpc*` (Runtime) |
| VMM 버퍼 공유, 노드 간 공유 | 공유 핸들 (Driver) |
| host 데이터를 복사 없이 커널이 읽기 | mapped host (Runtime) |
