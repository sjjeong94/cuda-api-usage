# 03. KV Cache: 용량 산정, 계층화, 배치 복사

KV cache 관리의 핵심 로직(paging, prefix 재사용, eviction)은 CUDA API가 아니라 자료구조와 커널의 문제입니다. CUDA API가 필요해지는 지점은 세 곳입니다. **얼마나 잡을지**(용량), **어디에 잡을지**(디바이스 할당·VMM), **GPU 밖으로 어떻게 옮길지**(계층화·전송)입니다.

| 단계 | 기능 | 등급 | 핵심 API |
|---|---|:---:|---|
| 용량 | [메모리 용량 산정](01-t0-essential.md#t0-meminfo-메모리-용량-산정) | T0 | `cudaMemGetInfo` |
| 할당 | 고정 풀 / [VMM arena](02-memory.md#t2-vmm-pool-vmm-확장-풀arena) | T0 / T2 | `cudaMalloc` / `cuMemMap` |
| 확장 | [Off-graph KV 지연 확장](#off-graph-kv-지연-확장-t0-조합) | T0 조합 | `cudaMemcpyAsync`, 이벤트 |
| 계층화 | [Host 오프로드](#kv-host-오프로드-t0--t1) | T0 + T1 | `cudaHostRegister`, `cudaMemcpyAsync` |
| 계층화 | [배치 복사](#t1-batchcopy-배치-복사) | T1 | `cu(da)MemcpyBatchAsync` |
| 계층화 | [Zero-copy 전송 커널](#zero-copy--tma-전송-커널-t1--t2) | T1 / T2 | `cudaHostGetDevicePointer`, `cuTensorMapEncodeTiled` |
| 계층화 | [링크 인지 복사 전략](#t3-linkaware-링크-인지-복사-전략) | T3 | `cudaDeviceGetPCIBusId`, `cudaGetKernel` → `cuLaunchKernel` |

---

## KV host 오프로드 (T0 + T1)

- **목적**: GPU에서 밀려난 KV 블록을 host 메모리에 두었다가 prefix가 다시 맞으면 가져옵니다. GPU 용량을 넘는 prefix cache를 만들 수 있습니다.
- **필수 API**
  - host 풀: 큰 영역을 할당한 뒤 `cudaHostRegister`로 pin (청크 단위 등록이 안전함)
  - 전송: `cudaMemcpyAsync` (전용 스트림)
  - 순서: `cudaEventRecord` + `cudaStreamWaitEvent` (전송 완료 전에 계산이 읽지 않도록)
- **선택 API**: `cudaPointerGetAttributes`로 버퍼가 pinned인지 판별해 경로 선택 (vLLM cache_kernels, hisparse)
- **근거**
  - vLLM: `v1/kv_offload/cpu/*`, `v1/simple_kv_offload/cuda_mem_ops.py` (`cudaHostRegister`, 공유 메모리 영역 pin)
  - SGLang HiCache: `mem_cache/pool_host/common.py` (큰 풀은 나눠서 등록)
  - TRT-LLM KV v2: Driver의 `cuMemHostRegister` 사용
- **결론**: 블록 수가 적다면 T0 API만으로 됩니다. 블록이 작고 많아지면 **호출 횟수 자체가 병목**이 되므로 다음의 배치 복사가 필요합니다.

## T1-BATCHCOPY 배치 복사

- **목적**: 페이지 단위 KV는 블록 하나가 작고(수십~수백 KB) 흩어져 있습니다. 블록 수백 개를 `cudaMemcpyAsync`로 하나씩 제출하면 API 호출 오버헤드가 전송 시간을 넘습니다. 배치 복사는 여러 (src, dst, size)를 **한 번의 호출**로 copy engine에 제출합니다.
- **API**: `cuMemcpyBatchAsync` (Driver) 또는 `cudaMemcpyBatchAsync` (Runtime)
- **폴백**: `cudaMemcpyAsync` 반복 호출. 세 프레임워크 모두 폴백 경로를 둡니다.
- **요구 사항**: CUDA 12.8+. vLLM 문서에 따르면 legacy default stream에서는 쓸 수 없습니다.
- **버전 분기 방법** ([08](08-driver-vs-runtime.md) 참고)

  | 프레임워크 | 방법 |
  |---|---|
  | vLLM | `cuGetProcAddress("cuMemcpyBatchAsync", ..., 12080)`로 Driver 심볼 해석 (C++ `cache_kernels.cu`, Python `cuda_mem_ops.py`) |
  | SGLang | `dlsym`으로 Runtime 심볼 해석 + `cudaRuntimeGetVersion`/`cudaDriverGetVersion` 확인 (`kvcacheio/transfer.cu`), JIT 커널은 함수 포인터 (`staged_write_back.cuh`) |
  | TRT-LLM | KV v2 `batchedPageCopy.cu`, 12.8 미만은 `cuMemcpyAsync` 반복 |

- **다른 용도**: TRT-LLM은 DWDP 가중치 이동(`weight_manager.py`)과 확산 모델 Ulysses all-to-all(`asyncUlyssesOp.cpp`)에도 씁니다. **copy engine으로 전송하므로 SM을 쓰지 않습니다.**
- **결론**: KV 오프로드를 한다면 사실상 표준입니다. 빌드 시점에 링크하지 말고 **런타임에 심볼을 찾아** 구버전 드라이버에서도 동작하게 하세요.

## Zero-copy / TMA 전송 커널 (T1 / T2)

- **목적**: copy engine 대신 **커널이 직접** host 메모리를 읽고 써서 흩어진 블록을 한 번에 gather/scatter합니다. 레이아웃 변환을 같이 할 수 있습니다.
- **API**
  - `cudaHostGetDevicePointer`: pinned host 버퍼의 디바이스 주소 (SGLang `transfer.cu`, `runtime.cuh`)
  - `cuTensorMapEncodeTiled`: TMA로 전송하는 커널 (SGLang `hicache_tma.cuh`, Runtime에서 `cudaGetDriverEntryPointByVersion`으로 해석)
- **트레이드오프**: SM을 소모하므로 계산과 경쟁합니다. 대신 블록이 아주 작거나 레이아웃 변환이 필요하면 copy engine보다 유리할 수 있습니다.
- **결론**: 선택 사항입니다. copy engine 경로(배치 복사)를 먼저 만들고, 프로파일링으로 이득이 확인될 때 추가하세요.

## T3-LINKAWARE 링크 인지 복사 전략

- **목적**: host-GPU 링크 종류에 따라 빠른 복사 방법이 다릅니다. NVLink-C2C(Grace Hopper/Blackwell)는 대역폭이 높아 SM 기반 복사 커널이 유리하고, PCIe에서는 SM을 쓰지 않는 copy engine 배치 복사가 유리합니다.
- **API**
  - 링크 판별: `cudaDeviceGetPCIBusId` → NVML 장치와 매칭
  - SM 복사: `cudaGetKernel`로 런타임 커널 핸들을 얻어 `cuLaunchKernel`로 실행
  - copy engine: `cuMemcpyBatchAsync`
- **근거**: TRT-LLM KV Cache Manager v2 (`batchedPageCopy.cu`, `kvCacheManagerV2Utils.cu`)
- **결론**: Grace 계열을 주요 대상으로 한다면 고려할 만합니다. PCIe 시스템만 대상이라면 배치 복사로 충분합니다.

## Off-graph KV 지연 확장 (T0 조합)

- **목적**: 최대 컨텍스트 길이만큼 KV를 미리 잡지 않고, 필요할 때 키웁니다.
- **API**: 새 버퍼 할당 → `cudaMemcpyAsync`로 기존 내용 복사. 호출자가 다른 스트림을 썼을 수 있으므로 `cudaEventRecord` + `cudaStreamWaitEvent`로 순서를 맞춤
- **근거**: ExecuTorch `cuda_kv_cache.cpp` (export 때 `kvcache::update_and_attend`를 그래프 밖 연산으로 바꿈)
- **대안**: 복사 없이 포인터를 유지하며 키우려면 [VMM arena](02-memory.md#t2-vmm-pool-vmm-확장-풀arena)를 씁니다.
- **결론**: 단일 사용자나 엣지 환경처럼 메모리가 작은 곳에서 유용합니다. 다중 요청 서빙에서는 paged KV + 고정 풀이 일반적입니다.

---

## 프레임워크별 KV 설계와 필요 API

| 프레임워크 | KV 설계 | Driver 필요 여부 |
|---|---|---|
| vLLM | 고정 풀(PyTorch) + CPU 오프로드 + 배치 복사 | 배치 복사(`cuMemcpyBatchAsync`)에만 |
| SGLang | VMM arena + HiCache 계층화 + 배치 복사 + TMA 전송 | arena(VMM), TMA |
| TRT-LLM | KV Cache Manager v2: VMM GPU 계층 + host 계층 + 링크별 복사 | 모듈 전체를 Driver로 작성 |
| llama.cpp | KV는 일반 디바이스 버퍼, 연산 임시 버퍼는 VMM 풀 | KV 자체에는 불필요 (임시 버퍼 풀에 VMM) |
| ExecuTorch | off-graph 지연 확장 | 불필요 |

**결론**: KV를 GPU 안에서만 관리한다면 Runtime API로 충분합니다. **host 계층화**에는 T1 배치 복사(버전 분기 필요)가, **포인터를 유지한 확장·공유**에는 Driver VMM이 필요합니다.
