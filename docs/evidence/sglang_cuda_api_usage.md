# SGLang CUDA API 사용 현황 및 최적화 기법 매핑

> 기준: SGLang `main` 커밋 `1291f31` (2026-10-06)
>
> 방법: 저장소 전체(`3rdparty/`, `docs/` 제외)에서 `cuXxx(` / `cudaXxx(` 호출을 grep하고 주석·타입·enum·로그 문자열은 제외. `dlsym`, `cuGetProcAddress`, `cudaGetDriverEntryPointByVersion`으로 이름을 찾아 호출하는 API는 별도로 확인해 포함
> 범위: SGLang 저장소가 **직접** 호출하는 API만 포함. PyTorch, FlashInfer, sgl-attn(FlashAttention 포크), CUTLASS, DeepGEMM, Triton, NCCL, DeepEP 등 외부 의존성 내부 호출은 제외

---

## 목차

0. [SGLang의 CUDA 코드 구성](#0-sglang의-cuda-코드-구성)
1. [CUDA Driver API](#1-cuda-driver-api)
2. [CUDA Runtime API](#2-cuda-runtime-api)
3. [최적화 기법별 API 매핑](#3-최적화-기법별-api-매핑)
4. [요약](#4-요약)

---

## 0. SGLang의 CUDA 코드 구성

예전의 별도 `sgl-kernel` 패키지는 `python/sglang/kernels/` 아래로 통합되었습니다.

| 구성 요소 | 위치 | 형태 | 역할 |
|---|---|---|---|
| **SRT 런타임** | `python/sglang/srt/` | Python (`cuda-python` 바인딩, ctypes) | 스케줄러, 메모리 풀, CUDA Graph, 통신, HiCache |
| **AOT 커널** | `python/sglang/kernels/aot/csrc/` | C++/CUDA, wheel로 미리 빌드 | Custom AllReduce, KV 전송, MoE, 양자화 GEMM, green context |
| **JIT 커널** | `python/sglang/kernels/jit/csrc/` | C++/CUDA, 실행 시 nvcc로 빌드 (`tvm_ffi`) | 모델별 fused 커널, Marlin, KV write-back, IPC 통신 |
| **Python 커널 ops** | `python/sglang/kernels/ops/` | CuTe DSL, Triton, 소스 overlay | CuTe DSL 커널 로딩, TRT-LLM fused MoE overlay |
| **멀티모달 생성** | `python/sglang/multimodal_gen/` | Python | 확산 모델 서빙 (layerwise offload, IPC) |

**참고 사항**
- `kernels/ops/lora/moe/trtllm_lora_temp/`는 FlashInfer의 TRT-LLM fused MoE 소스 중 SGLang이 수정한 파일만 둔 **overlay**입니다(NVIDIA 저작권 표기). 저장소 안에 있으므로 포함하되, 아래에 따로 표시했습니다.
- AOT 빌드가 받는 외부 소스(`kernels/aot/CMakeLists.txt`): CUTLASS, fmt, Triton v3.8.0, FlashInfer, sgl-attn

---

## 1. CUDA Driver API

본체 코드에서 **46개**를 사용합니다 (테스트에서만 쓰는 2개 별도).

### 1.1 CUDA VMM (가상 메모리) 공통 유틸: `srt/utils/cuda_vmm_utils.py`

`cuda.bindings.driver`(cuda-python)로 호출합니다. KV cache, DWDP, 멀티모달 feature 전송, Custom AllReduce v2가 공통으로 씁니다.

| API | 용도 |
|---|---|
| `cuMemAddressReserve` / `cuMemAddressFree` | 가상 주소 공간 예약 / 해제 (`VmmReservation`) |
| `cuMemCreate` / `cuMemRelease` | 물리 메모리 핸들 생성 / 해제 |
| `cuMemMap` / `cuMemUnmap` | 예약 영역의 offset에 물리 메모리 매핑 / 해제 |
| `cuMemSetAccess` | 매핑 영역 RW 권한 부여 |
| `cuMemGetAllocationGranularity` | 할당 단위 조회 |
| `cuMemExportToShareableHandle` / `cuMemImportFromShareableHandle` | 프로세스 간 공유 핸들 (FABRIC 또는 POSIX fd) 내보내기 / 가져오기 |
| `cuMemRetainAllocationHandle` | 포인터가 VMM 할당인지 판별 (`is_vmm_pointer`, `cudaMalloc` 포인터에서는 실패) |
| `cuMemGetAllocationPropertiesFromHandle` | 핸들의 할당 속성 조회 |
| `cuMemGetAddressRange` | 포인터가 속한 할당의 base·크기 조회 (graph capture 입력을 base 기준으로 매핑) |

### 1.2 Green Context (SM 분할): `kernels/aot/csrc/spatial/greenctx_stream.cu`

| API | 용도 |
|---|---|
| `cuDriverGetVersion` | green context 지원 드라이버인지 확인 |
| `cuGetProcAddress` | `cuGreenCtxStreamCreate` 심볼을 런타임에 해석 |
| `cuDeviceGetDevResource` | 디바이스의 SM 리소스 조회 |
| `cuDevSmResourceSplitByCount` | SM을 지정한 개수로 분할 |
| `cuDevResourceGenerateDesc` | 분할한 리소스로 descriptor 생성 |
| `cuGreenCtxCreate` / `cuGreenCtxDestroy` | green context 생성 / 해제 |
| `cuGreenCtxGetDevResource` | green context의 리소스 조회 |
| `cuGreenCtxStreamCreate` | green context에 묶인 스트림 생성 |
| `cuCtxFromGreenCtx` / `cuCtxPushCurrent` / `cuCtxPopCurrent` / `cuStreamCreate` | 구버전 드라이버 폴백 경로 (컨텍스트를 바꿔 스트림 생성) |

### 1.3 CUDA Graph 분석: `srt/model_executor/runner_backend/cuda_graph_dedup_mixin.py`

| API | 용도 |
|---|---|
| `cuGraphGetNodes` / `cuGraphGetEdges` / `cuGraphNodeGetType` | 캡처한 그래프의 구조 추출 |
| `cuGraphKernelNodeGetParams` / `cuGraphKernelNodeGetAttribute` | 커널 노드의 파라미터·속성 비교 |
| `cuGraphMemcpyNodeGetParams` / `cuGraphMemsetNodeGetParams` | memcpy·memset 노드 비교 |
| `cuGraphChildGraphNodeGetGraph` | 하위 그래프 재귀 탐색 |

### 1.4 그 밖의 Driver API

| API | 위치 | 용도 |
|---|---|---|
| `cuStreamWaitValue32` / `cuStreamWriteValue32` | `srt/multimodal/transport/memory_pool.py` | GPU 메모리의 플래그로 스트림 순서 동기화 (프로세스 간) |
| `cuMemcpyDtoD` | `srt/layers/moe/dwdp/transport.py` | 전문가 가중치를 VMM 예약 영역으로 복사 |
| `cuMulticastGetGranularity` | `srt/layers/flashinfer_comm_fusion.py` | FlashInfer AllReduce fusion workspace 할당 전 multicast 지원 사전 점검 (`cuMemCreate` 시퀀스 probe 포함) |
| `cuPointerGetAttribute` | `kernels/aot/csrc/allreduce/custom_all_reduce.cuh` | IPC 버퍼의 base 주소 조회 |
| `cuTensorMapEncodeTiled` | `kernels/jit/csrc/attention/kda_prefill.cu`, `deepseek_v4/wo_a_fused.cuh`, `kvcacheio/hicache_tma.cuh` | TMA descriptor 생성 (hicache는 `cudaGetDriverEntryPointByVersion`으로 해석) |
| `cuLibraryLoadData` / `cuLibraryEnumerateKernels` / `cuKernelGetFunction` / `cuFuncGetAttribute` | `kernels/ops/attention/flash_attn/cute/cute_dsl_utils.py` | CuTe DSL로 컴파일한 CUBIN 로드, 커널 핸들·속성 조회 |
| `cuGetErrorString` / `cuGetErrorName` | `aot/csrc/spatial/cuda_utils.h`, `jit/csrc/distributed/ipc.cuh` 등 | 에러 메시지 |

### 1.5 테스트에서만 사용

| API | 위치 |
|---|---|
| `cuMemcpyDtoH`, `cuMemsetD8` | `test/registered/unit/test_cuda_vmm_utils.py` |

---

## 2. CUDA Runtime API

본체 코드에서 **호스트 API 44개와 디바이스 측 API 2개**를 사용합니다.

### 2.1 AOT 커널 (`kernels/aot/csrc/`)

| 분류 | API | 주요 위치 |
|---|---|---|
| 디바이스 / 속성 | `cudaGetDevice`, `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cudaDeviceGetAttribute` | `include/utils.h`, cutlass_extensions, infllm_v2, w4a8 MoE |
| 버전 확인 | `cudaRuntimeGetVersion`, `cudaDriverGetVersion` | `kvcacheio/transfer.cu` (배치 복사 경로 선택) |
| 커널 설정 | `cudaFuncSetAttribute` | topk, deepseek_v4_topk, causal_conv1d(Mamba), speculative_sampling, infllm_v2 flash_attn |
| | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | cutlass_extensions, sm100 mxfp8 group quant, infllm_v2 |
| 커널 실행 | `cudaLaunchKernel` | `speculative/speculative_sampling.cuh` |
| | `cudaLaunchKernelEx` | `gemm/per_token_group_quant_8bit_v2.cu` (PDL) |
| IPC (Custom AllReduce) | `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle`, `cudaIpcCloseMemHandle`, `cudaMemcpy` | `allreduce/custom_all_reduce.cuh` |
| CUDA Graph 대응 | `cudaStreamIsCapturing` | `allreduce/custom_all_reduce.cuh` |
| KV 전송 (HiCache) | `cudaMemcpyBatchAsync` (`dlsym`), `cudaMemcpyAsync` (폴백), `cudaHostGetDevicePointer` | `kvcacheio/transfer.cu` |
| 메모리 | `cudaMemsetAsync` | cutlass_extensions |
| 에러 | `cudaGetLastError`, `cudaGetErrorString` | 다수 |
| 디바이스 측 (PDL) | `cudaGridDependencySynchronize`, `cudaTriggerProgrammaticLaunchCompletion` | `per_token_group_quant_8bit_v2.cu`, `custom_all_reduce.cuh` |

### 2.2 JIT 커널 (`kernels/jit/csrc/`)

| 분류 | API | 주요 위치 |
|---|---|---|
| 디바이스 / 속성 | `cudaGetDevice`, `cudaSetDevice`, `cudaGetDeviceProperties`, `cudaDeviceGetAttribute` | `include/sgl_kernel/runtime.cuh`, `utils.cuh`, Marlin, inkling all-reduce |
| 버전 확인 | `cudaRuntimeGetVersion`, `cudaDriverGetVersion` | `runtime.cuh`, `kvcacheio/staged_write_back.cuh` |
| 드라이버 심볼 해석 | `cudaGetDriverEntryPointByVersion` | `kvcacheio/hicache_tma.cuh` (`cuTensorMapEncodeTiled`) |
| 커널 설정 | `cudaFuncSetAttribute` (34곳) | kda_fused_decode, deep_select topk, deepseek_v4 topk 등 |
| | `cudaFuncGetAttributes` | `gemm/marlin_moe/moe_wna16_marlin.cuh` |
| Occupancy | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | `runtime.cuh`, `utils.cuh`, kda_prefill, sm100 mxfp8 |
| | `cudaOccupancyAvailableDynamicSMemPerBlock` | `runtime.cuh` |
| | `cudaOccupancyMaxActiveClusters` | `occupancy/cluster_probe.cuh` |
| 커널 실행 | `cudaLaunchKernelEx` | `utils.cuh`, `moe/moe_finalize_fuse_shared.cu` (PDL) |
| IPC | `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle`, `cudaIpcCloseMemHandle` | `distributed/ipc.cuh` |
| 메모리 | `cudaMemcpyAsync`, `cudaMemcpy`, `cudaMemsetAsync`, `cudaHostGetDevicePointer` | deepseek_v4 c128/c_plan, custom_all_reduce, staged_write_back, kda_prefill, `runtime.cuh` |
| KV write-back | `cudaMemcpyBatchAsync` (함수 포인터) | `kvcacheio/staged_write_back.cuh` |
| 에러 | `cudaGetLastError`, `cudaGetErrorString` | 다수 |
| 디바이스 측 (PDL) | `cudaGridDependencySynchronize`, `cudaTriggerProgrammaticLaunchCompletion` | `gemm/dsv3_fused_a_gemm.cuh`, `moe/moe_finalize_fuse_shared.cu` |

### 2.3 Python 커널 ops (`kernels/ops/`)

| API | 위치 | 비고 |
|---|---|---|
| `cudaLibraryLoadData` | `flash_attn/cute/cute_dsl_ptxas.py` | CuTe DSL CUBIN 로드 |
| `cudaFuncSetAttribute`, `cudaLaunchKernelEx`, `cudaDeviceGetAttribute`, `cudaGetDevice`, `cudaEventRecord`, `cudaStreamWaitEvent`, `cudaGetLastError`, `cudaGetErrorString`, PDL 디바이스 API | `lora/moe/trtllm_lora_temp/data/` | **FlashInfer TRT-LLM overlay** |
| `cudaSetDevice` | `diffusion/ext/hunyuan3d_rasterizer/rasterizer_gpu.cu` | 3D rasterizer 확장 |

### 2.4 SRT 런타임 (`srt/`)

**ctypes 래퍼.** `srt/distributed/device_communicators/cuda_wrapper.py`가 vLLM과 같은 구조로 `libcudart`를 로드합니다.
`cudaSetDevice`, `cudaDeviceSynchronize`, `cudaDeviceReset`, `cudaGetErrorString`, `cudaMalloc`, `cudaFree`, `cudaMemset`, `cudaMemcpy`, `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle`

| 사용처 | API |
|---|---|
| `custom_all_reduce.py`, `custom_all_reduce_utils.py` (P2P 테스트, 버퍼 공유) | 위 래퍼 전부 |
| `disaggregation/common/staging_handler.py` (PD 분리 KV 스테이징) | `cudaMalloc` |
| `mem_cache/pool_host/common.py` (HiCache host KV pool) | `cudaHostRegister`, `cudaHostUnregister` (큰 풀은 나눠서 등록) |
| `utils/host_shared_memory.py`, `layers/engram.py` | `cudaHostRegister` |
| `cuda_graph_dedup_mixin.py` (cuda-python runtime) | `cudaGraphInstantiateWithFlags`, `cudaGraphExecUpdate`, `cudaGraphLaunch`, `cudaGraphExecDestroy` |
| `breakable_cuda_graph/breakable_cuda_graph.py` | `cudaStreamGetCaptureInfo` |
| `models/qwen4_exp_ple_table.py` | `cudaDeviceGetAttribute` |
| `profiler_manager.py`, `utils/profile_utils.py` | `cudaProfilerStart`, `cudaProfilerStop` |

### 2.5 멀티모달 생성 (`multimodal_gen/`)

| API | 위치 |
|---|---|
| `cudaHostRegister`, `cudaHostUnregister` | `memory_managers/layerwise_offload.py`, `entrypoints/utils.py` |
| `cudaDeviceEnablePeerAccess` | `device_communicators/ipc_a2a.py` (ctypes) |

### 2.6 제외한 항목
- `cudaMallocAsync`(`multimodal_gen/runtime/distributed/ipc_cuda.py`): 에러 메시지 문자열에만 나옵니다.
- `cudaEvent`(`moe_overlap.py`), `cuInit`, `cuModuleLoadData`: 주석에만 나옵니다.
- 테스트·벤치마크에서만 쓰는 것: `cudaMalloc`, `cudaFree`, `cudaFuncGetAttributes`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 등 위에 나온 API의 재사용뿐이고, 새로 추가되는 API는 없습니다.

---

## 3. 최적화 기법별 API 매핑

표기: **[S]** SRT 런타임, **[A]** AOT 커널, **[J]** JIT 커널, **[K]** Python 커널 ops, **[M]** 멀티모달 생성

### 3.1 CUDA VMM 기반 메모리 아키텍처 [S]

SGLang은 VMM을 여러 기능의 공통 기반으로 씁니다. 가상 주소를 크게 예약해 두고, 물리 메모리는 필요할 때 매핑하며, 공유 핸들로 다른 프로세스·GPU와 주고받습니다.

| 기능 | 하는 일 | API |
|---|---|---|
| **KV cache VMM arena** (`mem_cache/kv_vmm_backing.py`) | 디바이스마다 VMM 예약 영역을 `torch.cuda.MemPool`로 노출하고, 버퍼마다 필요한 범위만 커밋 | `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, `cuMemGetAllocationGranularity` |
| **DWDP** (`layers/moe/dwdp/`) | MoE prefill에서 토큰은 rank에 두고, 다른 rank의 전문가 가중치를 NVLink로 미리 가져와 하나의 VMM 주소 공간에 이어 붙임 | `cuMemExportToShareableHandle`, `cuMemImportFromShareableHandle`, `cuMemMap`, `cuMemcpyDtoD`, `cuMemRelease` |
| **멀티모달 feature 전송** (`--mm-feature-transport=cuda_vmm`) | tokenizer 프로세스가 GPU에서 만든 feature를 공유 핸들로 scheduler에 넘김 (복사 없음) | `cuMemExportToShareableHandle` (POSIX fd), `cuMemImportFromShareableHandle` |
| **Custom AllReduce v2** | symmetric push/pull 버퍼를 VMM으로 구성 | `cuda_vmm_utils` 공통 API |
| **CUDA Graph 입력 매핑** | graph capture 입력 포인터를 VMM base 기준으로 변환 | `cuMemGetAddressRange`, `cuMemRetainAllocationHandle` |

### 3.2 Green Context 기반 PD Multiplexing [A] [S]

한 GPU의 SM을 prefill용과 decode용으로 나눠, 두 단계를 같은 GPU에서 동시에 실행합니다 (`srt/multiplex/pdmux_context.py`가 `create_greenctx_stream_by_value(prefill_sm, decode_sm, gpu_id)` 호출).

| 단계 | API |
|---|---|
| 지원 확인 | `cuDriverGetVersion`, `cuGetProcAddress` |
| SM 분할 | `cuDeviceGetDevResource` → `cuDevSmResourceSplitByCount` → `cuDevResourceGenerateDesc` |
| 컨텍스트·스트림 생성 | `cuGreenCtxCreate`, `cuGreenCtxGetDevResource`, `cuGreenCtxStreamCreate` |
| 구버전 폴백 | `cuCtxFromGreenCtx`, `cuCtxPushCurrent`, `cuStreamCreate`, `cuCtxPopCurrent` |
| 해제 | `cuGreenCtxDestroy` |

### 3.3 CUDA Graph [S]

SGLang은 PyTorch의 `torch.cuda.CUDAGraph`로 캡처하고, 그 위에 최적화를 직접 얹습니다.

| 기법 | 하는 일 | API |
|---|---|---|
| **Graph exec dedup** | 캡처한 그래프들의 노드·엣지·커널 파라미터를 비교해, 구조가 같은 그래프끼리 executable 하나를 `cudaGraphExecUpdate`로 재사용 (instantiate 비용과 메모리 절약) | Driver: `cuGraphGetNodes`, `cuGraphGetEdges`, `cuGraphNodeGetType`, `cuGraphKernelNodeGetParams`, `cuGraphKernelNodeGetAttribute`, `cuGraphMemcpyNodeGetParams`, `cuGraphMemsetNodeGetParams`, `cuGraphChildGraphNodeGetGraph` / Runtime: `cudaGraphInstantiateWithFlags`, `cudaGraphExecUpdate`, `cudaGraphLaunch`, `cudaGraphExecDestroy` |
| **Breakable CUDA Graph** | 캡처 구간을 eager break point로 나눠 여러 그래프 조각으로 실행 | `cudaStreamGetCaptureInfo` |
| **공유 graph memory pool** (`runner_utils/pool.py`) | prefill과 decode 그래프가 메모리 풀 하나를 공유 (동시에 replay하지 않으므로 큰 쪽만큼만 예약) | (PyTorch API 사용) |
| Custom AllReduce의 캡처 대응 [A] | 캡처 중이면 버퍼를 나중에 일괄 등록 | `cudaStreamIsCapturing` |

### 3.4 HiCache (계층형 KV 캐시) [A] [J] [S]

GPU KV cache를 host 메모리(및 그 아래 저장소)로 확장합니다.

| 목적 | API | 위치 |
|---|---|---|
| host KV pool을 pinned로 등록 | `cudaHostRegister`, `cudaHostUnregister` (청크 단위) | [S] `pool_host/common.py` |
| 배치 복사 (CUDA 12.8 이상) | `cudaMemcpyBatchAsync` (`dlsym`), 버전 확인 `cudaRuntimeGetVersion` / `cudaDriverGetVersion` | [A] `kvcacheio/transfer.cu`, [J] `staged_write_back.cuh` |
| 폴백 | `cudaMemcpyAsync` 반복 | [A] [J] |
| Zero-copy 커널 전송 (host 메모리를 GPU 커널이 직접 접근) | `cudaHostGetDevicePointer` | [A] `transfer.cu`, [J] `runtime.cuh` |
| TMA 기반 전송 커널 | `cudaGetDriverEntryPointByVersion` → `cuTensorMapEncodeTiled` | [J] `hicache_tma.cuh` |

### 3.5 Custom AllReduce / 통신 최적화 [A] [J] [S]

| 기법 | API |
|---|---|
| IPC 버퍼 공유 | `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle`, `cudaIpcCloseMemHandle` [A] [J] [S] |
| IPC 버퍼 base 주소 | `cuPointerGetAttribute` [A], `cuMemGetAddressRange` [J] `ipc.cuh` |
| P2P 사전 테스트 | `cudaSetDevice`, `cudaMalloc`, `cudaMemset`, `cudaMemcpy`, `cudaDeviceSynchronize`, `cudaDeviceReset` (ctypes) [S] |
| AllReduce에 PDL 적용 | `cudaTriggerProgrammaticLaunchCompletion` [A] `custom_all_reduce.cuh` |
| FlashInfer AllReduce fusion 사전 점검 | `cuMulticastGetGranularity`, `cuMemGetAllocationGranularity`, `cuMemCreate`, `cuMemRelease` [S] |
| 멀티모달 생성 IPC all-to-all | `cudaDeviceEnablePeerAccess` [M] |

### 3.6 PDL (Programmatic Dependent Launch) [A] [J] [K]

| 위치 | API | 사용 커널 |
|---|---|---|
| 호스트 | `cudaLaunchKernelEx` | per_token_group_quant_8bit_v2, moe_finalize_fuse_shared, JIT 공통 launcher(`utils.cuh`), TRT-LLM MoE overlay |
| 디바이스 | `cudaGridDependencySynchronize`, `cudaTriggerProgrammaticLaunchCompletion` | 위 커널들 + dsv3_fused_a_gemm, custom_all_reduce |

### 3.7 프로세스 간 스트림 순서 동기화 [S]

멀티모달 feature 풀에서, 호스트 동기화 없이 GPU 메모리의 플래그 값으로 생산자·소비자 스트림을 맞춥니다.

| API |
|---|
| `cuStreamWriteValue32` (생산 완료 표시), `cuStreamWaitValue32` (소비 측 대기) |

### 3.8 CuTe DSL 커널 로딩 [K]

CuTe DSL로 컴파일한 CUBIN을 라이브러리 API로 로드합니다 (FlashAttention CuTe 구현 등).

| API | 종류 |
|---|---|
| `cuLibraryLoadData`, `cuLibraryEnumerateKernels`, `cuKernelGetFunction`, `cuFuncGetAttribute` | Driver |
| `cudaLibraryLoadData` | Runtime (`cute_dsl_ptxas.py`) |

### 3.9 커널 튜닝 / 하드웨어 적응형 실행 [A] [J]

| 목적 | API |
|---|---|
| 아키텍처·SM 수 조회 | `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`, `cudaGetDevice` |
| 대용량 dynamic smem | `cudaFuncSetAttribute` (AOT 11곳, JIT 34곳) |
| Persistent 커널 그리드 크기 | `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaOccupancyAvailableDynamicSMemPerBlock`, `cudaFuncGetAttributes` |
| Thread Block Cluster 수 | `cudaOccupancyMaxActiveClusters` (`cluster_probe.cuh`) |
| TMA descriptor | `cuTensorMapEncodeTiled` (KDA prefill, DeepSeek-V4 wo_a fused) |
| 런타임 버전에 따른 경로 선택 | `cudaRuntimeGetVersion` (`runtime.cuh`) |

### 3.10 Pinned 메모리 / CPU Offload

| 기능 | API | 위치 |
|---|---|---|
| 확산 모델 layerwise offload | `cudaHostRegister`, `cudaHostUnregister` | [M] |
| 호스트 공유 메모리 pin | `cudaHostRegister` | [S] `host_shared_memory.py` |
| Engram 테이블 pin | `cudaHostRegister` | [S] `layers/engram.py` |
| PD 분리 KV 스테이징 버퍼 | `cudaMalloc` | [S] `staging_handler.py` |

### 3.11 공통 / 부가

| 분류 | API |
|---|---|
| 에러 처리 | `cudaGetLastError`, `cudaGetErrorString`, `cuGetErrorString`, `cuGetErrorName` |
| 프로파일링 구간 | `cudaProfilerStart`, `cudaProfilerStop` [S] |

---

## 4. 요약

| 최적화 기법 | 위치 | Driver API | Runtime API |
|---|:---:|:---:|:---:|
| CUDA VMM 메모리 (KV arena, DWDP, feature 전송) | S | ● (핵심) | |
| Green Context PD Multiplexing | A, S | ● (핵심) | |
| CUDA Graph exec dedup | S | ● (그래프 분석) | ● (exec 관리) |
| Breakable CUDA Graph | S | | ● |
| HiCache KV 전송 | A, J, S | ○ (TMA) | ● (배치 복사, pinning) |
| Custom AllReduce / 통신 | A, J, S | ○ | ● |
| PDL | A, J, K | | ● |
| 스트림 메모리 연산 동기화 | S | ● | |
| CuTe DSL 커널 로딩 | K | ● | ○ |
| 커널 튜닝 | A, J | ○ (TMA) | ● |
| Pinned 메모리 / Offload | S, M | | ● |

- SGLang은 네 프로젝트 중 **Driver API를 가장 넓게 씁니다 (46개).** VMM, green context, 그래프 노드 분석, 스트림 메모리 연산처럼 Runtime API로는 할 수 없는 기능을 **Python(`cuda-python`)에서 직접** 다룹니다.
- 커널 쪽(AOT/JIT)은 vLLM과 비슷하게 Runtime API로 PDL, smem 설정, occupancy 계산을 합니다. TMA처럼 Driver 전용 기능만 entry point로 찾아서 씁니다.
- CUDA Graph는 PyTorch로 캡처하지만, 캡처한 그래프를 **Driver API로 열어 분석하고 executable을 공유**하는 최적화를 직접 구현합니다.

### 네 프로젝트 비교

| 항목 | vLLM | Ollama | ExecuTorch | SGLang |
|---|---|---|---|---|
| Driver API 개수 | 18 | 26 (본체 8 + llama.cpp·MLX) | 0 (+ AOTI 생성 코드 9) | **46** |
| Driver API 주 용도 | VMM(sleep mode), 배치 복사 | GPU 탐지, VMM 풀, JIT 실행 | 생성 커널 로드·실행 | VMM 공유, green context, 그래프 분석 |
| VMM 활용 | sleep mode | 메모리 풀 | 없음 | KV arena, 가중치·feature 공유 |
| CUDA Graph | PyTorch에 위임 | 직접 (캡처 / 노드 조립) | 직접 (캡처) | PyTorch 캡처 + Driver API로 분석·dedup |
| SM 분할 | 없음 | 없음 | 호출자가 green context 스트림 전달 가능 | green context로 PD multiplexing |
| KV 오프로드 배치 복사 | `cuMemcpyBatchAsync` (Driver) | 없음 | 없음 | `cudaMemcpyBatchAsync` (Runtime) |
