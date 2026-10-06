# 06. 멀티 GPU: AllReduce, P2P, NVLS, MNNVL, Logical Endpoint

텐서 병렬(TP)과 전문가 병렬(EP)은 기본적으로 NCCL로 통신합니다. NCCL 내부 호출은 이 분석 범위 밖입니다. 이 장은 프레임워크가 **NCCL을 우회하거나 보완하려고 직접** 쓰는 CUDA API를 다룹니다. 단계가 올라갈수록 Driver API와 특정 하드웨어 의존이 커집니다.

| 기능 | 등급 | 하드웨어 조건 | 핵심 API | 종류 |
|---|:---:|---|---|---|
| [Custom AllReduce (IPC)](#t1-ipc-custom-allreduce-ipc-버퍼-공유) | T1 | 같은 노드, NVLink/P2P | `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle` | Runtime (+ Driver 1개) |
| [Peer access·P2P 복사](#t1-p2p-peer-accessp2p-복사) | T1 | 같은 노드 | `cudaDeviceEnablePeerAccess`, `cudaMemcpyPeerAsync` | Runtime |
| [PCIe AllReduce](#t3-pciear-pcie-allreduce) | T3 | NVLink 없는 PCIe 2-GPU | `cudaHostAlloc(Mapped)` | Runtime |
| [VMM 공유 기반 통신](#vmm-공유-기반-통신-t2) | T2 | 같은 노드 | `cuMemExportToShareableHandle` | **Driver** |
| [NVLS 멀티캐스트](#t3-nvls-nvls-멀티캐스트) | T3 | NVSwitch (NVLink SHARP) | `cuMulticastCreate`, `cuMulticastBindMem` | **Driver** |
| [MNNVL fabric 메모리](#t3-mnnvl-mnnvl-fabric-메모리) | T3 | 멀티노드 NVLink (GB200 NVL72), IMEX | FABRIC 핸들 | **Driver** |
| [Logical Endpoint](#t3-le-logical-endpoint) | T3 | NVSwitch fabric, IMEX, CUDA 13.4 | `cuLogicalEndpoint*` | **Driver** |

---

## T1-IPC Custom AllReduce (IPC 버퍼 공유)

- **목적**: decode 단계의 작은 텐서 AllReduce는 NCCL의 고정 오버헤드가 큽니다. 같은 노드의 GPU끼리 버퍼를 공유하고, 커널이 NVLink로 상대 GPU 메모리를 직접 읽고 써서 AllReduce를 한 번에 끝냅니다.
- **필수 API**

  | 단계 | API |
  |---|---|
  | 버퍼 할당 | `cudaMalloc` (+ 플래그 초기화 `cudaMemsetAsync`) |
  | 핸들 교환 | `cudaIpcGetMemHandle` → (프로세스 간 전달, 보통 `torch.distributed`) → `cudaIpcOpenMemHandle` |
  | 정리 | `cudaIpcCloseMemHandle` |
  | **base 주소** | `cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_RANGE_START_ADDR)` (**Driver**) |
  | 그래프 대응 | `cudaStreamIsCapturing`, `cudaThreadExchangeStreamCaptureMode` |

- **왜 Driver API가 하나 필요한가**: IPC 핸들은 **할당 단위**로 만들어집니다. 캐싱 할당기가 준 텐서는 큰 블록 안의 offset일 수 있으므로, 블록의 base 주소를 구해 핸들을 만들고 offset을 따로 전달해야 합니다. Runtime에는 이 정보를 주는 API가 없습니다.
- **선택 API**
  - P2P 가능 여부 사전 테스트: 별도 프로세스에서 `cudaSetDevice`, `cudaMalloc`, `cudaMemset`, `cudaMemcpy`, `cudaDeviceSynchronize`, `cudaDeviceReset` (vLLM `all_reduce_utils.py`, SGLang `custom_all_reduce_utils.py`)
  - `cudaDeviceCanAccessPeer` (TRT-LLM `ipcUtils.cpp`)
  - PDL 적용: AllReduce 커널에 `cudaTriggerProgrammaticLaunchCompletion` (SGLang)
- **CUDA Graph와 함께 쓰기**: 캡처 중 쓰인 버퍼는 주소만 기록해 두었다가 캡처가 끝난 뒤 일괄로 IPC 등록합니다 (vLLM·SGLang `custom_all_reduce.cuh`).
- **변형**: Lamport 방식 fused AllReduce + RMSNorm (vLLM MiniMax, cuda-python으로 `cudaIpc*`)
- **근거**: vLLM, SGLang(vLLM과 같은 구조의 ctypes 래퍼), TRT-LLM
- **결론**: TP 서빙의 **표준 최적화**입니다. Runtime API + Driver API 1개(`cuPointerGetAttribute`)로 만들 수 있습니다.

## T1-P2P Peer access·P2P 복사

- **목적**: layer split(파이프라인처럼 레이어를 GPU별로 나눔)에서 활성값을 GPU 간에 직접 복사합니다.
- **API**: `cudaDeviceCanAccessPeer` → `cudaDeviceEnablePeerAccess` → `cudaMemcpyPeerAsync`, 백엔드 간 순서는 `cudaEventRecord` + `cudaStreamWaitEvent`
- **근거**: llama.cpp(`ggml-cuda.cu`), TRT-LLM(`ipcUtils.cpp`, `allreduceOp.cpp`), SGLang 멀티모달 IPC all-to-all(`ipc_a2a.py`)
- **결론**: 같은 프로세스에서 여러 GPU를 다루는 [SA] 엔진이라면 필요합니다. 프로세스를 GPU마다 두는 [PT] 엔진은 IPC를 씁니다.

## T3-PCIEAR PCIe AllReduce

- **목적**: NVLink가 없는 소비자용 2-GPU 구성에서 tensor parallel을 하려고, pinned host 메모리를 거쳐 합산합니다.
- **API**: `cudaHostAlloc(cudaHostAllocMapped | cudaHostAllocPortable)`, `cudaHostGetDevicePointer`, 전용 스트림·이벤트, `cudaMemcpyAsync`, `cudaMemsetAsync`
- **근거**: llama.cpp `allreduce.cu` (`GGML_CUDA_ALLREDUCE`로 선택)
- **결론**: 로컬·소비자 GPU를 대상으로 하는 엔진에서만 의미가 있습니다.

## VMM 공유 기반 통신 (T2)

- **목적**: VMM으로 만든 메모리는 `cudaIpc*`로 공유할 수 없습니다. VMM 핸들을 공유해 symmetric 버퍼를 만들고, 원격 GPU 메모리를 한 가상 주소 공간에 이어 붙입니다.
- **API**: [T2-VMM-SHARE](02-memory.md#t2-vmm-share-vmm-공유-핸들) 참고. 원격 버퍼의 base는 `cuMemGetAddressRange` (SGLang `ipc.cuh`)
- **근거**
  - SGLang Custom AllReduce v2: symmetric push/pull 버퍼를 VMM으로
  - **DWDP** (SGLang, TRT-LLM): MoE prefill에서 다른 rank의 전문가 가중치를 NVLink로 가져와 하나의 VMM 주소 공간에 매핑. 복사에 `cuMemcpyDtoD`(SGLang), `cudaMemcpyBatchAsync`(TRT-LLM)
- **결론**: 새로 만든다면 VMM 기반 공유가 IPC보다 유연합니다(멀티캐스트·fabric으로 확장 가능). 대신 할당기를 VMM으로 만들어야 합니다.

## T3-NVLS NVLS 멀티캐스트

- **목적**: NVSwitch의 NVLink SHARP를 써서, 한 번 쓰면 모든 GPU에 브로드캐스트되고 스위치에서 reduce되는 멀티캐스트 메모리를 만듭니다. AllReduce 대역폭과 지연을 크게 줄입니다.
- **필수 API**

  | 단계 | API |
  |---|---|
  | 지원 확인 | `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED)` |
  | 멀티캐스트 객체 | `cuMulticastGetGranularity` → `cuMulticastCreate` → `cuMulticastAddDevice` |
  | 물리 메모리 | `cuMemCreate` → `cuMemExportToShareableHandle`/`cuMemImportFromShareableHandle` |
  | 바인딩 | `cuMulticastBindMem` (해제·재바인딩 `cuMulticastUnbind`) |
  | 매핑 | `cuMemAddressReserve` → `cuMemMap` → `cuMemSetAccess` |

- **요구 사항**: CUDA 12.1+, NVSwitch 시스템(HGX H100/H200/B200 등)
- **근거**: TRT-LLM `runtime/mcastDeviceMemory.cpp`, `ipcNvlsMemory.cu`, UserBuffers(`userbuffers-host.cpp`, GEMM과 통신 겹치기). SGLang은 FlashInfer AllReduce fusion을 쓰기 전에 `cuMulticastGetGranularity`로 지원 여부만 점검합니다.
- **PyTorch**: `torch.distributed._symmetric_memory`(CUDA 백엔드, `CUDASymmetricMemory.cu`)가 같은 시퀀스로 멀티캐스트 버퍼를 만듭니다. 지원 여부는 세 가지로 확인합니다: CUDA 12.3 이상으로 빌드했는지(컴파일 시점), `cuMulticastCreate` 심볼이 있는지(드라이버 535 이상), `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` 값. signal pad는 `cuMemsetD32Async`로 초기화하고 `cuStreamWriteValue32`로 씁니다. [PT] 엔진이라면 직접 구현하기 전에 이 기능을 먼저 검토할 만합니다.
- **결론**: NVSwitch 시스템에서 TP AllReduce를 극한까지 줄이려면 필요합니다. **Runtime 대응 API가 없습니다.**

## T3-MNNVL MNNVL fabric 메모리

- **목적**: GB200 NVL72처럼 여러 노드가 NVLink로 묶인 시스템에서, 다른 노드의 GPU 메모리를 직접 매핑합니다.
- **API**: `cuMemCreate`(`CU_MEM_HANDLE_TYPE_FABRIC`) → `cuMemExportToShareableHandle` → (노드 간 전달) → `cuMemImportFromShareableHandle` → `cuMemAddressReserve`, `cuMemMap`, `cuMemSetAccess`. 지원 확인은 `cuDeviceGetAttribute`
- **요구 사항**: 멀티노드 NVLink, IMEX 데몬
- **근거**: TRT-LLM `_mnnvl_utils.py`, disaggregated serving KV 전송 버퍼(`cacheTransBuffer.cpp`의 `FabricMemory`), 전송 에이전트 메모리 등록(`cuMemGetAddressRange`)
- **결론**: NVL72급 시스템을 대상으로 하는 엔진의 EP·disaggregation에 필요합니다.

## T3-LE Logical Endpoint

- **목적**: MoE AlltoAll에서 rank마다 peer별 unicast endpoint를 만들어 수신 버퍼에 연결하고, dispatch 커널이 fabric counted write로 상대 endpoint에 직접 씁니다.
- **API** (모두 `cuGetProcAddress`로 해석): `cuLogicalEndpointIdReserve`/`IdRelease`, `cuLogicalEndpointCreate`/`Destroy`, `cuLogicalEndpointBindMem`/`Unbind`, `cuLogicalEndpointExport`/`Import`, `cuLogicalEndpointQuery` + 컨텍스트 준비(`cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent`)
- **요구 사항**: CUDA 13.4 헤더, LE 지원 드라이버, IMEX 데몬, NVSwitch fabric
- **근거**: TRT-LLM `kernels/moe/communication/moeAlltoAllCftManager.h`
- **결론**: 최신 NVSwitch fabric에서 대규모 EP를 할 때만 해당합니다. 다섯 프로젝트 중 TRT-LLM만 씁니다.

---

## 단계별 도입 순서

1. **NCCL만** (CUDA API 직접 호출 없음): 모든 엔진의 출발점
2. **+ Custom AllReduce (T1)**: Runtime IPC + `cuPointerGetAttribute`. 같은 노드 TP의 decode 지연 개선
3. **+ VMM 공유 (T2)**: 할당기를 VMM으로 바꾸면 IPC를 대체하고 DWDP 같은 가중치 공유가 가능해짐
4. **+ NVLS / MNNVL / Logical Endpoint (T3)**: 대상 하드웨어가 NVSwitch·NVL72일 때만
