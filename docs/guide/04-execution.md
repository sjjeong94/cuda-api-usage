# 04. 실행: CUDA Graph, PDL, 커널 튜닝, Hopper 기능

디코딩 단계는 토큰 하나를 만들 때마다 작은 커널 수백 개를 실행합니다. batch가 작으면 커널 실행 시간보다 **launch 오버헤드와 커널 사이의 빈틈**이 더 커집니다. 이 장의 기능은 대부분 그 빈틈을 줄이는 것입니다.

| 기능 | 등급 | 줄이는 것 | 핵심 API | 종류 |
|---|:---:|---|---|---|
| [CUDA Graph](#t1-graph-cuda-graph) | T1 | CPU launch 오버헤드 | `cudaStreamBeginCapture`, `cudaGraphInstantiate`, `cudaGraphLaunch` | Runtime |
| [캡처 인지 동작](#t1-capaware-캡처-인지-동작) | T1 | 캡처 실패 | `cudaStreamIsCapturing` | Runtime |
| [PDL](#t1-pdl-programmatic-dependent-launch) | T1 | 커널 사이의 GPU 빈틈 | `cudaLaunchKernelEx` + 디바이스 API | Runtime |
| [Occupancy 기반 구성](#t1-occ-occupancy-기반-launch-구성) | T1 | SM 낭비 | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | Runtime |
| [TMA](#t2-tma-tma-descriptor) | T2 | 메모리 접근 비용 | `cuTensorMapEncodeTiled` | **Driver** |
| [Thread Block Cluster](#t2-cluster-thread-block-cluster) | T2 | SM 간 데이터 공유 | `cudaLaunchKernelEx`(cluster 속성) | Runtime |
| [그래프 직접 조립·분석](#t2-graph-build-그래프-직접-조립분석) | T2 | 그래프 인스턴스 비용·메모리 | `cudaGraphAddKernelNode`, `cuGraphGetNodes` | 둘 다 |
| [Cooperative launch](#t2-coop-cooperative-launch) | T2 | 커널 분할 | `cudaLaunchCooperativeKernel` | Runtime |
| [호스트 콜백](#t2-hostfn-스트림-순서-호스트-콜백) | T2 | 호스트 동기화 대기 | `cudaLaunchHostFunc` | Runtime |

---

## T1-GRAPH CUDA Graph

- **목적**: 한 step의 커널 수백 개를 그래프 하나로 묶어 한 번에 실행합니다. CPU launch 오버헤드가 사라지고, 호스트가 느려도 GPU가 쉬지 않습니다.
- **필수 API ([SA])**

  | 단계 | API |
  |---|---|
  | 캡처 | `cudaStreamBeginCapture` → (step 실행) → `cudaStreamEndCapture` |
  | 인스턴스화 | `cudaGraphInstantiate` (또는 `cudaGraphInstantiateWithFlags`) |
  | 실행 | `cudaGraphLaunch` |
  | 재사용 | `cudaGraphExecUpdate`: 구조가 같은 새 그래프로 기존 exec를 갱신해 인스턴스화 비용을 피함 |
  | 해제 | `cudaGraphExecDestroy`, `cudaGraphDestroy` |

- **선택 API**
  - `cudaGraphInstantiateFlagAutoFreeOnLaunch`: 그래프 안에서 할당한 메모리를 다음 launch 전에 자동 해제 (ExecuTorch)
  - `cudaDeviceGraphMemTrim`: 그래프 전용 메모리 반환 (ExecuTorch)
- **설계 포인트**
  - 입력은 고정 주소의 static buffer에 복사한 뒤 replay합니다 (`cudaMemcpyAsync`).
  - batch 크기마다 그래프를 따로 캡처하고, 실제 batch는 가장 가까운 크기로 padding합니다.
  - RNG 상태 초기화처럼 캡처할 수 없는 작업은 warmup에서 끝냅니다 (ExecuTorch `rand.cu`, warmup 3회 후 캡처).
- **프로파일**: [PT]는 `torch.cuda.CUDAGraph`로 캡처하므로 vLLM, SGLang, TRT-LLM 본체에는 `cudaGraph*` 호출이 없습니다 (TRT-LLM은 테스트에만 있음). [SA]는 직접 구현합니다 (llama.cpp 기본 켜짐, MLX 기본 켜짐, ExecuTorch method별 opt-in).
- **결론**: **디코딩 지연을 낮추려면 사실상 필수**입니다. [PT]라면 PyTorch 기능을 쓰고, 아래의 캡처 인지 동작만 직접 처리하면 됩니다.

## T1-CAPAWARE 캡처 인지 동작

- **목적**: 캡처 중에는 동기화, 일반 할당, 일부 라이브러리 호출을 할 수 없습니다. 직접 만든 커널이나 통신 코드가 캡처 중인지 알고 다르게 동작해야 합니다.
- **API**

  | API | 쓰임 | 근거 |
  |---|---|---|
  | `cudaStreamIsCapturing` | 캡처 중이면 다른 경로 선택 (캡처 비호환 CUB 정렬 회피, 버퍼 등록 지연) | llama.cpp `argsort.cu`, vLLM·SGLang Custom AllReduce, TRT-LLM NCCL·FMHA |
  | `cudaThreadExchangeStreamCaptureMode` | 캡처 중에도 버퍼를 할당할 수 있게 Relaxed 모드로 전환 | vLLM `custom_all_reduce.cu` |
  | `cudaStreamGetCaptureInfo` | 캡처 상태와 그래프 핸들 조회. 캡처 구간을 여러 조각으로 나누는 breakable graph | SGLang, TRT-LLM `breakable_cuda_graph.py` |

- **결론**: **[PT] 프로파일에서도 직접 필요한** CUDA Graph 관련 API입니다. 특히 Custom AllReduce처럼 캡처 중에 IPC 버퍼를 다루는 코드는 이 API 없이는 그래프와 함께 쓸 수 없습니다.

## T1-PDL Programmatic Dependent Launch

- **목적**: 앞 커널이 끝나기 전에 다음 커널을 미리 띄웁니다. 다음 커널은 prologue(인덱스 계산, 가중치 prefetch)를 먼저 하고, 앞 커널의 결과가 필요한 지점에서만 기다립니다. CUDA Graph가 줄이지 못하는 **GPU 쪽 커널 간 빈틈**을 줄입니다.
- **필수 API**

  | 위치 | API |
  |---|---|
  | 호스트 | `cudaLaunchKernelEx` + `cudaLaunchAttributeProgrammaticStreamSerialization` |
  | 디바이스 (뒤 커널) | `cudaGridDependencySynchronize()`: 앞 커널의 결과가 필요한 지점에서 대기 |
  | 디바이스 (앞 커널) | `cudaTriggerProgrammaticLaunchCompletion()`: 뒤 커널이 일찍 시작해도 된다는 신호 |

- **Driver 대응**: JIT/CUBIN 커널은 `cuLaunchKernelEx`로 같은 속성을 붙입니다 (TRT-LLM XQA·trtllmGen).
- **요구 사항**: 디바이스 측 동작은 SM90(Hopper) 이상에서만 효과가 있습니다(llama.cpp 문서 기준). 구형 GPU 동작은 대상 툴킷에서 확인하고, 필요하면 아키텍처로 분기하세요. llama.cpp는 CUDA 12.3+(Linux 11.8+)로 빌드할 때만 컴파일합니다(`GGML_CUDA_USE_PDL`).
- **설계 포인트**: 공통 launcher 함수 하나에서 PDL 속성을 붙이고(llama.cpp `ggml_cuda_kernel_launch`, SGLang JIT `utils.cuh`), 커널마다 PDL 지원 여부를 표시합니다. 런타임 스위치를 두세요 (llama.cpp `GGML_CUDA_PDL=0`).
- **근거**: TRT-LLM(호스트 117곳+, 디바이스 약 300곳), vLLM(호스트 27곳, 디바이스 약 70곳), SGLang, llama.cpp(기본 켜짐)
- **결론**: Hopper 이상을 대상으로 하는 서빙 엔진의 **표준 최적화**입니다. Runtime API만으로 되고, 커널 코드에 두 줄을 넣는 방식이라 점진적으로 적용할 수 있습니다.

## T1-OCC Occupancy 기반 launch 구성

- **목적**: persistent 커널(SM마다 블록을 상주시켜 작업을 나눠 처리)이나 stream-K 분할에서, GPU 전체를 채우는 그리드 크기를 계산합니다.
- **API**
  - `cudaOccupancyMaxActiveBlocksPerMultiprocessor`: SM당 블록 수 × SM 수 = 그리드 크기 (vLLM SM100 MLA·topk, llama.cpp FlashAttention, SGLang, TRT-LLM 15곳)
  - `cudaOccupancyMaxPotentialBlockSize`: 최적 블록 크기 (MLX)
  - `cudaOccupancyAvailableDynamicSMemPerBlock`: 주어진 occupancy에서 쓸 수 있는 smem (SGLang `runtime.cuh`)
  - JIT 커널용 Driver 대응: `cuOccupancyMaxPotentialBlockSize` (MLX)
- **결론**: 직접 작성한 attention·MoE·top-k 커널이 있다면 필요합니다. 외부 라이브러리 커널만 쓴다면 불필요합니다.

## T2-TMA TMA descriptor

- **목적**: Hopper의 Tensor Memory Accelerator로 전역 메모리 ↔ shared memory 타일 복사를 하드웨어가 처리하게 합니다. 최신 GEMM·attention 커널의 기반입니다.
- **필수 API**: `cuTensorMapEncodeTiled` (**Driver 전용**, Runtime 대응 API 없음)
- **Runtime 프로젝트에서 쓰는 방법**: libcuda에 직접 링크하지 않고 `cudaGetDriverEntryPoint` 또는 `cudaGetDriverEntryPointByVersion`(12.5+)으로 함수 포인터를 얻습니다 (SGLang `hicache_tma.cuh`, TRT-LLM DeepGEMM `tma_utils.cuh`).
- **요구 사항**: SM90+, CUDA 12.0+
- **근거**: TRT-LLM(XQA, trtllmGen, Mamba, mHC 등), SGLang(KDA prefill, DeepSeek-V4), MLX(FP 양자화), ExecuTorch AOTI(Triton TMA 커널)
- **결론**: Hopper용 커널을 **직접 작성**한다면 필요합니다. CUTLASS·Triton 커널을 쓰면 라이브러리 안에서 처리되지만, descriptor를 호스트에서 만들어 넘기는 구조라면 직접 호출하게 됩니다.

## T2-CLUSTER Thread Block Cluster

- **목적**: 블록 여러 개를 cluster로 묶어 distributed shared memory를 공유하고 함께 스케줄링되게 합니다.
- **API**
  - 실행: `cudaLaunchKernelEx`/`cudaLaunchKernelExC`의 `cudaLaunchAttributeClusterDimension`, JIT 커널은 `cuLaunchKernelEx`
  - 크기 결정: `cudaOccupancyMaxActiveClusters` (SGLang `cluster_probe.cuh`), `cuOccupancyMaxPotentialClusterSize` (TRT-LLM CuTe DSL top-k)
  - 그래프 노드: `cudaGraphKernelNodeSetAttribute`, `cuGraphKernelNodeSetAttribute` (MLX)
- **요구 사항**: SM90+, CUDA 11.8+
- **결론**: cluster를 쓰는 커널을 직접 실행할 때만 필요합니다. PDL과 같은 `cudaLaunchKernelEx` 경로를 쓰므로 launcher를 하나로 만들어 두면 추가 비용이 작습니다.

## T2-GRAPH-BUILD 그래프 직접 조립·분석

캡처 이상으로 그래프를 다루는 두 방향이 있습니다.

**(a) 노드 직접 조립 [SA]** — MLX

- **목적**: 매 step 스트림 캡처를 하지 않고, 커널을 노드로 바로 추가해 그래프를 만듭니다. 캡처할 수 없는 cuBLAS·cuDNN 호출만 캡처해서 child graph로 넣습니다.
- **API**: `cudaGraphCreate`, `cudaGraphAddKernelNode`(JIT 커널은 `cuGraphAddKernelNode`), `cudaGraphAddEmptyNode`, `cudaGraphAddDependencies`, `cudaGraphAddChildGraphNode`, `cudaGraphExecUpdate`
- **설계 포인트**: 그래프 캐시(MLX 기본 400개, `MLX_CUDA_GRAPH_CACHE_SIZE`)와 구조 비교(`cudaGraphGetNodes`, `cudaGraphNodeGetType`)

**(b) 캡처한 그래프 분석·dedup [PT]** — SGLang

- **목적**: batch 크기별로 캡처한 그래프 수십 개 중 구조가 같은 것끼리 executable 하나를 공유해 인스턴스화 비용과 메모리를 줄입니다.
- **API**: Driver의 `cuGraphGetNodes`, `cuGraphGetEdges`, `cuGraphNodeGetType`, `cuGraphKernelNodeGetParams`, `cuGraphKernelNodeGetAttribute`, `cuGraphMemcpyNodeGetParams`, `cuGraphMemsetNodeGetParams`, `cuGraphChildGraphNodeGetGraph`로 비교한 뒤, Runtime의 `cudaGraphInstantiateWithFlags`, `cudaGraphExecUpdate`, `cudaGraphLaunch`
- **참고**: Python(cuda-python)에서는 Driver 바인딩으로 그래프 노드를 다루는 편이 쉬워서 Driver API를 썼습니다. 같은 기능의 Runtime API(`cudaGraphGetNodes` 등)도 있습니다.

**(c) 그래프 메모리 추적** — ExecuTorch: `cudaGraphGetNodes` + `cudaGraphMemAllocNodeGetParams`/`cudaGraphMemFreeNodeGetParams`로 free 노드 없이 남는 할당을 찾아 그래프와 함께 해제

- **결론**: 기본 캡처로 충분하지 않을 때 고려합니다. 그래프 수가 많아 메모리·초기화 시간이 문제라면 (b), 캡처 비용 자체를 없애려면 (a)입니다.

## T2-COOP Cooperative launch

- **목적**: 그리드 전체 동기화(`grid.sync()`)가 필요한 알고리즘을 커널 하나로 실행합니다.
- **API**: `cudaLaunchCooperativeKernel` (그리드가 동시에 상주할 수 있어야 하므로 occupancy API로 크기 결정)
- **근거**: llama.cpp 열 방향 병렬 softmax(`softmax.cu`), TRT-LLM EPLB(`moeLoadBalanceKernels.cu`)
- **결론**: 특정 커널에서만 필요합니다.

## T2-HOSTFN 스트림 순서 호스트 콜백

- **목적**: 스트림의 앞선 작업이 끝나는 시점에 호스트 함수를 실행합니다. 호스트가 기다리지 않아도 됩니다.
- **API**: `cudaLaunchHostFunc` (TRT-LLM은 `cudaLaunchHostFunc_v2`, `cudaStreamAddCallback`도 사용, KV v2는 Driver의 `cuLaunchHostFunc`)
- **근거**
  - MLX: 가중치 업로드가 끝나면 host 버퍼 해제 (`load.cpp`), 완료 시그널 (`worker.cpp`)
  - TRT-LLM: 스트림(그래프 안 포함)에서 Python 함수 호출 (`nanobind/runtime/hostfunc.cpp`, GIL 획득), KV v2 비동기 순서 관리
- **결론**: 리소스 정리를 비동기로 하거나, 그래프 안에 host 작업을 넣어야 할 때 씁니다. 콜백 안에서 CUDA API를 부르면 안 됩니다.

---

## 실행 경로 설계 요약

1. **launcher를 하나로 만드세요.** `cudaLaunchKernelEx` 기반 공통 launcher가 PDL·cluster 속성을 붙이고, JIT 커널은 `cuLaunchKernelEx` 버전을 둡니다. llama.cpp, SGLang, TRT-LLM이 모두 이 구조입니다.
2. **CUDA Graph는 [PT]면 PyTorch에 맡기고**, 직접 만든 통신·커널 코드에만 `cudaStreamIsCapturing` 분기를 넣으세요.
3. **Hopper 이상이 대상이면** PDL(Runtime)은 바로 적용하고, TMA(Driver, entry point 경유)는 직접 작성하는 커널에 필요할 때 추가하세요.
