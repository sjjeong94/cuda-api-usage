# 01. T0 필수 API: 최소 서빙 엔진

**결론부터 말하면, 최소 서빙 엔진은 Runtime API만으로 만들 수 있습니다.** ExecuTorch 런타임은 Driver API를 하나도 쓰지 않고, vLLM의 Driver API 16개도 모두 T1 이상의 기능(VMM, 배치 복사, IPC base 주소)과 그 에러 처리에만 쓰입니다.

아래 8개 기능은 프레임워크 5개가 모두 갖춘 기반입니다. 표의 API는 [PT] 프로파일이라면 PyTorch를 통해 간접적으로 쓰게 되고, [SA] 프로파일이라면 직접 호출해야 합니다.

---

## 1. 필수 API 한 장 요약

| 기능 | 필수 API | 서빙에서의 쓰임 |
|---|---|---|
| **T0-DEV** 디바이스 열거·선택 | `cudaGetDeviceCount`, `cudaGetDevice`, `cudaSetDevice` | 워커와 GPU 매핑, TP rank별 디바이스 선택 |
| **T0-CAP** 하드웨어·버전 조회 | `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`, `cudaRuntimeGetVersion`, `cudaDriverGetVersion` | SM 수와 compute capability에 맞는 커널·타일 선택, 기능 분기 |
| **T0-MEMINFO** 메모리 용량 산정 | `cudaMemGetInfo` | 가중치를 올린 뒤 남는 메모리로 **KV cache 블록 수**를 결정 |
| **T0-ALLOC** 디바이스 메모리 | `cudaMalloc`, `cudaFree`, `cudaMemsetAsync` | 가중치, KV, 활성값, workspace |
| **T0-XFER** Host↔Device 전송 | `cudaMallocHost`/`cudaFreeHost` 또는 `cudaHostRegister`/`cudaHostUnregister`, `cudaMemcpyAsync`, `cudaMemcpy` | 가중치 적재, 입력 토큰·샘플링 결과 전송 |
| **T0-STREAM** 스트림·이벤트 | `cudaStreamCreateWithFlags`, `cudaStreamDestroy`, `cudaStreamSynchronize`, `cudaStreamWaitEvent`, `cudaEventCreateWithFlags`, `cudaEventRecord`, `cudaEventQuery`, `cudaEventSynchronize`, `cudaEventDestroy` | 계산과 전송 겹치기, 스트림 간 순서 보장, 블로킹 없는 완료 확인 |
| **T0-LAUNCH** 커널 실행·설정 | `<<<>>>` 또는 `cudaLaunchKernel`, `cudaFuncSetAttribute` | 커널 실행, 48KB를 넘는 dynamic shared memory 사용 |
| **T0-ERR** 에러 처리 | `cudaGetLastError`, `cudaGetErrorString` | 커널 실행 후 확인, `CUDA_CHECK` 매크로 |

`cudaFuncSetAttribute`(`cudaFuncAttributeMaxDynamicSharedMemorySize`)는 "최적화"처럼 보이지만, FlashAttention, Marlin, MoE 같은 현대 LLM 커널은 48KB 이상의 shared memory를 기본으로 쓰기 때문에 사실상 필수입니다. 실제로 5개 프레임워크가 모두 사용합니다(ExecuTorch는 AOTI 생성 코드에서 Driver의 `cuFuncSetAttribute` 사용).

---

## 2. 기능 카드

### T0-MEMINFO 메모리 용량 산정

- **목적**: 서빙 엔진의 처리량은 KV cache 크기로 정해집니다. 가중치와 workspace를 올린 뒤 남는 메모리를 측정해 KV 블록 수를 정해야 합니다.
- **필수 API**: `cudaMemGetInfo` (free, total)
- **프로파일**: [PT]는 `torch.cuda.mem_get_info()`로 대신합니다(vLLM, SGLang). [SA]는 직접 호출합니다(llama.cpp `ggml_backend_cuda_get_device_memory`, MLX allocator, ExecuTorch `memory_tracker.h`).
- **근거**: TRT-LLM은 Driver의 `cuMemGetInfo`도 씁니다(`trace_log_utils.py`).
- **결론**: KV 용량을 동적으로 정하는 엔진이라면 필수입니다. Ollama는 이 값을 스케줄러의 offload 결정에도 씁니다. 다만 Ollama 본체는 Driver API(`cuDeviceTotalMem`)로 전체 용량을 조회하고, 실측 free 메모리는 runner(llama.cpp)가 `cudaMemGetInfo`로 확인합니다.

### T0-XFER Host↔Device 전송과 pinned 버퍼

- **목적**: 매 step 입력 토큰을 GPU로 보내고, 샘플링 결과를 CPU로 받습니다. 가중치도 처음에 올려야 합니다. pageable 메모리에서 하는 `cudaMemcpyAsync`는 사실상 동기 동작이라, 계산과 겹치려면 pinned 메모리가 필요합니다.
- **필수 API**: `cudaMemcpyAsync`, `cudaMemcpy`, pinned 할당(`cudaMallocHost`) 또는 기존 버퍼 pin(`cudaHostRegister`)
- **선택 API**
  - `cudaMemcpy2DAsync`: strided 텐서, row split 분배, scale 패딩 (vLLM, llama.cpp, TRT-LLM)
  - `cudaPointerGetAttributes`: 버퍼가 pinned·pageable·device 중 무엇인지 판별해 경로 선택 (vLLM cache_kernels, ExecuTorch AOTI 상수 적재, TRT-LLM)
  - `cudaMemcpyToSymbol`: constant memory에 LUT 적재 (vLLM W4A8)
- **설계 선택**
  - `cudaMallocHost`는 새 버퍼를 할당합니다. llama.cpp가 CPU에 남는 레이어용으로 기본으로 씁니다.
  - `cudaHostRegister`는 **이미 있는 메모리**(mmap한 모델 파일, 공유 메모리, 큰 KV host pool)를 pin합니다. vLLM KV offload, SGLang HiCache(큰 풀은 청크 단위로 등록), llama.cpp(`GGML_CUDA_REGISTER_HOST`, 기본 꺼짐)
  - ExecuTorch AOTI는 pageable 가중치를 pinned 버퍼로 스테이징한 뒤 전용 스트림(`cudaStreamCreateWithFlags(NonBlocking)`)으로 비동기 복사합니다.
- **결론**: `cudaMemcpyAsync`와 pinned 할당 방식 하나는 필수입니다. 큰 host 영역(KV 계층, mmap 가중치)을 다룰 계획이라면 `cudaHostRegister`가 필요합니다.

### T0-STREAM 스트림·이벤트 동기화

- **목적**: 계산, H2D/D2H 전송, 통신을 서로 다른 스트림에서 겹치고, 호스트가 블로킹 없이 GPU 진행 상황을 확인합니다.
- **필수 API**: 스트림 생성·해제·동기화, `cudaStreamWaitEvent`, 이벤트 생성·기록·조회·동기화
- **설계 포인트**
  - `cudaEventCreateWithFlags(cudaEventDisableTiming)`: 타이밍이 필요 없는 동기화용 이벤트는 오버헤드가 적습니다.
  - `cudaEventQuery`: 스케줄러 루프에서 블로킹 없이 완료 여부를 확인합니다 (ExecuTorch AOTI의 `is_finished()`, MLX).
  - `cudaStreamWaitEvent`: 호스트를 거치지 않는 스트림 간 의존성. 호출자가 다른 스트림을 넘기는 경우(ExecuTorch KV cache)나 multi-GPU 백엔드 간 동기화(llama.cpp)에 씁니다.
- **프로파일**: [PT]는 `torch.cuda.Stream`/`Event`가 담당해서 vLLM, SGLang에는 `cudaStream*`/`cudaEvent*` 생성 호출이 거의 없습니다. [SA]는 RAII 래퍼를 직접 만듭니다(llama.cpp `common.cuh`, MLX `event.cu`, ExecuTorch guard).
- **Driver 대응 API**: TRT-LLM KV Cache Manager v2는 `cuStreamCreate`, `cuEvent*`로 같은 일을 합니다. 모듈 전체를 Driver API로 통일하려는 선택이지, Driver가 꼭 필요해서가 아닙니다.
- **결론**: 필수입니다. 비동기 스케줄링(다음 batch를 준비하는 동안 현재 batch를 실행)을 하려면 이벤트 기반 완료 확인이 핵심입니다.

### T0-CAP 하드웨어·버전 조회

- **목적**: 같은 바이너리가 여러 세대의 GPU에서 돌아야 합니다. SM 수, compute capability, shared memory 한도에 따라 커널·타일·그리드를 고르고, CUDA·드라이버 버전에 따라 기능을 켜고 끕니다.
- **필수 API**: `cudaGetDeviceProperties` (한 번 조회해 캐시), `cudaDeviceGetAttribute` (개별 속성, 더 가벼움)
- **버전 분기**: `cudaRuntimeGetVersion`, `cudaDriverGetVersion`. 배치 복사(12.8+) 같은 기능을 켤지 정합니다 (SGLang `kvcacheio/transfer.cu`, vLLM NVFP4 scaled_mm). 자세한 패턴은 [08](08-driver-vs-runtime.md).
- **근거**: TRT-LLM은 `cudaDeviceGetAttribute`를 52곳, vLLM은 25곳에서 씁니다. ExecuTorch는 여러 SM용으로 컴파일한 AOTI 변형 중 하나를 `major * 10 + minor`로 고릅니다.
- **결론**: 필수입니다. 핫 패스에서 `cudaGetDeviceProperties`를 반복 호출하지 말고 캐시하세요.

### T0-DEV, T0-ALLOC, T0-LAUNCH, T0-ERR

설명이 필요 없는 기반 API입니다. 서빙 관점의 주의점만 적습니다.

- **T0-DEV**: `cudaSetDevice`는 **스레드별** 상태입니다. 여러 GPU를 한 프로세스에서 다룬다면 디바이스 가드(RAII)가 필요합니다 (ExecuTorch `device_guard.cpp`). `cudaSetDeviceFlags(cudaDeviceScheduleSpin)`은 호스트 동기화 지연을 줄입니다 (llama.cpp).
- **T0-ALLOC**: 요청마다 `cudaMalloc`/`cudaFree`를 부르면 디바이스 전체가 동기화되어 지연이 튑니다. 서빙 엔진은 반드시 그 위에 풀을 둡니다. [PT]는 PyTorch 캐싱 할당기, [SA]는 [02의 메모리 풀](02-memory.md) 중 하나를 씁니다.
- **T0-LAUNCH**: PDL이나 cluster 속성을 붙이려면 `<<<>>>` 대신 `cudaLaunchKernelEx`가 필요합니다([04](04-execution.md)).
- **T0-ERR**: 비동기 에러는 나중 호출에서 드러나므로, 디버그 빌드에서는 커널 실행 직후 `cudaGetLastError`로 확인하세요. TRT-LLM은 상태를 지우지 않는 `cudaPeekAtLastError`도 씁니다.

---

## 3. 보조 (AUX)

| 목적 | API | 근거 |
|---|---|---|
| 프로파일러 구간 지정 | `cudaProfilerStart`, `cudaProfilerStop` | SGLang `profiler_manager.py`, TRT-LLM `profiling.py`, vLLM 벤치마크 |
| GPU 시간 측정 | `cudaEventElapsedTime` (이벤트를 타이밍 가능으로 생성해야 함) | TRT-LLM·ExecuTorch 테스트 |
| 그래프 덤프 | `cudaGraphDebugDotPrint` | MLX (`MLX_SAVE_CUDA_GRAPHS_DOT_FILE`) |
