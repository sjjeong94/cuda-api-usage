# ExecuTorch CUDA API 사용 현황 및 최적화 기법 매핑

> 기준:
> - ExecuTorch `main` 커밋 `91b2e90` (2026-10-06)
> - PyTorch `v2.14.0` (`2b3ec34`): AOTInductor 코드 생성 템플릿과 `aoti_runtime` 헤더. ExecuTorch `torch_pin.py`는 `2.14.0` nightly `dev20260913`을 가리키므로, 릴리스 태그로 근사했습니다
>
> 방법: `cuXxx(` / `cudaXxx(` 호출을 grep하고 주석·타입·enum·로그 문자열·HIP shim의 함수 정의는 제외. 템플릿 인자로 넘기는 함수도 별도로 확인
> 범위: ExecuTorch CUDA 백엔드가 실행하는 코드가 **직접** 호출하는 API. cuBLAS, cuRAND, Thrust/CUB, Triton 등 라이브러리 내부 호출은 제외

---

## 목차

0. [ExecuTorch의 CUDA 코드 구성](#0-executorch의-cuda-코드-구성)
1. [CUDA Driver API](#1-cuda-driver-api)
2. [CUDA Runtime API](#2-cuda-runtime-api)
3. [최적화 기법별 API 매핑](#3-최적화-기법별-api-매핑)
4. [요약](#4-요약)

---

## 0. ExecuTorch의 CUDA 코드 구성

ExecuTorch CUDA 백엔드는 **AOTInductor(AOTI) 기반**입니다. export할 때 PyTorch Inductor가 모델을 Triton 커널과 C++ wrapper로 컴파일해서 `.so`를 만들고, 런타임은 이 `.so`를 `dlopen`해서 실행합니다. 그래서 실제로 GPU에서 도는 코드는 두 곳에서 옵니다.

| 구성 요소 | 소스 | 역할 |
|---|---|---|
| **ExecuTorch CUDA 런타임** | `backends/cuda/runtime/`, `backends/aoti/`, `extension/cuda/` | `.so` 로드, 메모리 할당, CUDA Graph, KV cache, AOTI C shim(`aoti_torch_*`) 구현, 커스텀 양자화 커널 |
| **AOTI 생성 코드** (`.so` 안) | PyTorch `torch/_inductor/codegen/`의 템플릿 + `torch/csrc/inductor/aoti_runtime/` 헤더 | Triton 커널 로드·실행, 상수(가중치) 적재, 실행 완료 이벤트 |

ExecuTorch가 AOTI에 넘기는 주요 컴파일 옵션(`backends/cuda/cuda_backend.py`)은 다음과 같습니다.

| 옵션 | 값 | 영향 |
|---|---|---|
| `aot_inductor.embed_kernel_binary` | `True` | 커널 CUBIN을 `.so`에 내장 → `cuModuleLoadData`로 로드 |
| `aot_inductor.link_libtorch` | `False` | libtorch 없이 실행. 텐서·메모리 연산은 ExecuTorch의 `aoti_torch_*` shim이 처리 |
| `aot_inductor.package_constants_in_so` | `False` | 가중치는 `.so` 밖(`.ptd`)에 두고 런타임에 적재 |
| `max_autotune` + `max_autotune_gemm_backends` | `True`, `"TRITON"` | GEMM·conv도 Triton 커널로 생성 |
| `aot_inductor.custom_ops_to_c_shims` | int4/5/6/8 `plain_mm` | 커스텀 양자화 GEMM을 ExecuTorch 쪽 CUDA 커널로 연결 |

**제외한 항목**
- `backends/mlx`: `MLX_BUILD_CUDA=OFF`, `MLX_BUILD_METAL=ON`이라 Apple 전용입니다.
- `third-party/ao`(torchao): export 단계의 Python 라이브러리이고, 런타임에 포함되지 않습니다.
- `extension/cuda/runtime_api.h`: CUDA 이름을 HIP로 바꾸는 shim이라 함수 정의는 세지 않았습니다.

---

## 1. CUDA Driver API

| 구성 요소 | 개수 |
|---|---:|
| ExecuTorch 런타임 | **0** |
| AOTI 생성 코드 | 9 (설정상 실제 사용 7) |

### 1.1 ExecuTorch 런타임

**직접 호출하는 Driver API는 없습니다.** `cuGreenCtxStreamCreate`는 `extension/cuda/caller_stream.h`의 주석에 나오는데, 호출자가 green context로 만든 스트림을 넘길 수 있다는 설명일 뿐입니다.

### 1.2 AOTI 생성 코드

`torch/_inductor/codegen/cuda/device_op_overrides.py`(커널 로드·실행 템플릿), `cpp_wrapper_gpu.py`, `aoti_runtime/model_base.h`

| API | 용도 | ExecuTorch 설정에서 |
|---|---|---|
| `cuModuleLoadData` | `.so`에 내장된 CUBIN 로드 | **사용** (`embed_kernel_binary=True`) |
| `cuModuleLoad` | 파일 경로로 CUBIN 로드 | 미사용 (내장 모드라서) |
| `cuModuleGetFunction` | Triton 커널 함수 획득 | 사용 |
| `cuFuncSetAttribute` | smem이 필요한 커널의 max dynamic smem 설정 (`sharedMemBytes > 0`) | 사용 |
| `cuLaunchKernel` | **모든 Triton 커널 실행** | 사용 |
| `cuTensorMapEncodeTiled` | TMA descriptor 생성 (Triton TMA 커널, Hopper 이상) | 해당 커널이 생성될 때만 |
| `cuModuleUnload` | 모델 해제 시 모듈 언로드 | 사용 |
| `cuGetErrorString` | `CUDA_DRIVER_CHECK` 에러 메시지 | 사용 |
| `cuCtxSynchronize` | 디버그 동기화 (`generate_debug_sync`) | 디버그 옵션에서만 |

---

## 2. CUDA Runtime API

| 구성 요소 | 호스트 API |
|---|---:|
| ExecuTorch 런타임 | 37 |
| AOTI 생성 코드 | 20 (ROCm 전용 1 포함) |
| 테스트·예제에서만 | 12 |

### 2.1 ExecuTorch 런타임

#### 디바이스 / 스트림 가드

| API | 주요 위치 |
|---|---|
| `cudaGetDevice`, `cudaSetDevice` | `extension/cuda/device_guard.cpp`, `backends/aoti/slim/cuda/guard.cpp`, `cuda_allocator.cpp`, `cuda_kv_cache.cpp` |
| `cudaStreamCreate` | `backends/aoti/slim/cuda/guard.cpp` (디바이스별 스트림) |
| `cudaGetDeviceProperties` | `cuda_backend.cpp` (GPU SM에 맞는 AOTI 변형 선택) |
| `cudaMemGetInfo` | `memory_tracker.h` (메모리 사용량·peak 로깅) |
| `cudaDeviceSynchronize` | `cuda_backend.cpp`, `cuda_kv_cache.cpp`, `cuda_mutable_state.cpp` |

#### 메모리 할당 (`cuda_allocator.cpp`: `CudaAllocator`)

| API | 용도 |
|---|---|
| `cudaMalloc` / `cudaFree` | 동기 할당 (`allocate` / `deallocate`) |
| `cudaMemPoolCreate` | 디바이스별 전용 메모리 풀 생성 (실패하면 기본 풀 사용) |
| `cudaMemPoolSetAttribute` | `cudaMemPoolAttrReleaseThreshold` 설정 (해제한 메모리를 풀에 유지) |
| `cudaMallocFromPoolAsync` | 전용 풀에서 스트림 순서 할당 (`allocate_async`) |
| `cudaMallocAsync` / `cudaFreeAsync` | 기본 풀 폴백, `deallocate_async` |
| `cudaMemPoolTrimTo` | `release_cached_memory`에서 풀을 비움 (풀 자체는 재사용을 위해 유지) |
| `cudaDeviceGraphMemTrim` | CUDA Graph용 메모리 반환 |
| `cudaMemcpy` / `cudaMemcpyAsync` | `copy_host_to_device`, `copy_device_to_host`, `memcpy_async` |
| `cudaStreamSynchronize` | 동기화 |

#### CUDA Graph (`cuda_backend.cpp`, `cuda_delegate_handle.h`)

| API | 용도 |
|---|---|
| `cudaStreamBeginCapture` / `cudaStreamEndCapture` | AOTI 실행 1회를 캡처 |
| `cudaGraphInstantiate` | `cudaGraphInstantiateFlagAutoFreeOnLaunch`로 인스턴스화 |
| `cudaGraphLaunch` | replay |
| `cudaGraphDestroy` / `cudaGraphExecDestroy` | 해제 |
| `cudaGraphGetNodes` / `cudaGraphNodeGetType` / `cudaGraphMemAllocNodeGetParams` / `cudaGraphMemFreeNodeGetParams` | 캡처된 그래프에서 free 노드 없이 남는 메모리 할당 노드를 찾아 기록 (그래프와 함께 해제하기 위해) |

#### KV Cache / Mutable State / Weight Cache

| API | 위치 | 용도 |
|---|---|---|
| `cudaEventCreateWithFlags` / `cudaEventRecord` / `cudaStreamWaitEvent` / `cudaEventDestroy` | `cuda_kv_cache.cpp` | 이전 step이 다른 스트림(호출자 지정 스트림)에서 돌았을 때 순서 보장 |
| `cudaMemcpyAsync` | `cuda_kv_cache.cpp` | KV 저장소를 늘릴 때 기존 내용 복사 |
| `cudaMalloc` / `cudaMemcpy` / `cudaPointerGetAttributes` / `cudaFree` | `cuda_mutable_state.cpp` | mutable buffer 세션별 복제 |
| `cudaMalloc` / `cudaMemcpy` / `cudaFree` | `cuda_weight_cache.cpp` | 여러 method가 같은 가중치를 공유 |

#### AOTI C shim / 커스텀 커널 (`backends/cuda/runtime/shims/`)

| API | 위치 | 용도 |
|---|---|---|
| `cudaMemsetAsync` | `memory.cpp` | `aoti_torch_zero_` |
| `cudaMallocAsync` / `cudaFreeAsync` / `cudaMemcpyAsync` | `sort.cu` | Thrust `par_nosync`용 스트림 순서 임시 버퍼 |
| `cudaMallocAsync` / `cudaMemcpyAsync` / `cudaStreamSynchronize` | `rand.cu` | RNG 상태 초기화 (그래프 캡처 전 warmup에서 1회) |
| `cudaMalloc` / `cudaFree` | `int{4,5,6,8}_plain_mm.cuh` | 활성값 int8 양자화 버퍼 (필요할 때 확장) |
| `cudaFuncGetAttributes` | `int4mm.cuh` | 커널 속성 조회 (tinygemm 계열) |
| `cudaMemcpy` / `cudaMemcpyAsync` / `cudaStreamSynchronize` | `slim/core/storage.h`, `utils.h` | SlimTensor 저장소 복사 |

#### 에러 처리

| API | 용도 |
|---|---|
| `cudaGetLastError` / `cudaGetErrorString` | `ET_CUDA_CHECK`, `slim/c10/cuda/Exception.h` |

### 2.2 AOTI 생성 코드 (`aoti_runtime/model_base.h`, `model_container.h`, `cpp_wrapper_gpu.py`)

| 분류 | API | 용도 |
|---|---|---|
| 상수(가중치) 메모리 | `cudaMalloc`, `cudaFree`, `cudaMemcpy` | 상수 블롭 할당과 적재 |
| 상수 비동기 적재 | `cudaPointerGetAttributes` | 상수마다 pageable host인지 판별 |
| | `cudaStreamCreateWithFlags` (`NonBlocking`), `cudaEventCreateWithFlags`, `cudaMemcpyAsync` | pageable → pinned 스테이징 후 전용 스트림으로 비동기 복사 |
| 실행 완료 추적 | `cudaEventCreate`, `cudaEventRecord`, `cudaEventQuery`, `cudaEventSynchronize`, `cudaEventDestroy` | `run_finished_` 이벤트 (`is_finished()`) |
| 멀티 스트림 | `cudaStreamWaitEvent`, `cudaEventRecord` | 보조 스트림을 쓰는 그래프의 이벤트 동기화 |
| 스트림 / 디바이스 | `cudaStreamSynchronize`, `cudaStreamDestroy`, `cudaGetDevice`, `cudaSetDevice` | |
| 에러 | `cudaGetLastError`, `cudaGetErrorString` | `AOTI_RUNTIME_CUDA_CHECK` |
| 디버그 (ROCm 전용) | `cudaDeviceSynchronize` | NVIDIA에서는 `cuCtxSynchronize`를 대신 씀 |

### 2.3 테스트 / 예제에서만 사용

| 위치 | 추가 API |
|---|---|
| `backends/cuda/runtime/test`, `backends/aoti/slim/*/test` | `cudaDeviceGetGraphMemAttribute`, `cudaDeviceGetMemPool`, `cudaMemPoolGetAttribute`, `cudaEventCreate`, `cudaEventElapsedTime`, `cudaEventSynchronize`, `cudaMallocHost`, `cudaFreeHost`, `cudaGetDeviceCount`, `cudaLaunchHostFunc`, `cudaStreamCreateWithFlags`, `cudaStreamDestroy` |
| `examples/models/` (qwen3_5_moe, gemma4_31b, muse-glimmer 엔진) | `cudaMemGetInfo`, `cudaPointerGetAttributes`, `cudaMemcpy(Async)`, `cudaDeviceSynchronize`. 런타임에 이미 있는 API만 사용 |

---

## 3. 최적화 기법별 API 매핑

표기: **[E]** ExecuTorch 런타임, **[A]** AOTI 생성 코드

### 3.1 AOT 컴파일 + 커널 내장 실행 [A]

export할 때 Inductor가 연산 fusion과 max-autotune을 거쳐 Triton 커널을 만들고, CUBIN을 `.so`에 내장합니다. 런타임에는 JIT 컴파일 없이 Driver API로 바로 로드해서 실행합니다. libtorch 의존성이 없어서(`link_libtorch=False`) 배포 크기가 작습니다.

| 단계 | API | 종류 |
|---|---|---|
| 로드 | `cuModuleLoadData` → `cuModuleGetFunction` | Driver |
| 실행 설정 | `cuFuncSetAttribute` (dynamic smem) | Driver |
| 실행 | `cuLaunchKernel` | Driver |
| 해제 | `cuModuleUnload` | Driver |
| Hopper TMA 커널 | `cuTensorMapEncodeTiled` | Driver |

### 3.2 CUDA Graph [E]

method 단위로 켭니다(backend option `enable_cuda_graph_for_method`, **기본 꺼짐**). warmup 실행 3회(`kCudaGraphWarmupSteps`) 뒤 AOTI 실행 1회를 캡처하고, 이후에는 입력을 static buffer에 복사한 뒤 replay합니다.

| 단계 | API |
|---|---|
| 캡처 | `cudaStreamBeginCapture` → (AOTI 실행) → `cudaStreamEndCapture` |
| 인스턴스화 | `cudaGraphInstantiate` (`AutoFreeOnLaunch`: 그래프 안에서 할당한 메모리를 다음 launch 전에 자동 해제) |
| 남는 할당 추적 | `cudaGraphGetNodes`, `cudaGraphNodeGetType`, `cudaGraphMemAllocNodeGetParams`, `cudaGraphMemFreeNodeGetParams` |
| Replay | `cudaMemcpyAsync` (입력 → static buffer), `cudaGraphLaunch` |
| 해제 | `cudaGraphExecDestroy`, `cudaGraphDestroy`, `cudaDeviceGraphMemTrim` |
| 캡처 전 준비 | RNG 상태 초기화를 warmup에서 끝냄 (`rand.cu`: `cudaMallocAsync`, `cudaMemcpyAsync`) |

### 3.3 스트림 순서 메모리 풀 [E]

`cudaMalloc`·`cudaFree`의 동기화 비용을 없애고, 해제한 메모리를 풀에 남겨 재사용합니다. 그래프 캡처 안에서도 할당할 수 있습니다.

| 목적 | API |
|---|---|
| 디바이스별 전용 풀 | `cudaMemPoolCreate`, `cudaMemPoolSetAttribute` (`ReleaseThreshold`) |
| 할당 / 해제 | `cudaMallocFromPoolAsync`, `cudaFreeAsync` (풀 생성 실패 시 `cudaMallocAsync`) |
| 캐시 반환 | `cudaMemPoolTrimTo(pool, 0)` |
| Thrust 정렬의 임시 버퍼 | `cudaMallocAsync`, `cudaFreeAsync` (`par_nosync` 커스텀 할당기) |

### 3.4 Off-graph KV Cache (지연 확장) [E]

export할 때 `kvcache::update_and_attend`를 `index_copy_` + Triton SDPA로 바꾸고(`passes/lower_offgraph_kv.py`), KV 저장소는 그래프 밖에서 관리합니다. 최대 길이만큼 미리 잡지 않고 필요할 때 늘립니다.

| 목적 | API |
|---|---|
| 확장 시 기존 내용 복사 | `cudaMemcpyAsync` (할당은 `CudaAllocator` 경유) |
| 다른 스트림에서 돈 이전 step과 순서 보장 | `cudaEventCreateWithFlags`, `cudaEventRecord`, `cudaStreamWaitEvent` |
| 디바이스 전환 | `cudaGetDevice`, `cudaSetDevice` |

### 3.5 호출자 지정 스트림 / Green Context [E]

`extension/cuda/caller_stream.h`의 `CallerStreamGuard`로 호출자가 실행 스트림을 지정할 수 있습니다. green context 스트림을 넘기면 SM 일부만 쓰도록 격리됩니다. 이 기능 자체는 CUDA API를 부르지 않고, 스트림 간 순서는 3.4의 이벤트 API로 맞춥니다.

### 3.6 가중치 적재 최적화

| 기법 | API | 위치 |
|---|---|---|
| 여러 method가 같은 가중치를 한 번만 올림 | `cudaMalloc`, `cudaMemcpy` | [E] `cuda_weight_cache.cpp` |
| pageable 가중치를 pinned로 스테이징해 비동기 복사 | `cudaPointerGetAttributes`, `cudaStreamCreateWithFlags`, `cudaEventCreateWithFlags`, `cudaMemcpyAsync` | [A] `model_base.h`, `model_container.h` |
| 상수 블롭 할당 | `cudaMalloc`, `cudaMemcpy`, `cudaFree` | [A] |

### 3.7 Mutable State 세션 [E]

한 번 로드한 프로그램을 mutable buffer(KV 등)만 따로 가진 여러 인스턴스로 실행합니다. 모델을 여러 번 로드하지 않아도 됩니다.

| API |
|---|
| `cudaMalloc`, `cudaMemcpy`, `cudaPointerGetAttributes`, `cudaFree`, `cudaDeviceSynchronize` |

### 3.8 커스텀 양자화 GEMM [E]

AOTI가 만들기 어려운 저비트 GEMM은 ExecuTorch 쪽 CUDA 커널로 연결합니다 (`custom_ops_to_c_shims`).

| 커널 | API | 용도 |
|---|---|---|
| int4/5/6/8 `plain_mm` (dp4a) | `cudaMalloc`, `cudaFree` | 활성값 int8 양자화 버퍼 (커지면 재할당) |
| int4mm (tinygemm 계열) | `cudaFuncGetAttributes` | 커널 속성 조회 |

### 3.8.1 SM별 AOTI 변형 선택 [E]

하나의 `.pte`/`.ptd`에 여러 SM용으로 컴파일한 AOTI 변형을 함께 담을 수 있습니다(`merge_ptes.py`, `emit_multi_arch_kernel`). 로드할 때 GPU의 compute capability를 읽어 맞는 변형을 고르고, 맞는 것이 없으면 PTX 폴백이나 SM 미지정 변형을 씁니다.

| API |
|---|
| `cudaGetDevice`, `cudaGetDeviceProperties` (`major * 10 + minor`) |

### 3.9 실행 완료 추적 [A]

| API | 용도 |
|---|---|
| `cudaEventCreate`, `cudaEventRecord`, `cudaEventQuery`, `cudaEventSynchronize`, `cudaEventDestroy` | 실행 끝에 이벤트를 기록해 호스트가 블로킹 없이 완료 여부를 확인 |

### 3.10 안전성 / 관측

| 목적 | API | 위치 |
|---|---|---|
| CUDA 텐서가 실제로는 host 메모리인 경우 검출 | `cudaPointerGetAttributes` | [E] `cuda_backend.cpp` |
| 메모리 사용량·peak 로깅 | `cudaMemGetInfo` | [E] `memory_tracker.h` |
| 에러 처리 | `cudaGetLastError`, `cudaGetErrorString`, `cuGetErrorString` | [E] [A] |
| 디버그 동기화 | `cuCtxSynchronize` | [A] 디버그 옵션 |

---

## 4. 요약

| 최적화 기법 | 위치 | Driver API | Runtime API | 기본값 |
|---|:---:|:---:|:---:|:---:|
| AOT 컴파일 + 커널 내장 실행 | A | ● (핵심) | | 항상 |
| TMA 커널 | A | ● | | 커널 의존 |
| CUDA Graph | E | | ● | 꺼짐 (method별 opt-in) |
| 스트림 순서 메모리 풀 | E | | ● | 켜짐 |
| Off-graph KV Cache | E | | ● | export 설정 의존 |
| 호출자 스트림 / Green Context | E | | ○ | 호출자 선택 |
| 가중치 공유 / 비동기 적재 | E, A | | ● | 켜짐 |
| Mutable State 세션 | E | | ● | 호출자 선택 |
| 커스텀 양자화 GEMM | E | | ● | 양자화 설정 의존 |
| SM별 AOTI 변형 선택 | E | | ● | 변형이 여럿일 때 |
| 실행 완료 추적 | A | | ● | 항상 |

- **ExecuTorch 자체 코드는 Driver API를 전혀 쓰지 않습니다.** 런타임은 Runtime API 37개로 메모리 풀, CUDA Graph, KV cache, AOTI shim을 구현합니다.
- **Driver API는 모두 AOTI 생성 코드에서 옵니다.** 모든 Triton 커널이 `cuModuleLoadData` → `cuLaunchKernel` 경로로 실행됩니다.
- 커널 최적화(fusion, autotune, 타일 선택)는 **export 단계의 Inductor**에서 끝나기 때문에, 런타임에 커널 튜닝용 API(`cudaOccupancy*`, `cudaFuncSetAttribute` 등)를 거의 쓰지 않습니다.

### 세 프로젝트 비교

| 항목 | vLLM | Ollama | ExecuTorch |
|---|---|---|---|
| 커널 생성 | 손으로 작성 (`csrc/`) + 외부 라이브러리 | 손으로 작성 (llama.cpp) + JIT (MLX) | **AOT 생성** (Inductor → Triton) |
| Driver API 주 용도 | VMM(sleep mode), 배치 복사 | GPU 탐지, VMM 풀, JIT 실행 | 생성 커널 로드·실행 |
| 메모리 관리 | PyTorch 캐싱 할당기 + VMM | VMM 풀 / `cudaMallocAsync` | `cudaMallocFromPoolAsync` (전용 풀) |
| CUDA Graph | PyTorch에 위임 | 직접 (캡처 / 노드 조립) | 직접 (캡처, method별 opt-in) |
| 런타임 커널 튜닝 | 많음 (smem, occupancy, PDL) | 많음 | 거의 없음 (export 때 완료) |
