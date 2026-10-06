# 08. Driver vs Runtime: 언제 Driver가 필요하고 어떻게 쓰나

## 1. 결정 규칙

**기본은 Runtime API입니다.** 아래 표의 기능이 필요할 때만 Driver API를 씁니다. 표에 없는 기능이라면 같은 일을 하는 Runtime API가 있습니다.

| 필요한 기능 | Driver API | Runtime 대응 | 가이드 |
|---|---|---|---|
| 주소를 유지하는 확장·반납 (VMM 풀, Sleep/Wake) | `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess` … | 없음 | [02](02-memory.md) |
| VMM 메모리 공유 (POSIX fd, FABRIC) | `cuMemExportToShareableHandle`, `cuMemImportFromShareableHandle` | 없음 (`cudaIpc*`는 `cudaMalloc` 메모리만) | [02](02-memory.md), [06](06-multi-gpu.md) |
| 할당 블록의 base 주소 | `cuPointerGetAttribute(RANGE_START_ADDR)`, `cuMemGetAddressRange` | 없음 | [06](06-multi-gpu.md) |
| 외부 CUBIN·JIT 커널 로드 | `cuModuleLoadData`, `cuLibraryLoadData` … | `cudaLibraryLoadData` (일부) | [05](05-kernel-loading.md) |
| TMA descriptor | `cuTensorMapEncodeTiled` | 없음 (entry point로 호출) | [04](04-execution.md) |
| SM 분할 | `cuGreenCtx*`, `cuDevSmResourceSplit*` | 없음 | [07](07-scheduling-isolation.md) |
| NVSwitch 멀티캐스트 | `cuMulticast*` | 없음 | [06](06-multi-gpu.md) |
| Logical Endpoint | `cuLogicalEndpoint*` | 없음 | [06](06-multi-gpu.md) |
| GPU 측 신호 | `cuStreamWaitValue32`, `cuStreamWriteValue32` | 없음 | [07](07-scheduling-isolation.md) |
| CUDA 툴킷 없이 GPU 탐지 | `cuInit`, `cuDeviceGet*` (libcuda만 `dlopen`) | — (libcudart 필요) | [아래 4절](#4-t2-drvdetect-드라이버만으로-gpu-탐지) |

**같은 기능의 Driver·Runtime 두 버전이 모두 있는 경우**: 배치 복사(`cuMemcpyBatchAsync` / `cudaMemcpyBatchAsync`), 그래프 노드 조회(`cuGraphGetNodes` / `cudaGraphGetNodes`), 스트림·이벤트·메모리 기본 연산. 이때는 모듈의 다른 코드와 같은 계층을 고르면 됩니다. TRT-LLM KV Cache Manager v2는 모듈 전체를 Driver로 통일했고, SGLang은 Python에서 cuda-python Driver 바인딩이 편해서 그래프 분석을 Driver로 했습니다.

**집계로 본 근거**: 다섯 프레임워크와 PyTorch가 쓰는 Driver API 127개 중 T0 등급은 30개뿐이고, 모두 Runtime 대응 API가 있는 `alt` 역할입니다 ([카탈로그](../reference/api-catalog.md) 2절). Driver가 **반드시** 필요한 API는 모두 T1 이상의 기능에 속합니다. PyTorch도 같은 패턴입니다. Driver API 50개를 VMM(expandable segments), 공유·멀티캐스트(symmetric memory), 커널 로딩(jiterator, Inductor), green context, 컨텍스트 안전성에만 씁니다.

---

## 2. T2-VERGATE 버전 의존 심볼 해석

서빙 엔진은 다양한 드라이버 버전에서 돌아야 합니다. 새 API를 빌드 시점에 직접 링크하면 구버전 드라이버에서 **로드 자체가 실패**합니다. 그래서 새 API는 실행 중에 심볼을 찾고, 없으면 폴백합니다.

| 방법 | API | 쓰는 곳 |
|---|---|---|
| Driver 심볼을 버전 지정으로 찾기 | `cuGetProcAddress(name, &fn, cudaVersion, flags, &status)` | vLLM `cuMemcpyBatchAsync`(12080), SGLang `cuGreenCtxStreamCreate`, TRT-LLM Logical Endpoint·green context |
| Runtime에서 Driver 심볼 찾기 | `cudaGetDriverEntryPoint`, `cudaGetDriverEntryPointByVersion` (12.5+) | SGLang `hicache_tma.cuh`, TRT-LLM DeepGEMM (`cuTensorMapEncodeTiled`), **PyTorch `c10/cuda/driver_api.cpp`** (Driver 테이블 전체) |
| Runtime 심볼 찾기 | `dlsym(libcudart, ...)` | SGLang `cudaMemcpyBatchAsync` |
| 버전 확인 후 분기 | `cuDriverGetVersion`, `cudaDriverGetVersion`, `cudaRuntimeGetVersion` | SGLang green context·`transfer.cu`, vLLM NVFP4 |

**권장 패턴**

1. 기능 하나당 "해석 함수"를 하나 두고, 처음 호출할 때 심볼을 찾아 캐시합니다.
2. 심볼이 없거나 드라이버 버전이 낮으면 폴백 경로를 고르고, 어떤 경로를 골랐는지 로그를 남깁니다.
3. 폴백 경로도 테스트합니다 (예: 배치 복사를 끄는 스위치).

**참고 구현: PyTorch c10 Driver 테이블** (`c10/cuda/driver_api.{h,cpp}`)

- 매크로로 함수 목록과 **요청 버전**을 함께 선언합니다: `_(cuMemCreate, 12000)`, `_(cuMulticastCreate, 12030)`, `_(cuGreenCtxCreate, 12080)`, `_(cuLogsRegisterCallback, 12090)`.
- CUDA 12.5 이상으로 빌드했다면 `cudaGetDriverEntryPointByVersion(name, &fn, version, ...)`으로 해석합니다. 실패하면, CUDA 13 미만 빌드에서는 deprecated된 `cudaGetDriverEntryPoint`로 한 번 더 시도합니다.
- 필수 그룹은 심볼이 없으면 `TORCH_INTERNAL_ASSERT`로 실패합니다. 선택 그룹(멀티캐스트, green context, 로그 콜백)은 `nullptr`로 남겨 두고, 호출하는 쪽이 포인터를 확인해 폴백합니다 (예: `driver_api->cuMulticastCreate_ != nullptr`).
- 주석에 적힌 원칙: **요청 버전은 가능한 한 낮게** 잡습니다. 최신 버전을 요청하면 새 드라이버에서 동작이 바뀐 새 버전 함수에 묶일 수 있기 때문입니다.
- 같은 테이블로 NVML도 `dlopen`합니다.

---

## 3. T2-CTX 컨텍스트 관리 (Driver·Runtime 혼용)

Runtime API는 디바이스마다 **primary context**를 자동으로 만들고 씁니다. Driver API를 섞어 쓸 때 문제가 되는 것은 "현재 스레드에 context가 없는" 경우입니다.

| 상황 | API | 쓰는 곳 |
|---|---|---|
| 할당기가 Runtime 초기화 전 스레드에서 불림 | `cuCtxGetCurrent` → 없으면 `cuDevicePrimaryCtxRetain` → `cuCtxSetCurrent` | vLLM `cumem_allocator.cpp`, TRT-LLM Logical Endpoint |
| 현재 context의 디바이스 확인 | `cuCtxGetDevice` | TRT-LLM `cudaVirtMem.cpp`, `_mnnvl_utils.py` |
| 다른 context(green context 등)에서 잠깐 작업 | `cuCtxPushCurrent` / `cuCtxPopCurrent` | SGLang green context 폴백 |
| context별 캐시 키 | `cuCtxGetId` | TRT-LLM trtllmGen 모듈 캐시 |

**권장**: 별도 context(`cuCtxCreate`)를 만들지 말고 **primary context를 공유**하세요. Runtime API, PyTorch, NCCL이 모두 primary context를 쓰므로, 별도 context에서 만든 메모리나 스트림은 서로 섞어 쓸 수 없습니다. 다섯 프로젝트와 PyTorch 모두 본체 코드에서 `cuCtxCreate`를 쓰지 않습니다 (TRT-LLM 테스트에만 있음).

PyTorch는 cuBLAS·cuFFT·UCC·Triton을 부르기 전에 `cuCtxGetCurrent`로 확인하고, context가 없으면 `cuDevicePrimaryCtxRetain` → `cuCtxSetCurrent`로 설정합니다. context를 만들지 않고 이미 있는지만 확인할 때는 `cuDevicePrimaryCtxGetState`를 씁니다 (`torch.cuda` lazy init, fork 전 검사).

---

## 4. T2-DRVDETECT 드라이버만으로 GPU 탐지

- **목적**: 서버 프로세스가 CUDA 툴킷 없이(드라이버만 설치된 환경에서) GPU 개수, 이름, compute capability, 메모리, 드라이버 버전을 수집하고, 그 정보로 맞는 runner(CUDA 버전·아키텍처별 빌드)를 고릅니다.
- **API**: `libcuda.so`/`nvcuda.dll`을 `dlopen` → `cuInit`, `cuDriverGetVersion`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetAttribute`(CC, integrated 여부), `cuDeviceGetName`, `cuDeviceTotalMem_v2`(폴백 `cuDeviceTotalMem`), `cuDeviceGetPCIBusId`(선택)
- **근거**: Ollama `discover/native_probe_{linux,windows}.go` (cgo + `dlopen`). vLLM CI는 같은 방식으로 GPU UUID를 수집합니다(`cuDeviceGetUuid_v2`).
- **결론**: 여러 CUDA 버전용 바이너리를 함께 배포하는 엔진(로컬 실행 도구 등)이라면 유용합니다. 단일 컨테이너로 배포하는 데이터센터 엔진에는 필요 없습니다.

---

## 5. 호출 방식

| 방식 | 장점 | 단점 | 쓰는 곳 |
|---|---|---|---|
| libcuda·libcudart 직접 링크 | 단순 | 새 API를 쓰면 구버전 드라이버에서 로드 실패 | 대부분의 C++ 코드 |
| `dlopen` + 함수 테이블 | 드라이버 없이도 프로세스 시작 가능, 버전별 분기 | 래퍼 유지 비용 | TRT-LLM `cudaDriverWrapper`(25개 노출), Ollama |
| `cuGetProcAddress` / entry point | 링크는 유지하고 새 API만 동적으로 | 함수 포인터 타입 관리 | vLLM, SGLang, TRT-LLM |
| Python ctypes | 빌드 없이 Runtime 호출 | 타입 안전성 없음 | vLLM `CudaRTLibrary`, SGLang `cuda_wrapper.py` |
| Python cuda-python | Driver·Runtime 전체를 타입 있게 | 의존성 추가 | SGLang VMM·그래프 분석, TRT-LLM MNNVL·DWDP, vLLM MiniMax |

**권장**: C++ 코어는 Runtime에 직접 링크하고, Driver API는 `cuGetProcAddress`/`cudaGetDriverEntryPoint`로 쓰는 기능만 해석하세요. Python 계층에서 Driver 기능(VMM, 그래프 분석)을 다룬다면 cuda-python을 쓰세요.

---

## 6. 에러 처리

Driver와 Runtime은 에러 타입이 다릅니다(`CUresult` / `cudaError_t`). 매크로를 따로 둡니다.

| 계층 | API | 예 |
|---|---|---|
| Runtime | `cudaGetLastError`, `cudaGetErrorString`, `cudaPeekAtLastError`, `cudaGetErrorName` | `CUDA_CHECK` |
| Driver | `cuGetErrorString`, `cuGetErrorName` | `CU_CHECK`, `TLLM_CU_CHECK`, `CUDA_DRIVER_CHECK` (AOTI) |
