# 05. 커널 로딩: 직접 빌드, 미리 빌드한 CUBIN, JIT, AOT

커널을 어떻게 공급하느냐에 따라 필요한 API가 갈립니다. **커널을 프레임워크와 함께 nvcc로 빌드하면 로딩 API가 필요 없습니다.** 커널 바이너리를 **실행 중에 로드**해야 한다면(미리 빌드한 CUBIN, NVRTC JIT, 생성 코드) library API가 필요합니다. library API는 Driver(`cuLibrary*`)와 Runtime(`cudaLibrary*`) 양쪽에 있으므로 Driver가 꼭 필요하지는 않습니다. 다만 분석한 프로젝트 대부분은 Driver의 module/library API를 씁니다(Runtime은 SGLang의 `cudaLibraryLoadData`뿐).

| 공급 방식 | 대표 | 로딩 API | Driver 필요 |
|---|---|---|:---:|
| 함께 빌드 (`<<<>>>`, `cudaLaunchKernel(Ex)`) | vLLM, SGLang AOT, llama.cpp | 없음 (fatbin이 실행 파일에 포함) | 아니요 |
| 실행 시 nvcc 빌드 후 확장 모듈로 로드 | SGLang JIT (`tvm_ffi`) | 없음 (일반 공유 라이브러리) | 아니요 |
| **미리 빌드한 CUBIN** | TRT-LLM FMHA v2, trtllmGen | `cuModuleLoadData` | 선택 (Runtime `cudaLibraryLoadData`로 대체 가능) |
| **NVRTC JIT** | TRT-LLM XQA·DeepGEMM, MLX | `cuLibraryLoadData` / `cuModuleLoadDataEx` | 선택 (Runtime `cudaLibraryLoadData`로 대체 가능) |
| **DSL 컴파일 CUBIN** | SGLang CuTe DSL | `cuLibraryLoadData` 또는 `cudaLibraryLoadData` | 선택 |
| **AOT 생성 코드에 내장** | ExecuTorch (AOTInductor → Triton) | `cuModuleLoadData` (생성 코드 안에서) | **예** (생성 코드가 Driver를 사용하므로 런타임에 libcuda 필요) |

---

## T2-KLOAD 외부 CUBIN·JIT 커널 로딩

### 경로 1: Module API (context 종속)

- **API**: `cuModuleLoadData`(메모리의 CUBIN/PTX) 또는 `cuModuleLoadDataEx`(JIT 옵션 지정) → `cuModuleGetFunction` → `cuFuncSetAttribute`(dynamic smem) → `cuLaunchKernel`/`cuLaunchKernelEx` → `cuModuleUnload`
- **특징**: 모듈이 **현재 context에 묶입니다.** 여러 context(디바이스)에서 쓰려면 context마다 로드해야 하므로, 캐시 키에 context를 넣습니다 (TRT-LLM trtllmGen `cuCtxGetId`).
- **근거**
  - TRT-LLM: FMHA v2·trtllmGen의 미리 빌드한 CUBIN 9,322개 (`fused_multihead_attention_v2.cpp`, `trtllmGenKernels/*`)
  - MLX: NVRTC로 컴파일한 fused 커널 (`jit_module.cpp`)
  - ExecuTorch AOTI: `.so`에 내장한 Triton CUBIN (`embed_kernel_binary=True`이므로 파일 경로용 `cuModuleLoad`는 쓰지 않음)

### 경로 2: Library API (context 독립, CUDA 12.0+)

- **API**: `cuLibraryLoadData` → `cuLibraryGetKernel`(이름으로) 또는 `cuLibraryEnumerateKernels`/`cuLibraryGetKernelCount`(전부 나열) → `cuKernelSetAttribute` → `cuKernelGetFunction` 또는 `cuLaunchKernelEx`에 바로 → `cuLibraryUnload`
- **선택 API**: `cuLibraryGetGlobal`(전역 변수 주소, XQA), `cuKernelGetName`(DeepGEMM)
- **특징**: context에 묶이지 않으므로 한 번 로드해 모든 디바이스에서 씁니다. 새로 만든다면 이 경로를 추천합니다.
- **Runtime 대응**: `cudaLibraryLoadData`/`cudaLibraryLoadFromFile` → `cudaLibraryGetKernel` 또는 `cudaLibraryEnumerateKernels` → `cudaKernelSetAttributeForDevice` → `cudaLaunchKernel(Ex)`에 `cudaKernel_t`를 그대로 넘김 → `cudaLibraryUnload`. `cudaLibraryGetGlobal`도 있습니다. 분석한 프로젝트 중에는 SGLang(`cute_dsl_ptxas.py`)만 `cudaLibraryLoadData`를 씁니다. Runtime 경로를 쓰면 libcuda에 직접 링크하지 않고도 외부 CUBIN을 로드할 수 있습니다.
- **근거**: TRT-LLM XQA JIT(`decoderXQAImplJIT/cubinObj.cpp`), DeepGEMM JIT(`deep_gemm/runtime.cuh`), SGLang CuTe DSL(`cute_dsl_utils.py`)

### 공통으로 필요한 것

| 목적 | API |
|---|---|
| dynamic smem 설정 | `cuFuncSetAttribute` / `cuKernelSetAttribute` |
| 블록 크기 결정 | `cuOccupancyMaxPotentialBlockSize` (MLX) |
| cluster 크기 | `cuOccupancyMaxPotentialClusterSize` (TRT-LLM) |
| PDL·cluster 실행 | `cuLaunchKernelEx` |
| 그래프 노드로 추가 | `cuGraphAddKernelNode` (MLX) |
| 속성 조회 | `cuFuncGetAttribute` (SGLang, vLLM 벤치마크) |

### 런타임 커널을 Driver로 실행하기

`cudaGetKernel`은 nvcc로 함께 빌드한 커널의 `cudaKernel_t` 핸들을 줍니다. TRT-LLM KV v2는 이 핸들을 `cuLaunchKernel`로 실행해서, Driver API로 작성한 모듈 안에서도 일반 커널을 씁니다 (`kvCacheManagerV2Utils.cu`).

---

## 아키텍처 대응

| 방식 | 어떻게 맞는 커널을 고르나 | API |
|---|---|---|
| 함께 빌드 | fatbin에 여러 SM 버전을 넣고 드라이버가 선택 | (자동) |
| CUBIN 배포 | SM별 CUBIN을 따로 두고 compute capability로 선택 | `cudaGetDeviceProperties` / `cudaDeviceGetAttribute` |
| JIT | 실행 중인 GPU의 아키텍처로 컴파일 | NVRTC + 위와 같음 |
| AOT 변형 | 여러 SM용 AOTI 변형을 한 패키지에 담고, 없으면 PTX 폴백 | `cudaGetDeviceProperties` (ExecuTorch `major * 10 + minor`) |

---

## 결론

- **커널을 함께 빌드한다면 이 장의 API는 필요 없습니다.** vLLM은 module/library API를 전혀 쓰지 않습니다.
- **모델 설정(head 수, head dim, 양자화)에 맞춘 특화 커널**을 실행 중에 만들려면 NVRTC + library API(`cuLibraryLoadData` 또는 `cudaLibraryLoadData`)가 필요합니다 (TRT-LLM XQA, DeepGEMM, MLX fused 커널).
- **빌드 시간과 바이너리 크기**를 줄이려고 커널을 별도 CUBIN으로 배포한다면 로딩 API가 필요합니다 (TRT-LLM은 `cuModuleLoadData`/`cuLibraryLoadData`).
- 새로 만든다면 **Library API를 기본으로** 하세요. context 관리가 단순해집니다. 나머지 코드가 Runtime이라면 Runtime library API로 통일할 수 있습니다. 대상 툴킷에서 필요한 기능(JIT 옵션, 전역 변수 조회 등)이 Runtime 쪽에 있는지 확인하세요.
