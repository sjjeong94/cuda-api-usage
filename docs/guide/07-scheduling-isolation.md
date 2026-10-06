# 07. 스케줄링·격리: SM 분할, 스트림 우선순위, GPU 측 신호

한 GPU에서 성격이 다른 작업(prefill과 decode, 계산과 전송, 여러 모델)을 동시에 돌릴 때, 서로 간섭을 줄이는 기능입니다. 스트림을 여러 개 쓰는 것만으로는 SM 할당을 제어할 수 없어서, 강한 격리를 원하면 Driver API(green context)가 필요합니다.

| 기능 | 등급 | 격리 수준 | 핵심 API | 종류 |
|---|:---:|---|---|---|
| 다중 스트림 fork/join | T0 조합 | 없음 (동시 실행만) | `cudaEventRecord`, `cudaStreamWaitEvent` | Runtime |
| [스트림 우선순위](#t2-prio-스트림-우선순위) | T2 | 약함 (스케줄링 우선) | `cudaStreamCreateWithPriority` | Runtime |
| [Green Context](#t2-green-green-context-sm-분할) | T2 | 강함 (SM 분할) | `cuGreenCtxCreate`, `cuGreenCtxStreamCreate` | **Driver** |
| [스트림 메모리 연산](#t2-streammem-스트림-메모리-연산) | T2 | — (프로세스 간 순서) | `cuStreamWaitValue32`, `cuStreamWriteValue32` | **Driver** |

---

## 다중 스트림 fork/join (T0 조합)

- **목적**: 서로 의존하지 않는 연산 가지를 여러 스트림에서 동시에 실행합니다.
- **API**: `cudaStreamCreateWithFlags`, `cudaEventCreateWithFlags`, `cudaEventRecord`, `cudaStreamWaitEvent`
- **근거**: llama.cpp concurrent streams(`GGML_CUDA_GRAPH_OPT=1`, 기본 꺼짐, `fork_event`/`join_events`), ExecuTorch AOTI 보조 스트림
- **결론**: T0 API만으로 됩니다. 다만 두 작업이 SM을 어떻게 나눌지는 하드웨어 스케줄러에 맡겨집니다.

## T2-PRIO 스트림 우선순위

- **목적**: 지연에 민감한 작업(KV 전송, decode)을 높은 우선순위 스트림에 올려, 대기 중인 블록 스케줄링에서 먼저 처리되게 합니다.
- **API**: `cudaDeviceGetStreamPriorityRange` → `cudaStreamCreateWithPriority`
- **근거**: TRT-LLM disaggregated serving NIXL bounce 전송 스트림(`nixl_utils/bounce/ExecPool.cpp`). 완료 확인은 `cudaStreamQuery`
- **결론**: 비용이 거의 없어서 전송·통신 스트림에 적용할 만합니다. 이미 실행 중인 블록을 선점하지는 않으므로 격리가 보장되지는 않습니다.

## T2-GREEN Green Context (SM 분할)

- **목적**: GPU의 SM을 나눠 각 파티션에 묶인 스트림을 만듭니다. 파티션끼리는 SM을 공유하지 않으므로, 한쪽의 큰 작업이 다른 쪽의 지연에 영향을 주지 않습니다.
- **필수 API (Driver)**

  | 단계 | API |
  |---|---|
  | 지원 확인 | `cuDriverGetVersion`, `cuGetProcAddress` (심볼 해석) |
  | SM 리소스 조회 | `cuDeviceGetDevResource` |
  | 분할 | `cuDevSmResourceSplitByCount` (SGLang) 또는 `cuDevSmResourceSplit` (TRT-LLM) |
  | descriptor | `cuDevResourceGenerateDesc` |
  | 생성 | `cuGreenCtxCreate`, (확인) `cuGreenCtxGetDevResource` |
  | 스트림 | `cuGreenCtxStreamCreate` |
  | 해제 | `cuGreenCtxDestroy`, `cuStreamDestroy` |

- **폴백 (구버전 드라이버)**: `cuGreenCtxStreamCreate`가 없으면 `cuCtxFromGreenCtx` → `cuCtxPushCurrent` → `cuStreamCreate` → `cuCtxPopCurrent` (SGLang)
- **요구 사항**: CUDA 12.4+ (`cuGreenCtxStreamCreate`는 12.5+). SM은 정해진 단위로만 나눌 수 있으므로 원하는 비율과 정확히 맞지 않을 수 있습니다.
- **활용**

  | 프레임워크 | 용도 |
  |---|---|
  | SGLang | **PD Multiplexing**: 한 GPU에서 prefill용과 decode용 SM을 나눠 동시에 실행 (`srt/multiplex/pdmux_context.py`가 `create_greenctx_stream_by_value(prefill_sm, decode_sm, gpu_id)` 호출) |
  | TRT-LLM | **Locality Domain**: GPU를 domain 두 개로 나눠 domain마다 SM 파티션·스트림·할당기(`cuMemAlloc`)를 둠 (`locality_domain_utils.cpp`) |
  | ExecuTorch | 직접 만들지 않음. 호출자가 green context 스트림을 넘기면 그 위에서 실행 (`CallerStreamGuard`) |
  | PyTorch | `torch.cuda.green_contexts.GreenContext.create(...)` → `GreenContext.Stream()`. cuda-python으로 `cuDevSmResourceSplitByCount`, `cuGreenCtxCreate`, `cuGreenCtxStreamCreate`를 부르고, SM 수 외에 **work queue 공유 범위**(`CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG`)도 설정. green context마다 스트림 32개를 풀로 둠 |

- **결론**: prefill-decode 간섭을 같은 GPU에서 해결하려면 필요합니다. PD를 GPU 단위로 분리(disaggregation)하는 대안과 비교해 선택하세요. **Runtime 대응 API가 없습니다.** [PT] 엔진이라면 PyTorch의 `GreenContext`로 시작할 수 있습니다. 반환된 스트림이 `torch.cuda.Stream`이라 PyTorch 연산을 그대로 올릴 수 있습니다. 라이브러리 형태 엔진이라면 ExecuTorch처럼 호출자의 스트림을 받는 인터페이스만 두는 방법도 있습니다.

## T2-STREAMMEM 스트림 메모리 연산

- **목적**: 호스트 동기화 없이, GPU 메모리의 플래그 값으로 서로 다른 프로세스의 스트림 순서를 맞춥니다.
- **API**: 생산자 `cuStreamWriteValue32`(완료 표시), 소비자 `cuStreamWaitValue32`(대기)
- **근거**: SGLang 멀티모달 feature 풀 (`srt/multimodal/transport/memory_pool.py`). VMM 공유 메모리([T2-VMM-SHARE](02-memory.md#t2-vmm-share-vmm-공유-핸들))와 함께 써서 tokenizer 프로세스가 GPU에서 만든 feature를 scheduler가 기다렸다 읽습니다.
- **결론**: 프로세스를 나눈 파이프라인(전처리 프로세스 → 엔진)에서 GPU 데이터를 주고받을 때 유용합니다. 신호가 GPU 메모리에 있으므로, 호스트가 개입하지 않고 스트림끼리 순서를 맞출 수 있습니다.
