#!/usr/bin/env python3
"""Build docs/redesign/data/api_redesign_map.csv: every API in api_meta.csv -> redesigned API.

change:
  merged   - the call folds into a redesigned function (often as a descriptor field)
  removed  - the redesign makes the call unnecessary (reason says why)
  helper   - not in the core ABI; provided by the optional convenience header

Usage: python3 scripts/redesign/build_map.py
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "docs" / "reference" / "data" / "api_meta.csv"
OUT = ROOT / "docs" / "redesign" / "data" / "api_redesign_map.csv"

# Redesigned core API (name -> group). Order is the order used in the design doc.
CORE = {
    "cuxGetApi": "init", "cuxGetErrorInfo": "init", "cuxSetLogCallback": "init",
    "cuxRelease": "object", "cuxExport": "object", "cuxImport": "object",
    "cuxDeviceList": "device", "cuxDeviceQuery": "device", "cuxDeviceMemInfo": "device", "cuxDevicePeerQuery": "device",
    "cuxPartitionCreate": "partition",
    "cuxQueueCreate": "queue", "cuxQueueQuery": "queue",
    "cuxFenceCreate": "fence", "cuxFenceQuery": "fence", "cuxFenceWait": "fence", "cuxFenceSignal": "fence",
    "cuxAddressReserve": "memory", "cuxMemCreate": "memory", "cuxMemMap": "memory", "cuxMemUnmap": "memory",
    "cuxPointerQuery": "memory", "cuxMemAdvise": "memory",
    "cuxPoolCreate": "pool", "cuxPoolAlloc": "pool", "cuxPoolFree": "pool", "cuxPoolQuery": "pool",
    "cuxPoolTrim": "pool", "cuxPoolUpdate": "pool",
    "cuxEnqueueCopy": "command", "cuxEnqueueFill": "command", "cuxEnqueueLaunch": "command",
    "cuxEnqueueHostFn": "command", "cuxEnqueueSignal": "command", "cuxEnqueueWait": "command",
    "cuxEnqueueGraph": "command",
    "cuxLibraryLoad": "kernel", "cuxLibraryGetKernels": "kernel", "cuxKernelFromSymbol": "kernel",
    "cuxKernelQuery": "kernel", "cuxKernelSetAttr": "kernel", "cuxGetGlobal": "kernel", "cuxTensorMapEncode": "kernel",
    "cuxCaptureBegin": "graph", "cuxCaptureEnd": "graph", "cuxGraphCreate": "graph", "cuxGraphAddNode": "graph",
    "cuxGraphInspect": "graph", "cuxGraphNodeQuery": "graph", "cuxGraphInstantiate": "graph",
    "cuxGraphUpdate": "graph", "cuxGraphRetain": "graph",
    "cuxMulticastCreate": "fabric", "cuxMulticastBind": "fabric", "cuxEndpointCreate": "fabric",
    "cuxEndpointQuery": "fabric",
}

M = {}  # api -> (new, change, reason)


def m(apis, new, change="merged", reason=""):
    for a in apis.split():
        if a in M:
            sys.exit(f"duplicate mapping: {a}")
        if change != "removed" and new not in CORE and change != "helper":
            sys.exit(f"{a}: unknown core API {new}")
        M[a] = (new, change, reason)


# ---- init / version / errors ----
m("cuInit", "cuxGetApi", reason="초기화는 API 테이블을 얻을 때 함께")
m("cuGetProcAddress cudaGetDriverEntryPoint cudaGetDriverEntryPointByVersion", "cuxGetApi",
  reason="버전 지정 함수 테이블 한 번으로 심볼 해석 대체")
m("cuDriverGetVersion cudaDriverGetVersion cudaRuntimeGetVersion", "cuxGetApi",
  reason="테이블의 버전 필드 + 디바이스 capability 비트")
m("cudaGetErrorString cudaGetErrorName cuGetErrorString cuGetErrorName", "cuxGetErrorInfo")
m("cudaGetLastError cudaPeekAtLastError", "", "removed",
  "프로세스 전역 last-error 없음. 모든 호출이 상태를 반환하고, 비동기 에러는 cuxQueueQuery로 큐별 조회")
m("cuLogsRegisterCallback", "cuxSetLogCallback")

# ---- devices ----
m("cuDeviceGetCount cuDeviceGet cudaGetDeviceCount", "cuxDeviceList")
m("cudaGetDeviceProperties cudaDeviceGetAttribute cuDeviceGetAttribute cuDeviceGetName cuDeviceTotalMem "
  "cuDeviceGetPCIBusId cudaDeviceGetPCIBusId cuDeviceGetUuid_v2", "cuxDeviceQuery",
  reason="정적 속성·capability(VMM, multicast, fabric, RDMA, TMA, cluster)를 구조체 하나로")
m("cudaMemGetInfo cuMemGetInfo", "cuxDeviceMemInfo")
m("cudaDeviceCanAccessPeer", "cuxDevicePeerQuery", reason="접근 가능 여부 + 링크 종류(NVLink/C2C/PCIe)")
m("cudaDeviceEnablePeerAccess", "", "removed", "접근 권한은 매핑마다 cuxMemMap/cuxPoolCreate의 access 목록으로 지정")
m("cudaGetDevice cudaSetDevice", "current-device", "helper", "코어는 모든 호출이 디바이스·큐를 명시. 암묵적 현재 디바이스는 헬퍼만")
m("cudaDeviceReset", "", "removed", "객체 단위 cuxRelease로 정리. 프로세스 전체 리셋 불필요")
m("cudaSetDeviceFlags", "cuxFenceWait", reason="spin/yield/block은 대기 호출의 옵션")

# ---- contexts / partitions ----
m("cuCtxGetCurrent cuCtxSetCurrent cuCtxPushCurrent cuCtxPopCurrent cuCtxGetDevice cuCtxCreate "
  "cuDevicePrimaryCtxRetain cuDevicePrimaryCtxGetState cuCtxGetId", "", "removed",
  "사용자에게 보이는 context 없음. 모든 객체가 디바이스(또는 파티션)에 속함")
m("cuCtxFromGreenCtx", "", "removed", "파티션이 곧 큐를 만드는 단위라 context 변환 불필요")
m("cuDeviceGetDevResource cuDevSmResourceSplitByCount cuDevSmResourceSplit cuDevResourceGenerateDesc "
  "cuGreenCtxCreate cuGreenCtxGetDevResource", "cuxPartitionCreate", reason="SM 수·work queue 설정을 descriptor로")
m("cuGreenCtxDestroy", "cuxRelease")
m("cuGreenCtxStreamCreate", "cuxQueueCreate", reason="descriptor의 partition 필드")

# ---- queues ----
m("cudaStreamCreate cudaStreamCreateWithFlags cudaStreamCreateWithPriority cuStreamCreate "
  "cudaDeviceGetStreamPriorityRange", "cuxQueueCreate", reason="flags·priority·partition을 descriptor로")
m("cudaStreamDestroy cuStreamDestroy", "cuxRelease")
m("cudaStreamQuery cudaStreamGetPriority cudaStreamIsCapturing cudaStreamGetCaptureInfo", "cuxQueueQuery",
  reason="완료 상태·우선순위·캡처 상태·에러를 한 번에")
m("cudaStreamSynchronize cuStreamSynchronize", "cuxFenceWait", reason="큐마다 내장 timeline fence에 대기")
m("cudaDeviceSynchronize cuCtxSynchronize", "device-synchronize", "helper", "디바이스의 모든 큐 fence에 대기하는 헬퍼")

# ---- events / fences / signals ----
m("cudaEventCreate cudaEventCreateWithFlags cuEventCreate", "cuxFenceCreate",
  reason="이벤트를 64-bit timeline fence로 통합 (timing·shareable은 descriptor)")
m("cudaEventDestroy cuEventDestroy", "cuxRelease")
m("cudaEventRecord cudaEventRecordWithFlags cuEventRecord cuStreamWriteValue32", "cuxEnqueueSignal",
  reason="큐에서 fence를 값 N으로 signal. 메모리 값 쓰기도 같은 명령")
m("cudaStreamWaitEvent cuStreamWaitEvent cuStreamWaitValue32", "cuxEnqueueWait", reason="큐가 fence 값 N까지 대기")
m("cudaEventQuery cuEventQuery", "cuxFenceQuery")
m("cudaEventSynchronize cuEventSynchronize", "cuxFenceWait")
m("cudaEventElapsedTime cuEventElapsedTime", "cuxFenceQuery", reason="timing fence는 signal 시각을 함께 기록")
m("cudaIpcGetEventHandle", "cuxExport")
m("cudaIpcOpenEventHandle", "cuxImport")

# ---- memory: VMM core ----
m("cuMemAddressReserve", "cuxAddressReserve")
m("cuMemAddressFree", "cuxRelease")
m("cuMemCreate", "cuxMemCreate")
m("cuMemRelease", "cuxRelease")
m("cuMemMap cuMemSetAccess", "cuxMemMap", reason="매핑과 접근 권한을 한 호출로")
m("cuMemUnmap", "cuxMemUnmap")
m("cuMemGetAllocationGranularity", "cuxDeviceQuery", reason="granularity는 디바이스 속성")
m("cuMulticastGetGranularity", "cuxDeviceQuery", reason="multicast granularity도 디바이스 속성")
m("cuMemExportToShareableHandle cudaIpcGetMemHandle", "cuxExport", reason="legacy IPC와 VMM 공유를 handle type으로 통합")
m("cuMemImportFromShareableHandle cudaIpcOpenMemHandle", "cuxImport")
m("cudaIpcCloseMemHandle", "cuxRelease")
m("cuMemRetainAllocationHandle cuMemGetAllocationPropertiesFromHandle cuMemGetAddressRange cuPointerGetAttribute "
  "cudaPointerGetAttributes cudaHostGetDevicePointer cuMemHostGetDevicePointer", "cuxPointerQuery",
  reason="base·size·location·handle·디바이스 측 주소를 구조체 하나로")
m("cudaHostRegister cuMemHostRegister", "cuxMemCreate", reason="location=host-import (기존 host 메모리를 물리 메모리 객체로)")
m("cudaHostUnregister cuMemHostUnregister", "cuxRelease")
m("cudaMemAdvise", "cuxMemAdvise")
m("cudaMemPrefetchAsync", "cuxEnqueueCopy", reason="dst location만 지정한 migrate 항목")

# ---- memory: pools / allocation ----
m("cudaMalloc cuMemAlloc cudaMallocHost cuMemHostAlloc cudaHostAlloc cudaMallocManaged cudaMallocAsync "
  "cudaMallocFromPoolAsync", "cuxPoolAlloc",
  reason="풀의 location(device/host/managed/mapped)으로 구분, queue=NULL이면 즉시 사용 가능")
m("cudaFree cuMemFree cudaFreeHost cuMemFreeHost cudaFreeAsync", "cuxPoolFree")
m("cudaMemPoolCreate", "cuxPoolCreate")
m("cudaMemPoolDestroy", "cuxRelease")
m("cudaDeviceGetDefaultMemPool cudaDeviceGetMemPool", "cuxDeviceQuery", reason="기본 풀 핸들은 디바이스 정보에 포함")
m("cudaMemPoolSetAttribute cudaMemPoolSetAccess", "cuxPoolUpdate")
m("cudaMemPoolGetAttribute", "cuxPoolQuery")
m("cudaMemPoolTrimTo", "cuxPoolTrim")
m("cudaDeviceGraphMemTrim", "cuxPoolTrim", reason="그래프 안 할당도 풀을 거치므로 같은 trim")
m("cudaDeviceGetGraphMemAttribute cudaDeviceSetGraphMemAttribute", "cuxPoolQuery", reason="그래프 풀 통계")

# ---- copies / fills ----
m("cudaMemcpy cudaMemcpyAsync cudaMemcpy2DAsync cudaMemcpyPeerAsync cudaMemcpyBatchAsync cuMemcpyBatchAsync "
  "cuMemcpyAsync cuMemcpyHtoD cuMemcpyHtoDAsync cuMemcpyDtoH cuMemcpyDtoHAsync cuMemcpyDtoD cudaMemcpyToSymbol",
  "cuxEnqueueCopy", reason="항상 배치. 항목마다 1D/2D·방향 자동 판별. 동기 복사는 헬퍼(enqueue + wait)")
m("cudaMemset cudaMemsetAsync cuMemsetD8 cuMemsetD8Async cuMemsetD32 cuMemsetD32Async", "cuxEnqueueFill",
  reason="항목마다 원소 크기(1/2/4 B) 지정, 배치")

# ---- launch / kernels ----
m("cudaLaunchKernel cudaLaunchKernelEx cudaLaunchKernelExC cudaLaunchCooperativeKernel cuLaunchKernel "
  "cuLaunchKernelEx", "cuxEnqueueLaunch", reason="grid·block·smem·cluster·cooperative·PDL·priority를 launch descriptor로")
m("cudaGridDependencySynchronize cudaTriggerProgrammaticLaunchCompletion", "(device API 유지)", "helper",
  "디바이스 측 API는 커널 언어의 일부라 호스트 ABI 재설계 범위 밖. 그대로 유지")
m("cudaLaunchHostFunc cudaLaunchHostFunc_v2 cudaStreamAddCallback cuLaunchHostFunc", "cuxEnqueueHostFn")
m("cuModuleLoadData cuModuleLoadDataEx cuModuleLoad cuLibraryLoadData cudaLibraryLoadData", "cuxLibraryLoad",
  reason="context 독립 library 하나로. 파일·메모리·JIT 옵션은 descriptor")
m("cuModuleUnload cuLibraryUnload", "cuxRelease")
m("cuModuleGetFunction cuLibraryGetKernel cuLibraryEnumerateKernels cuLibraryGetKernelCount cuKernelGetFunction",
  "cuxLibraryGetKernels", reason="이름 목록 또는 전체 나열")
m("cudaGetKernel", "cuxKernelFromSymbol", reason="nvcc로 함께 빌드한 커널도 같은 kernel 핸들로")
m("cudaFuncGetAttributes cuFuncGetAttribute cuKernelGetName cuFuncGetName", "cuxKernelQuery")
m("cudaFuncSetAttribute cuFuncSetAttribute cuKernelSetAttribute cuFuncSetCacheConfig", "cuxKernelSetAttr")
m("cudaOccupancyMaxActiveBlocksPerMultiprocessor cudaOccupancyMaxPotentialBlockSize "
  "cudaOccupancyAvailableDynamicSMemPerBlock cudaOccupancyMaxActiveClusters cuOccupancyMaxPotentialBlockSize "
  "cuOccupancyMaxPotentialClusterSize cuOccupancyMaxActiveBlocksPerMultiprocessor", "cuxKernelQuery",
  reason="launch descriptor를 넘기면 occupancy 결과를 함께 계산")
m("cuLibraryGetGlobal cuModuleGetGlobal cudaGetSymbolAddress", "cuxGetGlobal")
m("cuTensorMapEncodeTiled", "cuxTensorMapEncode")

# ---- graphs ----
m("cudaStreamBeginCapture cudaStreamBeginCaptureToGraph", "cuxCaptureBegin",
  reason="캡처 대상 그래프(새 그래프 또는 조건 노드 본문)를 descriptor로")
m("cudaStreamEndCapture cudaStreamUpdateCaptureDependencies", "cuxCaptureEnd")
m("cudaThreadExchangeStreamCaptureMode", "", "removed",
  "캡처는 캡처 중인 큐에만 영향. 다른 스레드의 할당은 풀이 캡처 안전하게 처리")
m("cudaGraphCreate", "cuxGraphCreate")
m("cudaGraphAddKernelNode cudaGraphAddEmptyNode cudaGraphAddChildGraphNode cudaGraphAddMemAllocNode cudaGraphAddNode "
  "cudaGraphAddDependencies cudaGraphConditionalHandleCreate cuGraphAddKernelNode", "cuxGraphAddNode",
  reason="노드 종류는 descriptor의 type, 의존성은 인자")
m("cudaGraphGetNodes cudaGraphGetEdges cudaGraphGetRootNodes cudaGraphNodeGetDependencies "
  "cudaGraphNodeGetDependentNodes cuGraphGetNodes cuGraphGetEdges", "cuxGraphInspect")
m("cudaGraphNodeGetType cudaGraphKernelNodeGetParams cudaGraphKernelNodeGetAttribute cudaGraphHostNodeGetParams "
  "cudaGraphMemAllocNodeGetParams cudaGraphMemFreeNodeGetParams cudaGraphEventRecordNodeGetEvent "
  "cudaGraphEventWaitNodeGetEvent cudaGraphChildGraphNodeGetGraph cudaGraphNodeGetToolsId cudaGraphGetId "
  "cuGraphNodeGetType cuGraphKernelNodeGetParams cuGraphKernelNodeGetAttribute cuGraphMemcpyNodeGetParams "
  "cuGraphMemsetNodeGetParams cuGraphChildGraphNodeGetGraph", "cuxGraphNodeQuery",
  reason="노드 종류별 getter 대신 descriptor 하나를 돌려줌")
m("cudaGraphKernelNodeSetAttribute cuGraphKernelNodeSetAttribute", "cuxGraphUpdate",
  reason="노드 descriptor를 바꾼 그래프로 exec 갱신")
m("cudaGraphInstantiate cudaGraphInstantiateWithFlags", "cuxGraphInstantiate")
m("cudaGraphExecGetId", "cuxGraphNodeQuery", reason="exec·graph id는 조회 구조체의 필드")
m("cudaGraphExecUpdate", "cuxGraphUpdate")
m("cudaGraphLaunch", "cuxEnqueueGraph")
m("cudaGraphDestroy cudaGraphExecDestroy", "cuxRelease")
m("cudaUserObjectCreate cudaGraphRetainUserObject", "cuxGraphRetain")
m("cudaUserObjectRelease", "cuxRelease")
m("cudaGraphDebugDotPrint", "graph-dot", "helper", "cuxGraphInspect 결과로 헬퍼가 DOT 생성")

# ---- fabric ----
m("cuMulticastCreate cuMulticastAddDevice", "cuxMulticastCreate", reason="참여 디바이스 목록을 descriptor로")
m("cuMulticastBindMem cuMulticastUnbind", "cuxMulticastBind", reason="mem=NULL이면 unbind")
m("cuLogicalEndpointIdReserve cuLogicalEndpointCreate cuLogicalEndpointBindMem", "cuxEndpointCreate",
  reason="ID 예약·생성·버퍼 바인딩을 descriptor 하나로")
m("cuLogicalEndpointIdRelease cuLogicalEndpointDestroy cuLogicalEndpointUnbind", "cuxRelease")
m("cuLogicalEndpointExport", "cuxExport")
m("cuLogicalEndpointImport", "cuxImport")
m("cuLogicalEndpointQuery", "cuxEndpointQuery")

# ---- tools ----
m("cudaProfilerStart cudaProfilerStop", "", "removed", "도구 제어는 런타임 ABI가 아니라 도구 API(CUPTI·NVTX) 몫")


def main():
    apis = [r["api"] for r in csv.DictReader(open(META))]
    missing = sorted(set(apis) - M.keys())
    extra = sorted(M.keys() - set(apis))
    if missing or extra:
        sys.exit(f"unmapped APIs: {missing}\nmapped but unknown: {extra}")
    unused_core = sorted(set(CORE) - {v[0] for v in M.values()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["api", "new_api", "group", "change", "reason"])
        for a in sorted(M, key=str.lower):
            new, change, reason = M[a]
            w.writerow([a, new, CORE.get(new, ""), change, reason])
    ch = Counter(v[1] for v in M.values())
    per = Counter(v[0] for v in M.values() if v[1] == "merged")
    print(f"current APIs: {len(M)} | core API: {len(CORE)} | " + ", ".join(f"{k} {n}" for k, n in sorted(ch.items())))
    print("core functions not mapped from any current API (new):", unused_core)
    print("biggest merges:", per.most_common(12))
    groups = Counter(CORE.values())
    print("core by group:", dict(groups))


if __name__ == "__main__":
    main()
