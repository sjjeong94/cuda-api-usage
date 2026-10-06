"""Scan a PyTorch source tree for direct CUDA Driver/Runtime API calls.

usage: python3 -I scripts/pytorch/scan.py <pytorch_root> <cuda_python_root> <out_json>
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

root, cp_root, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

# ---- official names from cuda-python declarations ----
official = set()
for pxd in cp_root.rglob("cy*.pxd"):
    if pxd.name not in ("cydriver.pxd", "cyruntime.pxd"):
        continue
    for name in re.findall(r"\b(cu[A-Z]\w*|cuda[A-Z]\w*)\s*\(", pxd.read_text(errors="ignore")):
        official.add(name)
# C++ template / device-side runtime APIs that the bindings do not declare
EXTRA = {
    "cudaLaunchKernelEx", "cudaGridDependencySynchronize", "cudaTriggerProgrammaticLaunchCompletion",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessor", "cudaFuncSetAttribute", "cudaFuncGetAttributes",
    "cudaLaunchCooperativeKernel", "cudaOccupancyMaxPotentialBlockSize", "cudaLaunchKernel",
    "cudaGetKernel", "cudaOccupancyAvailableDynamicSMemPerBlock", "cudaOccupancyMaxActiveClusters",
    "cudaMallocHost", "cudaMallocAsync", "cudaMallocManaged", "cudaMemcpyToSymbol",
    "cudaMemcpyFromSymbol", "cudaGetSymbolAddress", "cudaGraphAddMemcpyNode1D",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
}
official |= EXTRA

# ---- comment / string stripping ----
C_TOKEN = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', re.S)
PY_TOKEN = re.compile(
    r'#[^\n]*|"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\'|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')


def split_code(text, py):
    """Return (code_with_comments_and_strings_blanked, list_of_string_literals)."""
    strings = []

    def repl(m):
        s = m.group(0)
        if s.startswith(("#", "//", "/*")):
            return re.sub(r"[^\n]", " ", s)
        strings.append(s)
        return re.sub(r"[^\n]", " ", s)

    code = (PY_TOKEN if py else C_TOKEN).sub(repl, text)
    return code, strings


def blank_if0(code):
    """Blank out `#if 0 ... #endif` blocks (non-nested approximation with depth tracking)."""
    lines = code.split("\n")
    out, depth = [], 0
    for ln in lines:
        s = ln.strip()
        if depth == 0 and re.match(r"#\s*if\s+0\b", s):
            depth = 1
            out.append("")
            continue
        if depth:
            if re.match(r"#\s*if", s):
                depth += 1
            elif re.match(r"#\s*endif", s):
                depth -= 1
            elif depth == 1 and re.match(r"#\s*(else|elif)", s):
                depth = 0
            out.append("")
            continue
        out.append(ln)
    return "\n".join(out)


CALL = re.compile(r"\b(cu[A-Z]\w*?|cuda[A-Z]\w*?)(_v\d+)?_?\s*\(")
STR_NAME = re.compile(r"\b(cu[A-Z]\w*|cuda[A-Z]\w*)\b")
DECL_LINE = re.compile(r"^\s*(#\s*define|_\(|extern\b|typedef\b|using\b|DECLARE|C10_LIBCUDA|AT_FORALL|CUDA_STUB|NVRTC_STUB)")


def component(rel):
    p = rel.as_posix()
    if re.search(r"(^|/)(test|tests|benchmarks?)(/|$)|_test\.(cpp|cu)$|test_\w+\.py$", p):
        return "test"
    rules = [
        ("c10/cuda/CUDACachingAllocator", "c10-allocator"),
        ("c10/cuda/CUDAAllocatorConfig", "c10-allocator"),
        ("c10/cuda/driver_api", "c10-driver-table"),
        ("c10/cuda/", "c10-cuda"),
        ("aten/src/ATen/cuda/nvrtc_stub", "aten-nvrtc-stub"),
        ("aten/src/ATen/cuda/detail/LazyNVRTC", "aten-nvrtc-stub"),
        ("aten/src/ATen/cuda/CUDAGraph", "aten-cudagraph"),
        ("aten/src/ATen/native/cuda/jit_utils", "aten-jiterator"),
        ("aten/src/ATen/cuda/", "aten-cuda"),
        ("aten/src/ATen/native/transformers/cuda", "aten-kernels"),
        ("aten/src/ATen/native/", "aten-kernels"),
        ("aten/src/ATen/", "aten-other"),
        ("torch/csrc/distributed/c10d/symm_mem", "c10d-symm-mem"),
        ("torch/csrc/distributed/c10d/cuda", "c10d-cuda"),
        ("torch/csrc/distributed/", "c10d"),
        ("torch/csrc/inductor/aoti_runtime", "aoti-runtime"),
        ("torch/csrc/inductor/", "inductor-csrc"),
        ("torch/csrc/cuda/", "torch-csrc-cuda"),
        ("torch/csrc/profiler", "profiler"),
        ("torch/csrc/jit/", "jit"),
        ("torch/csrc/", "torch-csrc-other"),
        ("torch/_inductor/", "inductor-py"),
        ("torch/cuda/", "torch-cuda-py"),
        ("torch/distributed/", "distributed-py"),
        ("torch/utils/", "torch-utils-py"),
        ("torch/", "torch-py-other"),
    ]
    for prefix, name in rules:
        if p.startswith(prefix):
            return name
    return "other"


SRC_EXT = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py"}
calls = defaultdict(lambda: defaultdict(list))      # api -> component -> [file:line]
strings = defaultdict(lambda: defaultdict(list))    # api -> component -> [file:line] (names in string literals)
helpers = defaultdict(int)                          # non-official cu*/cuda* call names
for f in root.rglob("*"):
    if f.suffix not in SRC_EXT or not f.is_file():
        continue
    rel = f.relative_to(root)
    if rel.parts[0] not in ("aten", "c10", "torch") or "hip" in rel.as_posix().lower():
        continue
    text = f.read_text(errors="ignore")
    py = f.suffix == ".py"
    code, lits = split_code(text, py)
    if not py:
        code = blank_if0(code)
    comp = component(rel)
    lines = code.split("\n")
    for i, ln in enumerate(lines, 1):
        if DECL_LINE.match(ln):
            continue
        for m in CALL.finditer(ln):
            name = m.group(1)
            # strip trailing underscore of driver-table members (cuMemCreate_)
            base = name
            if base in official:
                # skip function definitions/declarations in headers: "CUresult cuX(" or "cudaError_t cudaX("
                prefix = ln[: m.start()]
                if re.search(r"\b(CUresult|cudaError_t|CUDAAPI|CUDARTAPI)\s*\*?\s*$", prefix):
                    continue
                calls[base][comp].append(f"{rel}:{i}")
            elif base.endswith("_") and base[:-1] in official:
                calls[base[:-1]][comp].append(f"{rel}:{i}")
            else:
                helpers[base] += 1
    for s in lits:
        for name in STR_NAME.findall(s):
            if name in official:
                strings[name][comp].append(str(rel))

json.dump({
    "official_count": len(official),
    "calls": {a: dict(c) for a, c in calls.items()},
    "strings": {a: dict(c) for a, c in strings.items()},
    "helpers": dict(sorted(helpers.items(), key=lambda x: -x[1])),
}, open(out, "w"), indent=1, ensure_ascii=False)
print("official names:", len(official), "| apis called:", len(calls), "| in strings:", len(strings),
      "| non-official names:", len(helpers))
