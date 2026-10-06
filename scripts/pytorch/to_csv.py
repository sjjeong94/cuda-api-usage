"""Turn pt_scan.json into pytorch rows for api_usage.csv (+ a location dump for writing the evidence doc).

usage: python3 -I scripts/pytorch/to_csv.py <scan.json> docs/reference/data/api_usage.csv <locations_out.json>
"""
import csv
import json
import sys
from collections import defaultdict

scan = json.load(open(sys.argv[1]))
usage_path, loc_out = sys.argv[2], sys.argv[3]

EXCLUDE_FILES = (
    "torch/_inductor/codegen/cutlass/lib_extensions/cutlass_mock_imports/",  # mock modules for codegen
)
# LazyNVRTC.cpp defines lazy stubs (definitions, not calls); real calls go through getNVRTC().
EXCLUDE_LOCS = {"aten/src/ATen/cuda/detail/LazyNVRTC.cpp"}
# string literals that are generated C++ (AOTInductor / cpp_wrapper)
GEN_FILES = {"torch/_inductor/codegen/cuda/device_op_overrides.py", "torch/_inductor/codegen/cpp_wrapper_gpu.py"}
GEN_CONDITIONAL = {"cuModuleLoad", "cuCtxSynchronize"}   # file-path load mode / debug sync only
GEN_HIP_ONLY = {"cudaDeviceSynchronize"}                 # ROCm debug sync in cpp_wrapper_gpu


def comp_of(path):
    p = path
    if p.startswith("c10/"):
        return "c10"
    if p.startswith("aten/"):
        return "aten"
    if p.startswith(("torch/csrc/distributed/", "torch/distributed/")):
        return "c10d"
    if p.startswith(("torch/csrc/inductor/", "torch/_inductor/")):
        return "inductor"
    if p.startswith("torch/csrc/jit/"):
        return "jit"
    if p.startswith(("torch/csrc/profiler", "torch/profiler/")):
        return "profiler"
    if p.startswith("torch/nativert/"):
        return "nativert"
    return "torch.cuda"


TEST_HINTS = ("/test/", "/tests/", "_test.", "/benchmark", "c10/cuda/impl/CUDATest.cpp")

rows = set()
locs = defaultdict(lambda: defaultdict(set))  # api -> component -> files
for api, comps in scan["calls"].items():
    for _, entries in comps.items():
        for e in entries:
            path = e.rsplit(":", 1)[0]
            if path.startswith(EXCLUDE_FILES) or path in EXCLUDE_LOCS:
                continue
            scope = "test" if any(h in path for h in TEST_HINTS) else "prod"
            c = comp_of(path)
            rows.add(("pytorch", c, api, scope))
            locs[api][c + ("" if scope == "prod" else " (test)")].add(e)

for api, comps in scan["strings"].items():
    for _, files in comps.items():
        for path in files:
            if path not in GEN_FILES:
                continue
            scope = "conditional" if api in GEN_CONDITIONAL else "hip-only" if api in GEN_HIP_ONLY else "prod"
            rows.add(("pytorch", "inductor-gen", api, scope))
            locs[api]["inductor-gen"].add(path)

# a prod row supersedes a test row for the same component
final = {r for r in rows if not (r[3] == "test" and ("pytorch", r[1], r[2], "prod") in rows)}

with open(usage_path, newline="") as fh:
    existing = [r for r in csv.reader(fh)]
header, body = existing[0], [r for r in existing[1:] if r[0] != "pytorch"]
body += sorted(map(list, final), key=lambda r: (r[1], r[2].lower(), r[3]))
with open(usage_path, "w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(header)
    w.writerows(body)

json.dump({a: {c: sorted(v) for c, v in cs.items()} for a, cs in sorted(locs.items())},
          open(loc_out, "w"), indent=1)
prod = {r[2] for r in final if r[3] == "prod"}
print("pytorch rows:", len(final), "| prod APIs:", len(prod),
      "| driver:", sum(not a.startswith("cuda") for a in prod),
      "| runtime:", sum(a.startswith("cuda") for a in prod))
for c in sorted({r[1] for r in final}):
    s = {r[2] for r in final if r[1] == c and r[3] == "prod"}
    print(f"  {c}: driver {sum(not a.startswith('cuda') for a in s)}, runtime {sum(a.startswith('cuda') for a in s)}")
