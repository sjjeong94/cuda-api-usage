#!/usr/bin/env python3
"""Generate docs/reference/api-catalog.md from docs/reference/data/*.csv.

Usage: python3 scripts/gen_api_catalog.py
"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "reference" / "data"
OUT = ROOT / "docs" / "reference" / "api-catalog.md"

FRAMEWORKS = [
    ("vllm", "vLLM"),
    ("sglang", "SGLang"),
    ("trtllm", "TRT-LLM"),
    ("ollama", "Ollama"),
    ("executorch", "ExecuTorch"),
    ("pytorch", "PyTorch"),
]
# PyTorch is the dependency of the [PT] frameworks, not a serving framework; its many
# components go in a separate column instead of being packed into its usage cell.
DEPENDENCY = "pytorch"
COMPONENT_LABEL = {
    ("ollama", "ollama"): "O",
    ("ollama", "llama.cpp"): "L",
    ("ollama", "mlx"): "M",
    ("executorch", "runtime"): "E",
    ("executorch", "aoti"): "A",
}
SCOPE_MARK = {"prod": "●", "test": "t", "conditional": "c", "hip-only": "h"}
TIER_ORDER = ["T0", "T1", "T2", "T3", "AUX"]
DEVICE_APIS = {"cudaGridDependencySynchronize", "cudaTriggerProgrammaticLaunchCompletion"}


def read(name):
    with open(DATA / name, newline="") as fh:
        return list(csv.DictReader(fh))


def kind(api):
    if api in DEVICE_APIS:
        return "device"
    return "runtime" if api.startswith("cuda") else "driver"


def main():
    usage = read("api_usage.csv")
    meta = {r["api"]: r for r in read("api_meta.csv")}
    features = {r["feature"]: r for r in read("features.csv")}

    for r in usage:
        if r["api"] not in meta:
            raise SystemExit(f"api_meta.csv is missing {r['api']}")

    def api_tier(api):
        tiers = [features[f]["tier"] for f in meta[api]["features"].split(";")]
        return min(tiers, key=TIER_ORDER.index)

    # usage[api][fw] -> list of (component, scope)
    by_api = defaultdict(lambda: defaultdict(list))
    for r in usage:
        by_api[r["api"]][r["framework"]].append((r["component"], r["scope"]))

    def cell(api, fw):
        entries = by_api[api].get(fw, [])
        if not entries:
            return ""
        if fw == DEPENDENCY:
            scopes = {scope for _, scope in entries}
            return next(SCOPE_MARK[s] for s in SCOPE_MARK if s in scopes)
        parts = []
        for comp, scope in entries:
            label = COMPONENT_LABEL.get((fw, comp), "")
            mark = SCOPE_MARK[scope]
            parts.append(f"{label}{mark}" if label else mark)
        # a single framework-wide prod/test mark needs no component label
        return " ".join(dict.fromkeys(parts))

    def used_in_prod(api, fw):
        return any(scope == "prod" for _, scope in by_api[api].get(fw, []))

    # features.csv lists frameworks by hand (shared APIs such as the VMM set would
    # otherwise credit every VMM user with every VMM-based feature); check each claim.
    fw_key = {label: fw for fw, label in FRAMEWORKS}
    for fid, f in features.items():
        apis = [a for a in meta if fid in meta[a]["features"].split(";")]
        for label in filter(None, f["frameworks"].split(";")):
            if not any(used_in_prod(a, fw_key[label]) for a in apis):
                raise SystemExit(f"features.csv: {fid} claims {label}, but no tagged API is used there")
        both = set(filter(None, f["frameworks"].split(";"))) & set(filter(None, f["delegated"].split(";")))
        if both:
            raise SystemExit(f"features.csv: {fid} lists {sorted(both)} as both implementing and delegating")

    lines = []
    w = lines.append
    w("# CUDA API 카탈로그")
    w("")
    w("> 이 파일은 `scripts/gen_api_catalog.py`가 `docs/reference/data/*.csv`에서 생성합니다. 직접 고치지 말고 CSV를 고친 뒤 다시 생성하세요.")
    w("")
    w("표기: ● 본체 코드에서 사용, t 테스트·CI·벤치마크에서만 사용, c 설정에 따라 사용, h HIP(ROCm) 빌드 전용.")
    w("Ollama의 O/L/M은 Ollama 본체/llama.cpp/MLX, ExecuTorch의 E/A는 ExecuTorch 런타임/AOTI 생성 코드입니다.")
    w("PyTorch는 vLLM·SGLang·TRT-LLM이 의존하는 라이브러리라서 함께 실었습니다. PyTorch 열이 ●이면 본체 코드 어딘가에서 쓰고, 어느 구성 요소인지는 `PyTorch 위치` 열에 있습니다.")
    w("등급은 그 API가 쓰이는 기능 중 가장 낮은 등급입니다. 역할 `alt`는 같은 기능을 하는 다른 계층(Driver↔Runtime)의 API가 기본이라는 뜻입니다.")
    w("")

    # ---- 1. counts ----
    w("## 1. 프레임워크별 집계 (본체 코드, 중복 제외)")
    w("")
    w("| 프레임워크 | Driver | Runtime (호스트) | Runtime (디바이스 측) |")
    w("|---|---:|---:|---:|")
    for fw, label in FRAMEWORKS:
        comps = sorted({r["component"] for r in usage if r["framework"] == fw})
        rows = [(label, None)] + ([(f"└ {c}", c) for c in comps] if len(comps) > 1 else [])
        for row_label, comp in rows:
            apis = {
                r["api"]
                for r in usage
                if r["framework"] == fw and r["scope"] == "prod" and (comp is None or r["component"] == comp)
            }
            counts = {k: sum(1 for a in apis if kind(a) == k) for k in ("driver", "runtime", "device")}
            w(f"| {row_label} | {counts['driver']} | {counts['runtime']} | {counts['device']} |")
    w("")

    # ---- 2. tier summary ----
    w("## 2. 등급별 API 수")
    w("")
    w("| 등급 | Driver | Runtime (호스트) | Runtime (디바이스 측) | 합계 |")
    w("|---|---:|---:|---:|---:|")
    for t in TIER_ORDER:
        apis = [a for a in meta if api_tier(a) == t]
        c = {k: sum(1 for a in apis if kind(a) == k) for k in ("driver", "runtime", "device")}
        w(f"| {t} | {c['driver']} | {c['runtime']} | {c['device']} | {len(apis)} |")
    w(f"| 합계 | {sum(kind(a) == 'driver' for a in meta)} | {sum(kind(a) == 'runtime' for a in meta)} "
      f"| {sum(kind(a) == 'device' for a in meta)} | {len(meta)} |")
    w("")

    # ---- 3. features ----
    w("## 3. 기능별 API")
    w("")
    w("`직접 구현`은 그 기능을 자기 코드에서 CUDA API로 구현한 프레임워크, `위임`은 PyTorch 등에 맡기거나 호출자에게 넘기는 프레임워크입니다.")
    w("")
    w("| 기능 | 등급 | 기본 API | 다른 계층 대응 API | 직접 구현 | 위임 | 가이드 |")
    w("|---|:---:|---|---|---|---|---|")
    for fid, f in features.items():
        apis = sorted((a for a in meta if fid in meta[a]["features"].split(";")), key=str.lower)
        prim = ", ".join(f"`{a}`" for a in apis if meta[a]["role"] == "primary")
        alt = ", ".join(f"`{a}`" for a in apis if meta[a]["role"] == "alt")
        fws = ", ".join(filter(None, f["frameworks"].split(";")))
        dele = ", ".join(filter(None, f["delegated"].split(";")))
        guide = f"[{f['guide']}](../guide/{f['guide']})"
        w(f"| **{fid}** {f['name']} | {f['tier']} | {prim} | {alt} | {fws} | {dele} | {guide} |")
    w("")

    # ---- 4. all APIs ----
    header = ["API", "종류", "등급", "기능", "역할"] + [label for _, label in FRAMEWORKS] + ["PyTorch 위치", "최소 CUDA", "하드웨어", "비고"]
    for k, title in (("driver", "4. Driver API"), ("runtime", "5. Runtime API (호스트)"),
                     ("device", "6. Runtime API (디바이스 측)")):
        w(f"## {title}")
        w("")
        w("| " + " | ".join(header) + " |")
        w("|" + "|".join(["---"] * 5 + [":---:"] * len(FRAMEWORKS) + ["---"] * 4) + "|")
        apis = sorted((a for a in meta if kind(a) == k), key=lambda a: (TIER_ORDER.index(api_tier(a)), a.lower()))
        for a in apis:
            mrow = meta[a]
            feats = ", ".join(mrow["features"].split(";"))
            cells = [cell(a, fw) for fw, _ in FRAMEWORKS]
            pt_where = ", ".join(sorted({c for c, scope in by_api[a].get(DEPENDENCY, []) if scope == "prod"}))
            w("| " + " | ".join([f"`{a}`", k, api_tier(a), feats, mrow["role"], *cells, pt_where,
                                 mrow["min_cuda"], mrow["arch"], mrow["note"]]) + " |")
        w("")

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT.relative_to(ROOT)}")
    print("\n".join(lines[lines.index("## 1. 프레임워크별 집계 (본체 코드, 중복 제외)"):lines.index("## 3. 기능별 API")]))


if __name__ == "__main__":
    main()
