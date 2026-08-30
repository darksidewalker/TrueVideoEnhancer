#!/usr/bin/env python3
"""Verification runner for GitHub issue #2 (corrupted refitted TRT upscaler engine).

Reproduces the issue's two-job scenario without the full encode pipeline:

  Phase A: RVE_UPSCALER_TRT_ENGINE_CACHE off   -> fresh compile (correct baseline).
  Phase B: RVE_UPSCALER_TRT_ENGINE_CACHE=1     -> first opt-in build: compile + cache write ("job 1").
  Phase C: RVE_UPSCALER_TRT_ENGINE_CACHE=1     -> cache hit + refit ("job 2", the buggy path).

Each phase runs in its OWN subprocess so Phase C genuinely hits the persistent
disk cache (an in-process second build would use the in-memory compile cache
and never exercise the refit path). A synthetic frame is generated, so no
sample video or GPU-adjacent assets are required.

Usage (from the repo root):
    runtime/venv/bin/python tools/verify_issue2_trt_cache.py [options]
    runtime/venv/bin/python tools/verify_issue2_trt_cache.py > tve-verify.log 2>&1

Options:
    --model PATH      safetensors upscaler (default: models/2x-AnimeSharpV4_RCAN.safetensors)
    --compile-size N  TRT compile edge (default 160 -> 160x160, batch 8, ~1-2 GiB VRAM);
                      the synthetic input frame is exactly this size (static-shape engine)
    --keep-cache      do NOT wipe models/.tensorrt-engine-cache before Phase B
    --workdir DIR     where phase outputs (phase_X.npy) go (default: system temp)

Exit codes:
    0  clean: fresh vs refit outputs agree, no refit warnings, no solid frames
    1  issues found (corruption signature or refit warnings) — matches issue #2
    2  setup failure (missing model / CUDA unavailable)
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

REFIT_MARKER = "not found in weight mapping"
PHASES = ("A", "B", "C")


def make_frame(size: int) -> np.ndarray:
    """Synthetic BGR test frame: gradients, checker quadrant, red band."""
    x = np.linspace(0, 255, size, dtype=np.float32)
    horizontal = np.tile(x, (size, 1)).astype(np.uint8)
    vertical = np.tile(horizontal.T, (1, 1)).astype(np.uint8)
    frame = np.dstack([horizontal, vertical, np.full((size, size), 128, np.uint8)])
    frame[: size // 2, : size // 2] ^= np.dstack([np.zeros((size // 2, size // 2), np.uint8)] * 3)
    frame[-8:, :] = np.array([255, 0, 0], np.uint8)  # red band: easy to spot if corrupted
    return frame


def look_solid(out: np.ndarray) -> bool:
    """The corrupted frames in issue #2 were solid red: near-zero std, R>>G,B."""
    std = float(out.std())
    r, g, b = out.reshape(-1, 3).astype(np.float64).mean(axis=0)
    return std < 5.0 and r > 200 and r > 2.0 * g and r > 2.0 * b


def run_single_phase(phase: str, model: Path, compile_size: int, workdir: Path) -> int:
    """Build the upscaler (env toggle set per phase), upscale one frame, save output.

    The frame is exactly the (static) compile size: a direct upscaler built with
    input_size=(N,N) produces a static-shape TRT engine, so its input must be N x N.
    """
    module = importlib.import_module("upscale_inference")
    enabled = phase in ("B", "C")
    os.environ["RVE_UPSCALER_TRT_ENGINE_CACHE"] = "1" if enabled else "0"
    print(f"[{phase}] RVE_UPSCALER_TRT_ENGINE_CACHE={'1' if enabled else '0'} "
          f"compile={compile_size}x{compile_size}", flush=True)

    start = time.monotonic()
    upscaler = module.create_upscaler(
        model, backend="tensorrt", device="cuda", precision="float16",
        input_size=(compile_size, compile_size),
    )
    build_elapsed = time.monotonic() - start

    out = np.ascontiguousarray(upscaler.upscale_batch([make_frame(compile_size)])[0])
    upscaler.close()
    total_elapsed = time.monotonic() - start

    print(f"[{phase}] build={build_elapsed:.1f}s total={total_elapsed:.1f}s "
          f"out_shape={out.shape} mean={float(out.mean()):.2f} std={float(out.std()):.2f} "
          f"channel_means={out.reshape(-1, 3).mean(axis=0).round(1)}", flush=True)

    workdir.mkdir(parents=True, exist_ok=True)
    np.save(workdir / f"phase_{phase}.npy", out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/2x-AnimeSharpV4_RCAN.safetensors",
                       help="safetensors upscaler (default: the model from issue #2)")
    parser.add_argument("--compile-size", type=int, default=160,
                       help="TRT compile edge in px (160 -> 160x160, batch 8, ~1-2 GiB VRAM); "
                            "the synthetic input frame is exactly this size (static-shape engine)")
    parser.add_argument("--keep-cache", action="store_true",
                        help="do not wipe models/.tensorrt-engine-cache before Phase B")
    parser.add_argument("--workdir", default="", help="where phase_X.npy files go")
    parser.add_argument("--phase", choices=PHASES,
                       help=argparse.SUPPRESS)  # internal: single-phase subprocess mode
    args = parser.parse_args()

    if args.phase:
        model = (ROOT / args.model).resolve()
        if not model.is_file():
            print(f"FATAL: model not found: {model}", file=sys.stderr)
            return 2
        workdir = Path(args.workdir) if args.workdir else Path(tempfile.gettempdir()) / "tve_verify"
        return run_single_phase(args.phase, model, args.compile_size, workdir)

    # ---- orchestration: one subprocess per phase ----
    model = (ROOT / args.model).resolve()
    if not model.is_file():
        print(f"FATAL: model not found: {model}", file=sys.stderr)
        return 2

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.gettempdir()) / "tve_verify"
    cache_dir = model.parent / ".tensorrt-engine-cache"

    if not args.keep_cache:
        if cache_dir.exists():
            print(f"[setup] wiping stale cache {cache_dir}", flush=True)
            shutil.rmtree(cache_dir)
        else:
            print("[setup] no existing cache to wipe", flush=True)

    child_stderr: dict[str, str] = {}
    for phase in PHASES:
        t0 = time.monotonic()
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--phase", phase,
             "--model", args.model, "--compile-size", str(args.compile_size),
             "--workdir", str(workdir)],
            capture_output=True, text=True,
        )
        print(result.stdout, end="")
        if result.stderr:
            print(f"[{phase} stderr]\n{result.stderr}", end="")
        child_stderr[phase] = result.stderr or ""
        if result.returncode != 0:
            print(f"ABORT: phase {phase} failed with exit code {result.returncode}")
            return 2
        print(f"[{phase}] finished in {time.monotonic() - t0:.1f}s (wall, incl. imports)", flush=True)

    out_a = np.load(workdir / "phase_A.npy")
    out_c = np.load(workdir / "phase_C.npy")

    print("\n=== verdict ===")
    problems: list[str] = []

    cache_entries = sorted(p.name for p in cache_dir.iterdir()) if cache_dir.exists() else []
    print(f"cache entries: {len(cache_entries)}")
    if not cache_entries:
        problems.append("cache dir still empty after opt-in build — cache path not active")

    diff_ac = float(np.abs(out_a.astype(np.int32) - out_c.astype(np.int32)).mean())
    print(f"mean|A-C| = {diff_ac:.2f} (0.0-2.0 = clean refit; larger = stale-weight corruption)")
    if diff_ac > 2.0:
        problems.append(f"fresh vs refit outputs differ (mean|A-C|={diff_ac:.2f})")

    for phase in PHASES:
        out = np.load(workdir / f"phase_{phase}.npy")
        if look_solid(out):
            problems.append(f"phase {phase} output looks solid-red/corrupted")

    refit_hits = [(p, [ln for ln in child_stderr[p].splitlines() if REFIT_MARKER in ln])
                  for p in PHASES]
    refit_hits = [(p, hits) for p, hits in refit_hits if hits]
    if refit_hits:
        for phase, hits in refit_hits:
            print(f"refit warnings in phase {phase} ({len(hits)}):")
            for line in hits[:10]:
                print(f"  - {line}")
        problems.append("torch_tensorrt refit 'not found in weight mapping' warnings present")
    else:
        print("refit warnings: none")

    if problems:
        print("\nRESULT: ISSUES FOUND — matches the issue #2 failure mode (review above)")
        return 1
    print("\nRESULT: OK — fresh vs refit outputs agree, no refit warnings, no solid frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
