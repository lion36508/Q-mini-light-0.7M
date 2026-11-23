from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


def build_kernel(output=None):
    source = Path(__file__).resolve().parent / "native_kernel.cpp"
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("Install clang++ or g++ to build the native kernel.")
    target = Path(output) if output else source.with_name("libqmini_kernel.so")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    subprocess.run(
        [compiler, "-O3", "-DNDEBUG", "-std=c++17", "-fPIC", "-shared",
         str(source), "-o", str(temporary)],
        check=True,
    )
    temporary.replace(target)
    return target


class NativeKernel:
    def __init__(self, library_path):
        self.path = Path(library_path).resolve()
        self.library = ctypes.CDLL(str(self.path))
        try:
            self.function = self.library.qmini_linear_i8_v2
            self.neon_function = self.library.qmini_kernel_has_neon
        except AttributeError as exc:
            raise RuntimeError(
                "Native library is outdated (Q2-v2 ABI required). Rebuild it with: python -m qmini.native --build"
            ) from exc
        i8 = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
        u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        self.function.argtypes = [
            i8, u8, f32, f32, f32, f32,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.function.restype = ctypes.c_int
        self.neon_function.argtypes = []
        self.neon_function.restype = ctypes.c_int
        self.uses_neon = bool(self.neon_function())

    def linear(
        self, activations_i8, packed_weights, weight_scales, activation_scales,
        bias=None, q2_version: int = 2,
    ):
        raw_x = np.asarray(activations_i8)
        raw_packed = np.asarray(packed_weights)
        if raw_x.dtype != np.int8:
            raise TypeError("activations_i8 must already be int8; silent narrowing is refused.")
        if raw_packed.dtype != np.uint8:
            raise TypeError("packed_weights must already be uint8.")
        x = np.ascontiguousarray(raw_x)
        packed = np.ascontiguousarray(raw_packed)
        ws = np.ascontiguousarray(weight_scales, dtype=np.float32)
        xs = np.ascontiguousarray(activation_scales, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("activations_i8 must have non-empty shape [rows, in_features].")
        if packed.ndim != 2 or ws.ndim != 1 or xs.ndim != 1:
            raise ValueError("Packed weights and scales have invalid ranks.")
        expected_bytes = (x.shape[1] + 3) // 4
        if packed.shape[0] != ws.size or packed.shape[1] != expected_bytes:
            raise ValueError("Packed weight shape does not exactly match activation dimensions.")
        if xs.shape != (x.shape[0],):
            raise ValueError("activation_scales must contain one scale per input row.")
        if (
            isinstance(q2_version, bool)
            or not isinstance(q2_version, int)
            or q2_version not in (1, 2)
        ):
            raise ValueError("q2_version must be the integer 1 or 2.")
        if not np.isfinite(ws).all() or (
            np.any(ws <= 0) if q2_version == 1 else np.any(ws < 0)
        ):
            raise ValueError(
                "weight_scales must be finite and positive for Q2-v1, "
                "or finite and non-negative for Q2-v2."
            )
        if not np.isfinite(xs).all() or np.any(xs <= 0):
            raise ValueError("activation_scales must be finite and positive.")
        if bias is None:
            b = np.zeros(ws.size, dtype=np.float32)
        else:
            b = np.ascontiguousarray(bias, dtype=np.float32)
            if b.shape != (ws.size,):
                raise ValueError("bias must contain one value per output feature.")
            if not np.isfinite(b).all():
                raise ValueError("bias values must be finite.")
        output = np.empty((x.shape[0], ws.size), dtype=np.float32)
        status = self.function(
            x, packed, ws, xs, b, output,
            x.shape[0], x.shape[1], ws.size, int(q2_version)
        )
        if status:
            raise RuntimeError(f"Native kernel returned error {status}.")
        return output


_NATIVE_KERNEL: NativeKernel | None = None
_NATIVE_CACHE_KEY: tuple[str, int, int] | None = None


def configured_library_path() -> Path:
    override = os.environ.get("QMINI_NATIVE_LIBRARY")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().with_name("libqmini_kernel.so")


def get_native_kernel() -> NativeKernel | None:
    """Load/reload the optional native CPU projection when its artifact changes."""
    global _NATIVE_KERNEL, _NATIVE_CACHE_KEY
    path = configured_library_path()
    if not path.is_file():
        _NATIVE_KERNEL = None
        _NATIVE_CACHE_KEY = (str(path), -1, -1)
        return None

    stat = path.stat()
    cache_key = (str(path), int(stat.st_mtime_ns), int(stat.st_size))
    if _NATIVE_CACHE_KEY == cache_key:
        return _NATIVE_KERNEL

    try:
        loaded = NativeKernel(path)
    except (OSError, RuntimeError) as exc:
        _NATIVE_KERNEL = None
        _NATIVE_CACHE_KEY = None
        raise RuntimeError(
            f"Q-Mini native library exists but cannot be loaded: {path}: {exc}"
        ) from exc
    _NATIVE_KERNEL = loaded
    _NATIVE_CACHE_KEY = cache_key
    return _NATIVE_KERNEL


def native_kernel_should_be_used(kernel: NativeKernel | None = None) -> bool:
    """Use NEON by default; make the scalar C++ path an explicit opt-in.

    On common x86 CPUs, optimized PyTorch/BLAS can outperform the scalar
    unpack-and-accumulate loop. Set QMINI_NATIVE_FORCE_SCALAR=1 only when a
    target-specific benchmark shows that scalar native execution is faster.
    """
    active = kernel if kernel is not None else get_native_kernel()
    if active is None:
        return False
    if active.uses_neon:
        return True
    return os.environ.get("QMINI_NATIVE_FORCE_SCALAR", "").strip().lower() in {
        "1", "true", "yes"
    }


def native_kernel_status() -> dict[str, str | bool]:
    path = configured_library_path()
    kernel = get_native_kernel()
    use_native = native_kernel_should_be_used(kernel)
    return {
        "available": kernel is not None,
        "path": str(path),
        "backend": (
            "native-int8-quaternary-neon"
            if use_native and kernel is not None and kernel.uses_neon
            else "native-int8-quaternary-scalar"
            if use_native and kernel is not None
            else "pytorch-fallback"
        ),
        "native_kernel_available": kernel is not None,
        "native_kernel_force_scalar": (
            os.environ.get("QMINI_NATIVE_FORCE_SCALAR", "").strip().lower()
            in {"1", "true", "yes"}
        ),
        "uses_neon": bool(kernel.uses_neon) if kernel is not None else False,
    }


def main():
    parser = argparse.ArgumentParser(description="Build the optional Q-Mini C++ projection primitive.")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.build:
        print(build_kernel(args.output))
    else:
        print(native_kernel_status())


if __name__ == "__main__":
    main()
